"""
mini_cgcnn_v2.py
================

修复版 MiniCGCNN(multi-task),替代原 mini_cgcnn.py.

修复内容(三处):
  1. ★ target_scaler 改为每折内 fit(消除 data leakage)
  2. ★ 增加 voltage parity 数据 dump(原版缺失,与 RF/Roost 不对齐)
  3. ★ 移除 cgcnn_metrics.json 的重复保存(原版第 542 + 546 行)

不变(继续保留):
  - 整体架构: Embed -> RBF -> 3×GatedConv -> MaskedMeanPool -> MLP
  - N_MAX=100, K=12, cutoff=8 Å
  - AdamW lr=1e-3, weight_decay=1e-4, batch=16, epochs=200, patience=30
  - 5 折 CV,L1 loss
  - 纯 numpy 邻居搜索(规避 pymatgen Cython bug)

依赖:
  pip install torch  (CPU 即可)

输出:
  - mp_baseline/cgcnn_metrics_v2.json
  - mp_baseline/cgcnn_parity_plots_v2.png
  - mp_baseline/cgcnn_vs_baselines_v2.png
  - mp_baseline/cgcnn_parity_delta_volume_v2.npz   ← 给 fig4 用
  - mp_baseline/cgcnn_parity_voltage_v2.npz        ★ 新增, 给 fig3 用
  - mp_baseline/cgcnn_parity_capacity_v2.npz       ★ 新增
"""

import warnings
warnings.filterwarnings("ignore")

import itertools as _it
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from utils import log, compute_metrics, save_json


# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "mp_baseline"
OUTPUT_DIR.mkdir(exist_ok=True)


TARGETS = ["average_voltage", "capacity_grav", "max_delta_volume"]
TARGET_LABELS = {
    "average_voltage": "Voltage (V)",
    "capacity_grav": "Capacity (mAh/g)",
    "max_delta_volume": "Volume change",
}

# 图配置
N_MAX = 100
K = 12
CUTOFF = 8.0

# 模型配置
ATOM_DIM = 64
EDGE_DIM = 40
N_CONV = 3
MLP_HIDDEN = 128
DROPOUT = 0.1

# 训练配置
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
EPOCHS = 200
PATIENCE = 30
N_SPLITS = 5
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. 晶体 -> 图 (与原版完全一致)
# ============================================================
def crystal_to_graph(struct, n_max=N_MAX, k=K, cutoff=CUTOFF):
    n = len(struct)
    if n > n_max:
        return None

    cart = np.asarray(struct.cart_coords, dtype=np.float64)
    lattice = np.asarray(struct.lattice.matrix, dtype=np.float64)
    lattice_lengths = np.linalg.norm(lattice, axis=1)
    n_imgs = np.ceil(cutoff / lattice_lengths).astype(int) + 2

    image_grid = list(_it.product(
        range(-n_imgs[0], n_imgs[0] + 1),
        range(-n_imgs[1], n_imgs[1] + 1),
        range(-n_imgs[2], n_imgs[2] + 1),
    ))
    image_offsets = np.asarray(image_grid, dtype=np.float64) @ lattice
    M = len(image_offsets)
    all_pos = (cart[:, None, :] + image_offsets[None, :, :]).reshape(-1, 3)
    atom_indices_flat = np.repeat(np.arange(n), M)

    atom_z = np.zeros(n_max, dtype=np.int64)
    neighbor_idx = np.zeros((n_max, k), dtype=np.int64)
    neighbor_dist = np.full((n_max, k), cutoff, dtype=np.float32)
    atom_mask = np.zeros(n_max, dtype=np.float32)

    for i in range(n):
        atom_z[i] = struct[i].specie.Z
        atom_mask[i] = 1.0

        diffs = all_pos - cart[i]
        dists = np.linalg.norm(diffs, axis=1)
        mask = (dists > 0.01) & (dists < cutoff)

        if not mask.any():
            neighbor_idx[i, :] = i
            continue

        nbrs_i = atom_indices_flat[mask]
        dists_i = dists[mask]
        order = np.argsort(dists_i)[:k]
        sorted_nbrs = nbrs_i[order]
        sorted_dists = dists_i[order]

        n_real = len(sorted_nbrs)
        neighbor_idx[i, :n_real] = sorted_nbrs
        neighbor_dist[i, :n_real] = sorted_dists
        if n_real < k:
            neighbor_idx[i, n_real:] = i

    return atom_z, neighbor_idx, neighbor_dist, atom_mask


# ============================================================
# 2. Dataset
# ============================================================
class CrystalDataset(Dataset):
    def __init__(self, df, structures, targets):
        self.items = []
        skipped = 0
        for _, row in df.iterrows():
            bid = row["battery_id"]
            if bid not in structures:
                continue
            graph = crystal_to_graph(structures[bid])
            if graph is None:
                skipped += 1
                continue
            atom_z, nbr_idx, nbr_dist, mask = graph
            self.items.append({
                "atom_z": atom_z,
                "neighbor_idx": nbr_idx,
                "neighbor_dist": nbr_dist,
                "atom_mask": mask,
                "targets": np.array([row[t] for t in targets], dtype=np.float32),
                "battery_id": bid,
            })
        self._skipped = skipped

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        return (torch.from_numpy(it["atom_z"]),
                torch.from_numpy(it["neighbor_idx"]),
                torch.from_numpy(it["neighbor_dist"]),
                torch.from_numpy(it["atom_mask"]),
                torch.from_numpy(it["targets"]))


# ============================================================
# 3. 模型组件 (与原版完全一致)
# ============================================================
class RBFExpansion(nn.Module):
    """
    高斯径向基展开.

    数学形式: e_k = exp(-gamma * (d - mu_k)^2)
    含义:     gamma 是 inverse-bandwidth squared, gamma = 1/(2 sigma^2)
              gamma=10 Å^-2 -> sigma = 1/sqrt(20) Å ≈ 0.224 Å
              40 个中心在 [0, 8 Å] 等距 -> 中心间距 0.205 Å
              所以高斯宽度 ≈ 中心间距, 标准 CGCNN 风格的合理设置.
    """
    def __init__(self, n_centers=EDGE_DIM, cutoff=CUTOFF, gamma=10.0):
        super().__init__()
        centers = torch.linspace(0, cutoff, n_centers)
        self.register_buffer("centers", centers)
        self.gamma = gamma

    def forward(self, dist):
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers).pow(2))


class CGCNNConv(nn.Module):
    def __init__(self, atom_dim, edge_dim):
        super().__init__()
        in_dim = 2 * atom_dim + edge_dim
        self.W_f = nn.Linear(in_dim, atom_dim)
        self.W_s = nn.Linear(in_dim, atom_dim)
        self.bn = nn.BatchNorm1d(atom_dim)

    def forward(self, v, neighbor_idx, edge_feat, atom_mask):
        B, N, D = v.shape
        K_ = neighbor_idx.size(2)

        batch_idx = torch.arange(B, device=v.device).view(B, 1, 1).expand(B, N, K_)
        v_j = v[batch_idx, neighbor_idx]
        v_i = v.unsqueeze(2).expand(-1, -1, K_, -1)

        z = torch.cat([v_i, v_j, edge_feat], dim=-1)
        gate = torch.sigmoid(self.W_f(z))
        msg = F.softplus(self.W_s(z))
        agg = (gate * msg).sum(dim=2)
        agg = self.bn(agg.reshape(-1, D)).view(B, N, D)
        out = (v + agg) * atom_mask.unsqueeze(-1)
        return out


class MiniCGCNN(nn.Module):
    def __init__(self, atom_dim=ATOM_DIM, edge_dim=EDGE_DIM,
                 n_conv=N_CONV, mlp_hidden=MLP_HIDDEN,
                 n_targets=3, dropout=DROPOUT):
        super().__init__()
        self.atom_emb = nn.Embedding(119, atom_dim, padding_idx=0)
        self.rbf = RBFExpansion(n_centers=edge_dim, cutoff=CUTOFF)
        self.convs = nn.ModuleList([
            CGCNNConv(atom_dim, edge_dim) for _ in range(n_conv)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(atom_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_targets),
        )

    def forward(self, atom_z, neighbor_idx, neighbor_dist, atom_mask):
        v = self.atom_emb(atom_z)
        e = self.rbf(neighbor_dist)
        for conv in self.convs:
            v = conv(v, neighbor_idx, e, atom_mask)
        mask = atom_mask.unsqueeze(-1)
        crystal_feat = (v * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        return self.mlp(crystal_feat)


# ============================================================
# 4. 训练单折 (★ 接收每折的 target_scaler)
# ============================================================
def train_one_fold(train_ds, val_ds, target_scaler):
    """
    ★ 关键修复: target_scaler 是 per-fold scaler, 由外部传入.
       不再在内部对全数据集 fit.
    """
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    model = MiniCGCNN(n_targets=len(TARGETS)).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR,
                              weight_decay=WEIGHT_DECAY)
    loss_fn = nn.L1Loss()

    best_val = float("inf")
    best_state = None
    patience_cnt = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            atom_z, nbr_idx, nbr_dist, mask, y = [x.to(DEVICE) for x in batch]
            y_norm = torch.from_numpy(
                target_scaler.transform(y.cpu().numpy())
            ).float().to(DEVICE)
            pred = model(atom_z, nbr_idx, nbr_dist, mask)
            loss = loss_fn(pred, y_norm)
            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        val_loss, n_count = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                atom_z, nbr_idx, nbr_dist, mask, y = [x.to(DEVICE) for x in batch]
                y_norm = torch.from_numpy(
                    target_scaler.transform(y.cpu().numpy())
                ).float().to(DEVICE)
                pred = model(atom_z, nbr_idx, nbr_dist, mask)
                val_loss += loss_fn(pred, y_norm).item() * len(y)
                n_count += len(y)
        val_loss /= max(n_count, 1)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            atom_z, nbr_idx, nbr_dist, mask, y = [x.to(DEVICE) for x in batch]
            pred_norm = model(atom_z, nbr_idx, nbr_dist, mask).cpu().numpy()
            preds.append(target_scaler.inverse_transform(pred_norm))
            trues.append(y.numpy())
    return np.vstack(trues), np.vstack(preds), epoch + 1


# ============================================================
# 5. CV
# ============================================================
def cross_validate(df, structures):
    log(f"开始 {N_SPLITS} 折交叉验证(per-fold target scaler)...")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    all_true = {t: [] for t in TARGETS}
    all_pred = {t: [] for t in TARGETS}

    for fold, (tr_idx, val_idx) in enumerate(kf.split(df), 1):
        log(f"  Fold {fold}/{N_SPLITS} ...")
        tr_df = df.iloc[tr_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # ★ 关键: 在本折训练集上 fit target_scaler
        target_scaler = StandardScaler()
        target_scaler.fit(tr_df[TARGETS].values)

        t_fold = time.time()
        tr_ds = CrystalDataset(tr_df, structures, TARGETS)
        val_ds = CrystalDataset(val_df, structures, TARGETS)
        log(f"    train n={len(tr_ds)} (skipped {tr_ds._skipped}), "
            f"val n={len(val_ds)} (skipped {val_ds._skipped})")

        torch.manual_seed(SEED + fold)
        trues, preds, n_epochs = train_one_fold(tr_ds, val_ds, target_scaler)
        dt = time.time() - t_fold
        log(f"    stopped at epoch {n_epochs}, fold time: {dt:.1f}s")

        for i, t in enumerate(TARGETS):
            all_true[t].append(trues[:, i])
            all_pred[t].append(preds[:, i])

    metrics = {}
    predictions = {}
    for t in TARGETS:
        y_true = np.concatenate(all_true[t])
        y_pred = np.concatenate(all_pred[t])
        m = compute_metrics(y_true, y_pred)
        metrics[t] = m
        predictions[t] = (y_true, y_pred)
        log(f"  {t:20s}  MAE={m['MAE']:.3f}  R²={m['R2']:.3f}  MAE/σ={m['MAE_over_std']:.3f}")

    return metrics, predictions


# ============================================================
# 6. 数据加载
# ============================================================
def load_data():
    log("加载已清洗 csv 与结构 pkl...")
    df = pd.read_csv(OUTPUT_DIR / "na_electrodes.csv")
    n0 = len(df)
    df = df[df["average_voltage"] > 0]
    df = df[df["stability_discharge"] < 0.1]
    df = df.dropna(subset=TARGETS + ["formula_discharge"])
    df = df[df["max_delta_volume"].abs() < 1.0]
    df = df.reset_index(drop=True)
    log(f"  csv 清洗: {n0} -> {len(df)}")

    with open(OUTPUT_DIR / "na_structures.pkl", "rb") as f:
        structures = pickle.load(f)
    log(f"  加载结构: {len(structures)} 条")

    df = df[df["battery_id"].isin(structures)].reset_index(drop=True)
    log(f"  最终数据集: {len(df)} 条")
    return df, structures


# ============================================================
# 7. 可视化
# ============================================================
def plot_parity(predictions, metrics):
    log("绘制 parity 图...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, target in zip(axes, TARGETS):
        y_true, y_pred = predictions[target]
        ax.scatter(y_true, y_pred, alpha=0.4, s=15, edgecolor="none",
                   color="C3")
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"True {TARGET_LABELS[target]}")
        ax.set_ylabel(f"Predicted {TARGET_LABELS[target]}")
        m = metrics[target]
        ax.set_title(f"{target}\nMAE={m['MAE']:.3f}  R²={m['R2']:.3f}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("MiniCGCNN v2 (5-fold CV, per-fold scaler)", y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "cgcnn_parity_plots_v2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  保存 {out}")


def plot_three_way_comparison(cgcnn_metrics):
    rf_path = OUTPUT_DIR / "baseline_metrics.json"
    roost_path = OUTPUT_DIR / "roost_metrics_v2.json"
    if not roost_path.exists():
        roost_path = OUTPUT_DIR / "roost_metrics.json"

    if not (rf_path.exists() and roost_path.exists()):
        log("  跳过三模型对比图(缺少 RF/Roost metrics 文件)")
        return

    rf = json.loads(rf_path.read_text(encoding="utf-8"))
    roost = json.loads(roost_path.read_text(encoding="utf-8"))

    x = np.arange(len(TARGETS))
    w = 0.27

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    rf_r2 = [rf[t]["R2"] for t in TARGETS]
    rs_r2 = [roost[t]["R2"] for t in TARGETS]
    cg_r2 = [cgcnn_metrics[t]["R2"] for t in TARGETS]
    axes[0].bar(x - w, rf_r2, w, label="RF + Magpie", color="C0")
    axes[0].bar(x, rs_r2, w, label="MiniRoost", color="C2")
    axes[0].bar(x + w, cg_r2, w, label="MiniCGCNN", color="C3")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(TARGETS, rotation=15)
    axes[0].set_ylabel("R² (higher = better)")
    axes[0].set_title("Three-model R² comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    rf_rel = [rf[t]["MAE_over_std"] for t in TARGETS]
    rs_rel = [roost[t]["MAE_over_std"] for t in TARGETS]
    cg_rel = [cgcnn_metrics[t]["MAE_over_std"] for t in TARGETS]
    axes[1].bar(x - w, rf_rel, w, label="RF + Magpie", color="C0")
    axes[1].bar(x, rs_rel, w, label="MiniRoost", color="C2")
    axes[1].bar(x + w, cg_rel, w, label="MiniCGCNN", color="C3")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(TARGETS, rotation=15)
    axes[1].set_ylabel("MAE / σ_y (lower = better)")
    axes[1].set_title("Normalized error comparison")
    axes[1].axhline(0.5, ls="--", c="r", alpha=0.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Composition-based vs Structure-based (v2, per-fold scaler)",
                 y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "cgcnn_vs_baselines_v2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  保存 {out}")


# ============================================================
# 主流程
# ============================================================
def main():
    t0 = time.time()
    log("=" * 60)
    log("MiniCGCNN v2: per-fold scaler + complete parity dumps")
    log("=" * 60)
    log(f"设备: {DEVICE}")
    log(f"图: N_MAX={N_MAX}, K={K}, cutoff={CUTOFF}")
    log(f"模型: ATOM_DIM={ATOM_DIM}, EDGE_DIM={EDGE_DIM}, "
        f"N_CONV={N_CONV}, MLP_HIDDEN={MLP_HIDDEN}")
    log(f"训练: LR={LR}, BATCH={BATCH_SIZE}, EPOCHS={EPOCHS}, "
        f"PATIENCE={PATIENCE}")

    df, structures = load_data()
    metrics, predictions = cross_validate(df, structures)

    # ★ 修复 1: 三个目标都 dump parity
    for t in TARGETS:
        y_true, y_pred = predictions[t]
        out_npz = OUTPUT_DIR / f"cgcnn_parity_{t.replace('average_', '').replace('_grav', '').replace('max_delta_', 'delta_')}_v2.npz"
        np.savez(out_npz, y_true=y_true, y_pred=y_pred)
        log(f"  [Dump] {out_npz.name}, n={len(y_true)}")

    # ★ 修复 2: 重复保存清理 (单次保存)
    save_json(metrics, OUTPUT_DIR / "cgcnn_metrics_v2.json")

    plot_parity(predictions, metrics)
    plot_three_way_comparison(metrics)

    log("=" * 60)
    log("=== MiniCGCNN v2 结果 ===")
    for t, m in metrics.items():
        log(f"{t:20s}  MAE={m['MAE']:.4f}  R²={m['R2']:.4f}  "
            f"MAE/σ={m['MAE_over_std']:.4f}")
    log(f"总耗时: {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
