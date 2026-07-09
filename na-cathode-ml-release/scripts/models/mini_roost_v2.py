"""
mini_roost_v2.py
================

修复版 MiniRoost,替代原 mini_roost.py.

修复内容(两处):
  1. ★ target_scaler 改为每折内 fit(消除 data leakage)
  2. ★ 不再使用 "fraction-weighted attention-pooling" 这一可能误导的称谓:
     原模型实际是 "self-attention + fraction-weighted sum-pool",
     与原版 Roost 的 attention-as-weight 不完全相同, 论文叙述
     已对应调整为 "Roost-inspired composition attention".
     代码逻辑保留不变, 只更名内部变量增加可读性.

不变(继续保留):
  - Element embedding (32 维)
  - 单层 multi-head self-attention (4 heads)
  - 分数加权 sum-pooling
  - 两层 MLP head
  - AdamW lr=1e-3, weight_decay=1e-4, batch=32, epochs=200, patience=30
  - L1 loss

输出:
  - mp_baseline/roost_metrics_v2.json
  - mp_baseline/roost_parity_plots_v2.png
  - mp_baseline/roost_parity_voltage_v2.npz
  - mp_baseline/roost_parity_capacity_v2.npz
  - mp_baseline/roost_parity_delta_volume_v2.npz
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from pymatgen.core import Composition

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

EMBED_DIM = 32
N_HEADS = 4
MLP_HIDDEN = 64
DROPOUT = 0.1
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 200
PATIENCE = 30
SEED = 42
N_SPLITS = 5
MAX_ELEMENTS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. 数据集
# ============================================================
def parse_composition(formula: str):
    """从化学式解析 (元素序数列表, 分数列表)"""
    try:
        comp = Composition(formula)
        elements, fractions = [], []
        for el, frac in comp.fractional_composition.as_dict().items():
            elements.append(int(Composition(el).elements[0].Z))
            fractions.append(float(frac))
        while len(elements) < MAX_ELEMENTS:
            elements.append(0)
            fractions.append(0.0)
        return elements[:MAX_ELEMENTS], fractions[:MAX_ELEMENTS]
    except Exception:
        return None, None


class NaCathodeDataset(Dataset):
    def __init__(self, df, targets):
        self.elements, self.fractions, self.targets = [], [], []
        for _, row in df.iterrows():
            els, fracs = parse_composition(row["formula_discharge"])
            if els is None:
                continue
            self.elements.append(els)
            self.fractions.append(fracs)
            self.targets.append([row[t] for t in targets])
        self.elements = torch.LongTensor(self.elements)
        self.fractions = torch.FloatTensor(self.fractions)
        self.targets = torch.FloatTensor(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.elements[idx], self.fractions[idx], self.targets[idx]


# ============================================================
# 2. 模型 (与原版相同, 重命名内部以增加可读性)
# ============================================================
class MiniRoost(nn.Module):
    """
    Roost-inspired composition attention model.

    Architecture: 
        Element embedding -> multi-head self-attention -> 
        fraction-weighted sum pooling -> MLP head.

    NOTE: 这与原版 Roost (Goodall & Lee 2020) 的 attention-as-weight 风格
    不完全相同; 本实现使用 self-attention 更新 element embedding,
    然后用外部 molar fraction 做加权 sum-pool. 论文叙述应称为
    "Roost-inspired" 而非 "simplified Roost".
    """

    def __init__(self, n_targets, embed_dim=EMBED_DIM,
                 n_heads=N_HEADS, mlp_hidden=MLP_HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.element_emb = nn.Embedding(119, embed_dim, padding_idx=0)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_targets),
        )

    def forward(self, elements, fractions):
        emb = self.element_emb(elements)
        mask = (elements == 0)  # padding mask
        attn_out, _ = self.attn(emb, emb, emb, key_padding_mask=mask)
        emb = self.norm(emb + attn_out)
        # Fraction-weighted sum pool (跨元素求加权和)
        weights = fractions.unsqueeze(-1)
        pooled = (emb * weights).sum(dim=1)
        return self.mlp(pooled)


# ============================================================
# 3. 训练单折 (★ 接收 per-fold scaler)
# ============================================================
def train_one_fold(train_ds, val_ds, n_targets, target_scaler):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    model = MiniRoost(n_targets=n_targets).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.L1Loss()

    best_val = float("inf")
    best_state = None
    patience_cnt = 0

    for epoch in range(EPOCHS):
        model.train()
        for els, fracs, y in train_loader:
            els, fracs, y = els.to(DEVICE), fracs.to(DEVICE), y.to(DEVICE)
            y_norm = torch.from_numpy(
                target_scaler.transform(y.cpu().numpy())
            ).float().to(DEVICE)
            pred = model(els, fracs)
            loss = loss_fn(pred, y_norm)
            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        val_loss, n_count = 0, 0
        with torch.no_grad():
            for els, fracs, y in val_loader:
                els, fracs, y = els.to(DEVICE), fracs.to(DEVICE), y.to(DEVICE)
                y_norm = torch.from_numpy(
                    target_scaler.transform(y.cpu().numpy())
                ).float().to(DEVICE)
                pred = model(els, fracs)
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
        for els, fracs, y in val_loader:
            els, fracs = els.to(DEVICE), fracs.to(DEVICE)
            pred_norm = model(els, fracs).cpu().numpy()
            pred = target_scaler.inverse_transform(pred_norm)
            preds.append(pred)
            trues.append(y.numpy())

    return np.vstack(trues), np.vstack(preds), epoch + 1


def cross_validate(df):
    log("开始 5 折交叉验证(per-fold target scaler)...")
    n_targets = len(TARGETS)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    all_true = {t: [] for t in TARGETS}
    all_pred = {t: [] for t in TARGETS}

    for fold, (train_idx, val_idx) in enumerate(kf.split(df), 1):
        log(f"  Fold {fold}/{N_SPLITS} ...")
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # ★ 每折内 fit target_scaler
        target_scaler = StandardScaler()
        target_scaler.fit(train_df[TARGETS].values)

        train_ds = NaCathodeDataset(train_df, TARGETS)
        val_ds = NaCathodeDataset(val_df, TARGETS)

        torch.manual_seed(SEED + fold)
        trues, preds, n_epochs = train_one_fold(
            train_ds, val_ds, n_targets, target_scaler
        )
        log(f"     stopped at epoch {n_epochs}")

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
# 4. 数据
# ============================================================
def load_and_filter():
    csv_path = OUTPUT_DIR / "na_electrodes.csv"
    log(f"加载 {csv_path}")
    df = pd.read_csv(csv_path)
    n0 = len(df)
    df = df[df["average_voltage"] > 0]
    df = df[df["stability_discharge"] < 0.1]
    df = df.dropna(subset=TARGETS + ["formula_discharge"])
    df = df[df["max_delta_volume"].abs() < 1.0]
    df = df.reset_index(drop=True)
    log(f"  清洗: {n0} -> {len(df)} 条")
    return df


# ============================================================
# 5. 可视化
# ============================================================
def plot_parity(predictions, metrics):
    log("绘制 parity 图...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, target in zip(axes, TARGETS):
        y_true, y_pred = predictions[target]
        ax.scatter(y_true, y_pred, alpha=0.4, s=15, edgecolor="none",
                   color="C2")
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"True {TARGET_LABELS[target]}")
        ax.set_ylabel(f"Predicted {TARGET_LABELS[target]}")
        m = metrics[target]
        ax.set_title(f"{target}\nMAE={m['MAE']:.3f}  R²={m['R2']:.3f}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("MiniRoost v2 (5-fold CV, per-fold scaler)", y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "roost_parity_plots_v2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  保存 {out}")


# ============================================================
# 主流程
# ============================================================
def main():
    t0 = time.time()
    log("=" * 60)
    log("MiniRoost v2: per-fold scaler")
    log("=" * 60)
    log(f"设备: {DEVICE}")
    log(f"配置: EMBED_DIM={EMBED_DIM}, N_HEADS={N_HEADS}, "
        f"EPOCHS={EPOCHS}, PATIENCE={PATIENCE}")

    df = load_and_filter()
    metrics, predictions = cross_validate(df)

    # ★ 三个目标都 dump parity
    for t in TARGETS:
        y_true, y_pred = predictions[t]
        out_name = t.replace('average_', '').replace('_grav', '').replace('max_delta_', 'delta_')
        out_npz = OUTPUT_DIR / f"roost_parity_{out_name}_v2.npz"
        np.savez(out_npz, y_true=y_true, y_pred=y_pred)
        log(f"  [Dump] {out_npz.name}, n={len(y_true)}")

    save_json(metrics, OUTPUT_DIR / "roost_metrics_v2.json")
    plot_parity(predictions, metrics)

    log("=" * 60)
    log("=== MiniRoost v2 结果 ===")
    for t, m in metrics.items():
        log(f"{t:20s}  MAE={m['MAE']:.4f}  R²={m['R2']:.4f}  "
            f"MAE/σ={m['MAE_over_std']:.4f}")
    log(f"总耗时: {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
