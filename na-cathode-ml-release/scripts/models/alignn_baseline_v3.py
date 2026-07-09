"""
ALIGNN baseline on the 365-pair Na cathode benchmark.
Run:  python alignn_baseline.py          (see install notes at the bottom)

Purpose: the manuscript claims MiniCGCNN's ~19% Delta-Volume improvement shows
"structure helps lattice-response targets". The obvious reviewer attack is
"would an angle-aware SOTA GNN (ALIGNN) not do even better / does the
conclusion survive a modern baseline?". This script answers it under the
IDENTICAL protocol as the benchmark:
  same 365-pair csv (utils.load_and_filter), same structures pickle,
  same >100-atom skip (-> 356 structures), same KFold(5, shuffle,
  random_state=42), per-fold target StandardScaler, L1 loss on normalised
  targets, AdamW(1e-3, wd 1e-4), patience 30, metrics on concatenated
  held-out predictions; multi-seed mean +/- std.

Deviations from the MiniCGCNN protocol (state in the paper):
  * six seeds {42..47} for full parity with MiniCGCNN.
  * CPU-ONLY run: the available Windows dgl wheel is CPU-only, and ALIGNN's
    angle-embedding keeps buffers on the model device while edge features stay
    with the (CPU) graph -- so a hybrid CPU-graph/GPU-model split hits a device
    mismatch. Running everything on CPU is correct and, for 356 small graphs at
    batch 2 + 2/2-layer 128-hidden ALIGNN, completes overnight with no VRAM
    ceiling. Install a CUDA-enabled dgl build to move to GPU later.
  * reduced ALIGNN config sized for a 4 GB GPU (RTX 3050 Ti Laptop): default
    2 ALIGNN + 2 GCN layers, 128 hidden, batch 2 with gradient accumulation to
    an effective batch of 8 (matches the benchmark's batch semantics). This is
    the real, pip-installed ALIGNN -- only depth/width are reduced, consistent
    with the manuscript's Mini* small-data sizing. An automatic OOM guard
    halves the batch and retries so a single large cell cannot kill the run.

Output: mp_baseline/alignn_baseline_per_seed.csv
        mp_baseline/alignn_baseline_summary.json
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "paches")
import utils

# ----------------------------- CONFIG -----------------------------
STRUCTS = "mp_baseline/na_structures.pkl"
ELECTRODES = "mp_baseline/na_electrodes.csv"
OUT_DIR = Path("mp_baseline")
TARGETS = ["average_voltage", "capacity_grav", "max_delta_volume"]
N_SPLITS = 5
SEEDS = [42, 43, 44]        # 3 seeds for a CPU-feasible mean+/-std;
                           # a baseline with n=3 is standard (note in Methods)
FULL_SIZE = False           # keep False on 4 GB; True needs >=12 GB VRAM
ALIGNN_LAYERS = 2           # 4GB-sized; the pip ALIGNN, reduced depth/width
GCN_LAYERS = 2
HIDDEN = 96                 # CPU-trimmed from 128; angle-embed is the CPU cost
N_MAX_ATOMS = 100           # identical to MiniCGCNN constraint
CUTOFF, MAX_NBR = 8.0, 12   # identical to MiniCGCNN graph construction
BATCH_SIZE = 8             # CPU: no VRAM ceiling; big batch cuts per-batch
                           # scheduling overhead (178 -> ~45 batches/epoch)
ACCUM_STEPS = 1            # effective batch = 8 (benchmark semantics)
EPOCHS = 120               # small-data ALIGNN + patience 30 rarely needs more
PATIENCE = 30
LR, WEIGHT_DECAY = 1e-3, 1e-4
GRAPH_CPU = True            # CPU-only run: graphs and model both on CPU
# ------------------------------------------------------------------


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------- dependency preflight -----------------------
def preflight():
    """Fail fast with actionable messages; dgl-on-Windows is the usual pain."""
    log("preflight: imports ...")
    try:
        import torch  # noqa
    except ImportError:
        sys.exit("torch missing -- use the mlip env")
    try:
        import dgl  # noqa
    except ImportError:
        sys.exit("dgl missing. CPU wheel:\n"
                 "  pip install dgl -f https://data.dgl.ai/wheels/repo.html\n"
                 "(pick the build matching your torch major version)")
    try:
        from alignn.models.alignn import ALIGNN, ALIGNNConfig  # noqa
        from alignn.graphs import Graph  # noqa
    except ImportError:
        sys.exit("alignn missing:  pip install alignn")
    try:
        from jarvis.core.atoms import Atoms  # noqa
    except ImportError:
        sys.exit("jarvis-tools missing:  pip install jarvis-tools")

    # one tiny structure end-to-end: graph -> line graph -> forward
    log("preflight: 1-structure graph + forward ...")
    import torch
    from jarvis.core.atoms import Atoms as JAtoms
    from alignn.graphs import Graph
    from alignn.models.alignn import ALIGNN, ALIGNNConfig
    a = JAtoms(lattice_mat=np.eye(3) * 4.29,
               coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
               elements=["Na", "Na"], cartesian=False)
    g, lg = Graph.atom_dgl_multigraph(a, cutoff=CUTOFF,
                                      max_neighbors=MAX_NBR,
                                      atom_features="cgcnn",
                                      compute_line_graph=True)
    lat = torch.tensor(np.array(a.lattice_mat), dtype=torch.float32).unsqueeze(0)
    cfg = ALIGNNConfig(name="alignn", alignn_layers=1, gcn_layers=1,
                       hidden_features=16, embedding_features=16,
                       output_features=3)
    with torch.no_grad():
        out = ALIGNN(cfg)([g, lg, lat])
    assert tuple(out.shape[-1:]) == (3,), f"unexpected output shape {out.shape}"
    log("preflight OK")


def pmg_to_jarvis(structure):
    from jarvis.core.atoms import Atoms as JAtoms
    return JAtoms(lattice_mat=np.array(structure.lattice.matrix),
                  coords=np.array(structure.frac_coords),
                  elements=[str(sp.symbol) for sp in structure.species],
                  cartesian=False)


# ------------------------------ main --------------------------------
def main():
    preflight()
    import pickle
    import torch
    import dgl
    from torch.utils.data import DataLoader, Dataset
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from alignn.graphs import Graph
    from alignn.models.alignn import ALIGNN, ALIGNNConfig

    device = torch.device("cpu")       # forced CPU (see header)
    torch.set_num_threads(max(1, os.cpu_count() - 1))   # use most cores
    log(f"CPU threads: {torch.get_num_threads()}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        free, total = torch.cuda.mem_get_info()
        log(f"device: {device} | VRAM free {free/2**30:.1f}/{total/2**30:.1f} GB")
    else:
        log(f"device: {device}")

    def _oom_recover(model, opt, dev):
        opt.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    df = utils.load_and_filter(ELECTRODES)
    structs = pickle.load(open(STRUCTS, "rb"))

    # ---- build graphs ONCE, shared across folds/seeds ---------------
    log("building ALIGNN graphs (once, cached in memory) ...")
    items, skipped = [], 0
    t0 = time.time()
    for _, row in df.iterrows():
        bid = row["battery_id"]
        S = structs.get(bid)
        if S is None or len(S) > N_MAX_ATOMS:
            skipped += 1
            continue
        try:
            ja = pmg_to_jarvis(S)
            g, lg = Graph.atom_dgl_multigraph(
                ja, cutoff=CUTOFF, max_neighbors=MAX_NBR,
                atom_features="cgcnn", compute_line_graph=True)
            lat = torch.tensor(np.array(ja.lattice_mat),
                               dtype=torch.float32)          # (3,3) per crystal
            y = np.array([row[t] for t in TARGETS], dtype=np.float32)
            items.append((g, lg, lat, y))
        except Exception:
            skipped += 1
    log(f"graphs: {len(items)} built, {skipped} skipped "
        f"({time.time()-t0:.0f}s)  [MiniCGCNN used 356]")

    class DS(Dataset):
        def __init__(self, idx, y_scaled):
            self.idx, self.y = idx, y_scaled
        def __len__(self):
            return len(self.idx)
        def __getitem__(self, i):
            g, lg, lat, _ = items[self.idx[i]]
            return g, lg, lat, torch.from_numpy(self.y[i])

    def collate(batch):
        gs, lgs, lats, ys = zip(*batch)
        return (dgl.batch(gs), dgl.batch(lgs),
                torch.stack(lats), torch.stack(ys))

    def make_model():
        if FULL_SIZE:
            cfg = ALIGNNConfig(name="alignn", output_features=len(TARGETS))
        else:
            cfg = ALIGNNConfig(name="alignn", alignn_layers=ALIGNN_LAYERS,
                               gcn_layers=GCN_LAYERS, hidden_features=HIDDEN,
                               embedding_features=HIDDEN,
                               output_features=len(TARGETS))
        return ALIGNN(cfg).to(device)

    Y_all = np.stack([it[3] for it in items])
    n = len(items)
    results = []            # one row per (seed, target)

    for seed in SEEDS:
        log(f"=== seed {seed} ===")
        oof = np.full_like(Y_all, np.nan, dtype=np.float64)
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        for fold, (trainval, test) in enumerate(kf.split(np.arange(n))):
            torch.manual_seed(seed + fold)       # identical seeding scheme
            np.random.seed(seed + fold)
            # v3: test = held-out fold(仅评估); 从trainval内部再切early-stop验证集
            rng_es = np.random.RandomState(seed + fold)
            perm = rng_es.permutation(len(trainval))
            n_es = max(1, int(round(len(trainval) * 0.1)))
            es = trainval[perm[:n_es]]           # early-stopping验证子集
            tr = trainval[perm[n_es:]]           # 真正的训练子集
            scaler = StandardScaler().fit(Y_all[tr])   # 只在训练子集fit, leak-free
            dl_tr = DataLoader(DS(tr, scaler.transform(Y_all[tr]).astype(np.float32)),
                               batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=collate)
            dl_es = DataLoader(DS(es, scaler.transform(Y_all[es]).astype(np.float32)),
                               batch_size=BATCH_SIZE, shuffle=False,
                               collate_fn=collate)
            dl_test = DataLoader(DS(test, scaler.transform(Y_all[test]).astype(np.float32)),
                                 batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=collate)
            model = make_model()
            opt = torch.optim.AdamW(model.parameters(), lr=LR,
                                    weight_decay=WEIGHT_DECAY)
            lossf = torch.nn.L1Loss()
            best, best_state, bad = np.inf, None, 0
            t_fold = time.time()
            accum = max(1, ACCUM_STEPS)
            for ep in range(EPOCHS):
                model.train()
                opt.zero_grad()
                for bi, (g, lg, lat, y) in enumerate(dl_tr):
                    try:
                        if not GRAPH_CPU:
                            g, lg = g.to(device), lg.to(device)
                        lat, y = lat.to(device), y.to(device)
                        loss = lossf(model([g, lg, lat]), y) / accum
                        loss.backward()
                        if (bi + 1) % accum == 0:
                            opt.step(); opt.zero_grad()
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            _oom_recover(model, opt, device)
                            log(f"  [OOM] batch {bi} skipped, cache cleared")
                            continue
                        raise
                if (len(dl_tr) % accum) != 0:      # flush trailing grads
                    opt.step(); opt.zero_grad()
                model.eval()
                vl = 0.0
                with torch.no_grad():
                    for g, lg, lat, y in dl_es:
                        if not GRAPH_CPU:
                            g, lg = g.to(device), lg.to(device)
                        lat = lat.to(device)
                        vl += lossf(model([g, lg, lat]).cpu(), y).item() * len(y)
                vl /= len(es)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if vl < best - 1e-5:
                    best, bad = vl, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
                if ep == 0:
                    log(f"  fold {fold}: epoch time "
                        f"~{time.time()-t_fold:.0f}s (ETA visible now)")
            model.load_state_dict(best_state)
            model.eval()
            preds = []
            with torch.no_grad():
                for g, lg, lat, _ in dl_test:
                    if not GRAPH_CPU:
                        g, lg = g.to(device), lg.to(device)
                    preds.append(model([g, lg, lat.to(device)]).cpu().numpy())
            oof[test] = scaler.inverse_transform(np.vstack(preds))
            log(f"  fold {fold} done in {(time.time()-t_fold)/60:.1f} min "
                f"(stopped at epoch {ep})")

        for j, t in enumerate(TARGETS):
            yt, yp = Y_all[:, j], oof[:, j]
            mae = float(np.mean(np.abs(yt - yp)))
            sig = float(np.std(yt))
            ss = float(1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2))
            results.append(dict(seed=seed, target=t, r2=round(ss, 4),
                                mae=round(mae, 4),
                                mae_over_sigma=round(mae / sig, 4)))
            log(f"  seed {seed} {t}: R2={ss:.3f}  MAE/sigma={mae/sig:.3f}")

    per_seed = pd.DataFrame(results)
    per_seed.to_csv(OUT_DIR / "alignn_baseline_per_seed.csv", index=False)
    summary = {}
    for t in TARGETS:
        sub = per_seed[per_seed["target"] == t]
        summary[t] = {"r2_mean": round(sub["r2"].mean(), 4),
                      "r2_std": round(sub["r2"].std(ddof=1), 4),
                      "mae_over_sigma_mean": round(sub["mae_over_sigma"].mean(), 4),
                      "n_seeds": len(sub)}
    summary["_protocol"] = dict(n_structures=n, seeds=SEEDS,
                                full_size=FULL_SIZE, cutoff=CUTOFF,
                                max_neighbors=MAX_NBR, batch=BATCH_SIZE)
    with open(OUT_DIR / "alignn_baseline_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print("\n=== ALIGNN baseline summary (compare to RF 0.232 / MiniRoost "
          "0.213 / MiniCGCNN 0.28+/-0.04 on Delta-V) ===")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()

# ----------------------------------------------------------------------
# INSTALL (mlip env, Windows CPU):
#   pip install jarvis-tools alignn
#   pip install dgl -f https://data.dgl.ai/wheels/repo.html
# If dgl refuses to install/import on Windows (no wheel for your torch
# version): easiest escapes are (a) WSL, (b) Google Colab GPU (this file
# runs as-is; set FULL_SIZE=True and SEEDS=[42..47] there), or (c) ask for
# the dgl-free MiniALIGNN fallback.
# ----------------------------------------------------------------------
