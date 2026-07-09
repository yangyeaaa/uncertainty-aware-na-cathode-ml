"""
run_cgcnn_multi_seed.py
========================

批量跑 mini_cgcnn_v2.py 六个种子, 输出 Table S4 (mean ± std).

服务器用法:
    cd <project_dir>
    cp /path/to/utils.py .
    cp /path/to/mini_cgcnn_v2.py .
    python run_cgcnn_multi_seed.py

如果你有多 GPU, 可以拆成 6 个 nohup 并行跑(每个 GPU 1 种子),
但单卡顺序跑预计 6 × 30-60 min = 3-6 小时.

输出:
    mp_baseline/cgcnn_multi_seed_results.json
        每个种子的 metrics
    mp_baseline/cgcnn_multi_seed_summary.json
        mean ± std (论文 §3.3 + Table S4 用)
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
import importlib
import shutil
from pathlib import Path

import numpy as np
import torch

import mini_cgcnn_v2 as cgcnn  # 必须在同一目录
from utils import log, save_json


SEEDS = [42, 43, 44, 45, 46, 47]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "mp_baseline"
OUTPUT_DIR.mkdir(exist_ok=True)

#OUTPUT_DIR = Path("./mp_baseline")


def run_one_seed(seed: int) -> dict:
    log(f"\n{'='*60}")
    log(f"  SEED = {seed}")
    log(f"{'='*60}")

    # 设置种子(影响模型 init 和 fold shuffle 都同步)
    cgcnn.SEED = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    df, structures = cgcnn.load_data()
    metrics, _ = cgcnn.cross_validate(df, structures)
    return {t: dict(m) for t, m in metrics.items()}


def main():
    t0 = time.time()
    log("=" * 60)
    log(f"Multi-seed MiniCGCNN v2: {len(SEEDS)} seeds")
    log(f"Seeds: {SEEDS}")
    log("=" * 60)

    all_results = {}
    for seed in SEEDS:
        seed_metrics = run_one_seed(seed)
        all_results[str(seed)] = seed_metrics
        # 保存中间结果(防止中途崩盘)
        save_json(all_results, OUTPUT_DIR / "cgcnn_multi_seed_results.json")
        log(f"\n  Seed {seed} done. Elapsed: {(time.time()-t0)/60:.1f} min")

    # 计算 mean ± std
    log("\n" + "=" * 60)
    log("Computing mean ± std across seeds...")
    log("=" * 60)

    summary = {}
    metric_keys = ["MAE", "R2", "MAE_over_std"]
    targets = list(all_results[str(SEEDS[0])].keys())

    for t in targets:
        summary[t] = {}
        for mk in metric_keys:
            vals = [all_results[str(s)][t][mk] for s in SEEDS]
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))  # 样本标准差
            summary[t][mk] = {
                "mean": round(mean, 4),
                "std": round(std, 4),
                "per_seed": [round(v, 4) for v in vals],
            }
        log(f"\n  {t}:")
        log(f"    MAE          = {summary[t]['MAE']['mean']:.4f} "
            f"± {summary[t]['MAE']['std']:.4f}")
        log(f"    R²           = {summary[t]['R2']['mean']:.4f} "
            f"± {summary[t]['R2']['std']:.4f}")
        log(f"    MAE/σ_y      = {summary[t]['MAE_over_std']['mean']:.4f} "
            f"± {summary[t]['MAE_over_std']['std']:.4f}")
        log(f"    Per seed R²: {summary[t]['R2']['per_seed']}")

    save_json(summary, OUTPUT_DIR / "cgcnn_multi_seed_summary.json")

    # 论文叙事数字
    log("\n" + "=" * 60)
    log("PAPER FILL-IN NUMBERS (replace §3.3 6-seed numbers)")
    log("=" * 60)
    for t in targets:
        s = summary[t]
        log(f"  {t}:")
        log(f"    R² = {s['R2']['mean']:.2f} ± {s['R2']['std']:.2f}")
        log(f"    MAE/σ_y = {s['MAE_over_std']['mean']:.2f} "
            f"± {s['MAE_over_std']['std']:.2f}")

    # ΔV improvement vs composition baseline
    rf_path = OUTPUT_DIR / "baseline_metrics.json"
    if rf_path.exists():
        rf = json.loads(rf_path.read_text(encoding="utf-8"))
        dv_rf = rf["max_delta_volume"]["R2"]
        dv_cg = summary["max_delta_volume"]["R2"]["mean"]
        improvement = (dv_cg - dv_rf) / dv_rf * 100
        log(f"\n  ΔV improvement: {improvement:.1f}% over RF baseline")
        log(f"    (RF: {dv_rf:.3f}, MiniCGCNN: {dv_cg:.3f})")

    log(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
    log(f"Results saved to {OUTPUT_DIR}/cgcnn_multi_seed_results.json")
    log(f"Summary saved to {OUTPUT_DIR}/cgcnn_multi_seed_summary.json")


if __name__ == "__main__":
    main()
