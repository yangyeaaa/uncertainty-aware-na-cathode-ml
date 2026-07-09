"""
baseline_na_v2.py
=================

候选 1 RF Baseline (适配 na_paper_patch_x 工作目录).

与原 baseline_na.py 的区别:
  1. 使用 pathfix 风格的智能路径解析,无论工作目录怎么设都能跑
  2. 引用 utils.py 的工具函数,与 v2 系列对齐
  3. 输出与原文件名一致 (rf_parity_*.npz, baseline_metrics.json)
  4. RF 是 deterministic 模型,不用做多种子;但与 v2 一致用 per-fold scaler
     (注: 对 RF 几乎无影响, 但写出来更严谨)

输出:
  mp_baseline/baseline_metrics.json
  mp_baseline/baseline_parity_plots.png
  mp_baseline/baseline_feature_importance.png
  mp_baseline/baseline_feature_importance.json
  mp_baseline/rf_parity_delta_volume.npz   ← Fig4 需要这个
  mp_baseline/rf_parity_voltage.npz
  mp_baseline/rf_parity_capacity.npz       ← 新增
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from utils import (
    log, load_and_filter, magpie_featurize,
    cv_predict_with_per_fold_scaler, compute_metrics, save_json,
    DEFAULT_TARGETS,
)


# ============================================================
# ★ 智能找 mp_baseline 目录
# ============================================================
def find_data_dir():
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "mp_baseline",
        script_dir.parent / "mp_baseline",
        Path.cwd() / "mp_baseline",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            log(f"[OK] 找到数据目录: {c}")
            return c
    raise FileNotFoundError("mp_baseline 目录未找到")


OUTPUT_DIR = find_data_dir()
TARGETS = DEFAULT_TARGETS
TARGET_LABELS = {
    "average_voltage": "Voltage (V)",
    "capacity_grav": "Capacity (mAh/g)",
    "max_delta_volume": "Volume change",
}

# 配置
N_JOBS = 2
N_ESTIMATORS = 150
MAX_DEPTH = 20
N_SPLITS = 5
SEED = 42


def rf_factory():
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=2,
        n_jobs=N_JOBS,
        random_state=SEED,
    )


def train_evaluate(df, feature_cols):
    log(f"训练 Random Forest ({N_SPLITS} 折 CV, per-fold scaler)...")
    X = df[feature_cols].values.astype(np.float64)
    metrics = {}
    predictions = {}

    for i, target in enumerate(TARGETS, 1):
        log(f"  [{i}/{len(TARGETS)}] {target} ...")
        y = df[target].values.astype(np.float64)
        y_pred = cv_predict_with_per_fold_scaler(
            X, y,
            model_factory=rf_factory,
            n_splits=N_SPLITS,
            random_state=SEED,
            scale_X=True,
            scale_y=False,
        )
        m = compute_metrics(y, y_pred)
        metrics[target] = m
        predictions[target] = (y, y_pred)
        log(f"      MAE={m['MAE']:.3f}  R²={m['R2']:.3f}  "
            f"MAE/σ={m['MAE_over_std']:.3f}")

    return metrics, predictions


def plot_parity(predictions, metrics):
    log("绘制 parity 图...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, target in zip(axes, TARGETS):
        y_true, y_pred = predictions[target]
        ax.scatter(y_true, y_pred, alpha=0.4, s=15, edgecolor="none")
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"True {TARGET_LABELS[target]}")
        ax.set_ylabel(f"Predicted {TARGET_LABELS[target]}")
        m = metrics[target]
        ax.set_title(f"{target}\nMAE={m['MAE']:.3f}  R²={m['R2']:.3f}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Na Cathode RF Baseline (per-fold CV)", y=1.02)
    fig.tight_layout()
    out = OUTPUT_DIR / "baseline_parity_plots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  保存 {out}")


def plot_feature_importance(df, feature_cols, top_n=15):
    log(f"计算 voltage 特征重要性 (Top {top_n})...")
    X = df[feature_cols].values
    y = df["average_voltage"].values
    model = RandomForestRegressor(
        n_estimators=200, max_depth=MAX_DEPTH,
        n_jobs=N_JOBS, random_state=SEED,
    )
    model.fit(X, y)

    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=True)
    top = importances.tail(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color="steelblue")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Magpie Features for Na Voltage Prediction")
    fig.tight_layout()
    out = OUTPUT_DIR / "baseline_feature_importance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  保存 {out}")

    top_desc = top[::-1]
    top_dict = [{"feature": str(f), "importance": float(v)}
                for f, v in zip(top_desc.index, top_desc.values)]
    save_json(top_dict, OUTPUT_DIR / "baseline_feature_importance.json")

    return top


def main():
    t0 = time.time()
    log("=" * 60)
    log("RF Baseline (per-fold scaler, pathfix)")
    log("=" * 60)

    df = load_and_filter(OUTPUT_DIR / "na_electrodes.csv", targets=TARGETS)
    df, feature_cols = magpie_featurize(df, n_jobs=N_JOBS)

    metrics, predictions = train_evaluate(df, feature_cols)

    # Dump parity 数据
    for t in TARGETS:
        y_true, y_pred = predictions[t]
        # 文件名: 与原 baseline_na.py 保持一致(以便 Fig4_v3 能读到)
        if t == "average_voltage":
            name = "voltage"
        elif t == "capacity_grav":
            name = "capacity"
        elif t == "max_delta_volume":
            name = "delta_volume"
        out_npz = OUTPUT_DIR / f"rf_parity_{name}.npz"
        np.savez(out_npz, y_true=y_true, y_pred=y_pred)
        log(f"  [Dump] {out_npz.name}, n={len(y_true)}")

    save_json(metrics, OUTPUT_DIR / "baseline_metrics.json")

    plot_parity(predictions, metrics)
    plot_feature_importance(df, feature_cols)

    log("=" * 60)
    log("=== Baseline 结果总结 ===")
    log(f"样本数: {len(df)}")
    log(f"特征数: {len(feature_cols)}")
    for target, m in metrics.items():
        log(f"{target:20s}  MAE={m['MAE']:.3f}  R²={m['R2']:.3f}  "
            f"MAE/σ={m['MAE_over_std']:.3f}")
    log(f"总耗时: {time.time()-t0:.1f} 秒")


if __name__ == "__main__":
    main()
