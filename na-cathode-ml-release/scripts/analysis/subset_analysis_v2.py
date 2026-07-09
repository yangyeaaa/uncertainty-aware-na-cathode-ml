"""
subset_analysis_v2.py
=====================

修复版子集分析,替代原 subset_analysis.py 和 Subset analysis dominant.py.

修复内容:
  1. 使用 dominant_tm 计数(每个样本只归一类),与论文 §2.1 自洽
  2. 每折内 fit StandardScaler(消除 feature scaling 数据泄漏)
  3. 输出与 Figure 5 兼容的 subset_metrics_v2.json
  4. 自动打印"论文 §2.1 应当报告的 n 值"

CRITICAL: 此版本输出的 n 值与原论文 §2.1 不同.必须按本脚本的
输出更新论文 §2.1 的 Mn/V/Fe/Ni/Cr/Co/Cu/Ti 样本数,以及 §3.4 的所有 R²
数字.对照表见运行末尾输出.

用法:
    python subset_analysis_v2.py

输出:
    ./mp_baseline/subset_metrics_v2.json
    ./mp_baseline/subset_n_paper_update.json   # 论文 §2.1 表的修正数字
    ./mp_baseline/subset_r2_paper_update.json  # 论文 §3.4 的修正 R²
"""

import warnings
warnings.filterwarnings("ignore")

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# Local
from utils import (
    log,
    load_and_filter,
    magpie_featurize,
    assign_dominant_tm,
    report_subset_counts,
    cv_predict_with_per_fold_scaler,
    compute_metrics,
    save_json,
    TM_LIST,
    DEFAULT_TARGETS,
)


# ============================================================
# 配置(与原 subset_analysis.py 一致,便于复现)
# ============================================================
N_JOBS = 2
N_ESTIMATORS = 150
MAX_DEPTH = 20
N_SPLITS = 5
MIN_SUBSET_SIZE = 20
SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "mp_baseline"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 模型工厂(给 cv_predict_with_per_fold_scaler 用)
# ============================================================
def rf_factory():
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=2,
        n_jobs=N_JOBS,
        random_state=SEED,
    )


# ============================================================
# 评估单个子集
# ============================================================
def evaluate_subset(
    df_sub: pd.DataFrame,
    feature_cols: list,
    targets: list,
    label: str = "",
) -> dict:
    """
    在子集 df_sub 上做 N_SPLITS-fold CV (per-fold scaler),
    报告 3 个目标的 MAE / R² / MAE/σy.
    """
    n = len(df_sub)
    n_splits = min(N_SPLITS, max(3, n // 6))  # 防止小子集 fold 过细

    X = df_sub[feature_cols].values.astype(np.float64)
    metrics = {}

    for target in targets:
        y = df_sub[target].values.astype(np.float64)
        y_pred = cv_predict_with_per_fold_scaler(
            X, y,
            model_factory=rf_factory,
            n_splits=n_splits,
            random_state=SEED,
            scale_X=True,
            scale_y=False,
        )
        m = compute_metrics(y, y_pred)
        m["n_samples"] = n
        m["n_splits"] = n_splits
        metrics[target] = m

    log(f"  [{label:>4s}] n={n:3d}  "
        f"V_R²={metrics['average_voltage']['R2']:+.3f}  "
        f"C_R²={metrics['capacity_grav']['R2']:+.3f}  "
        f"ΔV_R²={metrics['max_delta_volume']['R2']:+.3f}")
    return metrics


# ============================================================
# 主流程
# ============================================================
def main():
    t0 = time.time()
    log("=" * 60)
    log("Subset analysis V2: dominant counting + per-fold scaler")
    log("=" * 60)

    # 1) 加载 + 清洗
    df = load_and_filter(OUTPUT_DIR / "na_electrodes.csv",
                         targets=DEFAULT_TARGETS)

    # 2) Magpie 特征
    df, feature_cols = magpie_featurize(df, n_jobs=N_JOBS)

    # 3) 给每行分配 dominant_tm
    df = assign_dominant_tm(df)

    # 4) 报告子集 n 值
    n_per_tm = report_subset_counts(df, tm_list=TM_LIST)

    # 5) 全数据集基准
    log("\n训练全数据集基准...")
    full_metrics = evaluate_subset(df, feature_cols,
                                   DEFAULT_TARGETS, label="All")

    # 6) 逐子集
    log("\n训练 TM 子集...")
    subset_metrics = {"All": full_metrics}

    for tm in TM_LIST:
        n = n_per_tm[tm]
        if n < MIN_SUBSET_SIZE:
            log(f"  [{tm:>3s}] n={n} < MIN ({MIN_SUBSET_SIZE}), skipped")
            continue
        df_sub = df[df["dominant_tm"] == tm].reset_index(drop=True)
        subset_metrics[tm] = evaluate_subset(
            df_sub, feature_cols, DEFAULT_TARGETS, label=tm
        )

    # 7) 保存
    save_json(subset_metrics, OUTPUT_DIR / "subset_metrics_v2.json")

    # 8) 论文表更新清单
    update_n = {tm: subset_metrics[tm]["average_voltage"]["n_samples"]
                for tm in TM_LIST if tm in subset_metrics}
    update_n["__total_assigned__"] = int(df["dominant_tm"].notna().sum())
    update_n["__total_dataset__"] = int(len(df))
    save_json(update_n, OUTPUT_DIR / "subset_n_paper_update.json")

    update_r2 = {}
    for tm in TM_LIST:
        if tm not in subset_metrics:
            continue
        update_r2[tm] = {
            "voltage_R2":  subset_metrics[tm]["average_voltage"]["R2"],
            "capacity_R2": subset_metrics[tm]["capacity_grav"]["R2"],
            "dvolume_R2":  subset_metrics[tm]["max_delta_volume"]["R2"],
            "n":           subset_metrics[tm]["average_voltage"]["n_samples"],
        }
    save_json(update_r2, OUTPUT_DIR / "subset_r2_paper_update.json")

    # 9) 论文对照报告
    log("\n" + "=" * 60)
    log("PAPER UPDATE TABLE — 必须按下列数字更新论文 §2.1 和 §3.4")
    log("=" * 60)

    paper_old = {
        "Mn": {"n_old": 51, "vR2_old": 0.60, "dvR2_old": 0.32},
        "V":  {"n_old": 47, "vR2_old": 0.54, "dvR2_old": 0.22},
        "Fe": {"n_old": 38, "vR2_old": -0.04, "dvR2_old": 0.43},
        "Ni": {"n_old": 31, "vR2_old": 0.64, "dvR2_old": -0.07},
        "Cr": {"n_old": 26, "vR2_old": 0.21, "dvR2_old": 0.12},
        "Co": {"n_old": 24, "vR2_old": 0.43, "dvR2_old": 0.05},
        "Cu": {"n_old": 22, "vR2_old": 0.43, "dvR2_old": 0.12},
        "Ti": {"n_old": 22, "vR2_old": 0.19, "dvR2_old": 0.19},
    }

    log(f"{'TM':>4s} | {'n_old':>5s} -> {'n_new':>5s} | "
        f"{'vR²_old':>7s} -> {'vR²_new':>7s} | "
        f"{'ΔvR²_old':>8s} -> {'ΔvR²_new':>8s}")
    log("-" * 75)
    for tm in TM_LIST:
        if tm not in subset_metrics:
            continue
        n_new = update_n[tm]
        vr2_new = update_r2[tm]["voltage_R2"]
        dvr2_new = update_r2[tm]["dvolume_R2"]
        po = paper_old[tm]
        log(f"{tm:>4s} | {po['n_old']:>5d} -> {n_new:>5d} | "
             f"{po['vR2_old']:>+7.2f} -> {vr2_new:>+7.3f} | "
             f"{po['dvR2_old']:>+8.2f} -> {dvr2_new:>+8.3f}")

    log("\n核心 Fe-Ni reversal 仍然成立? 检查:")
    fe_v = update_r2["Fe"]["voltage_R2"]
    fe_dv = update_r2["Fe"]["dvolume_R2"]
    ni_v = update_r2["Ni"]["voltage_R2"]
    ni_dv = update_r2["Ni"]["dvolume_R2"]
    all_v = full_metrics["average_voltage"]["R2"]
    all_dv = full_metrics["max_delta_volume"]["R2"]
    log(f"  Fe:  voltage R²={fe_v:+.3f}  (全集 {all_v:+.3f})  "
        f"ΔV R²={fe_dv:+.3f}  (全集 {all_dv:+.3f})")
    log(f"  Ni:  voltage R²={ni_v:+.3f}  (全集 {all_v:+.3f})  "
        f"ΔV R²={ni_dv:+.3f}  (全集 {all_dv:+.3f})")
    fe_worst_v = all(fe_v < update_r2[tm]["voltage_R2"]
                     for tm in TM_LIST
                     if tm in update_r2 and tm != "Fe")
    fe_best_dv = all(fe_dv > update_r2[tm]["dvolume_R2"]
                     for tm in TM_LIST
                     if tm in update_r2 and tm != "Fe")
    log(f"  Fe is worst on V across all TMs?   {fe_worst_v}")
    log(f"  Fe is best on ΔV across all TMs?   {fe_best_dv}")

    log(f"\n总耗时: {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
