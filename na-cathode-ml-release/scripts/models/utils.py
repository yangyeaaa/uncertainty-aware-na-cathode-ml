"""
utils.py
========

统一的数据加载、清洗、特征工程、评估工具,供所有训练/分析脚本调用。

替代以下文件中重复的 load_and_filter() 与 featurize():
  - baseline_na.py
  - subset_analysis.py
  - Subset analysis dominant.py
  - mini_roost.py
  - mini_cgcnn.py
  - mini_cgcnn_single.py

核心改进:
  1. dominant_tm 计数:每个样本仅归一类(消除 broadcasting 双重计入 bug)
  2. compute_subset_metrics() 统一接口
  3. fit_per_fold_scaler 工具(避免数据泄漏)
  4. compute_dataset_n_per_tm() 给论文 §2.1 的 n 表格用

依赖:
  pip install pandas numpy scikit-learn matminer pymatgen
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Composition

warnings.filterwarnings("ignore")


# ============================================================
# 默认配置(可被调用者覆盖)
# ============================================================
DEFAULT_TARGETS = ["average_voltage", "capacity_grav", "max_delta_volume"]

# 论文中考察的 8 种过渡金属
TM_LIST = ["Mn", "V", "Fe", "Ni", "Cr", "Cu", "Co", "Ti"]


# ============================================================
# 日志
# ============================================================
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================================================
# 1. 数据清洗:论文 §2.1 的三步过滤
# ============================================================
def load_and_filter(
    csv_path: str | Path,
    targets: List[str] = DEFAULT_TARGETS,
    voltage_min: float = 0.0,
    ehull_max: float = 0.1,
    dvol_max: float = 1.0,
    formula_col: str = "formula_discharge",
    stability_col: str = "stability_discharge",
) -> pd.DataFrame:
    """
    论文 §2.1 的三步清洗,与所有原脚本完全一致.

    Filters
    -------
    (i)   average_voltage > voltage_min  (默认 0)
    (ii)  stability_discharge < ehull_max  (默认 0.1 eV/atom)
    (iii) |max_delta_volume| < dvol_max  (默认 1.0)
    +     drop NaN in (targets + formula_col)

    Returns
    -------
    pd.DataFrame with reset_index, including the original columns.
    """
    csv_path = Path(csv_path)
    log(f"加载 {csv_path}")
    df = pd.read_csv(csv_path)
    n0 = len(df)

    df = df[df["average_voltage"] > voltage_min]
    df = df[df[stability_col] < ehull_max]
    df = df.dropna(subset=targets + [formula_col])
    df = df[df["max_delta_volume"].abs() < dvol_max]
    df = df.reset_index(drop=True)
    log(f"  清洗: {n0} -> {len(df)} 条")
    return df


# ============================================================
# 2. Magpie 特征工程
# ============================================================
def magpie_featurize(
    df: pd.DataFrame,
    formula_col: str = "formula_discharge",
    n_jobs: int = 2,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    用 matminer 抽取 132 维 Magpie 描述符.
    返回 (df_with_features, feature_cols).
    """
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.conversions import StrToComposition

    if verbose:
        log("Magpie 特征抽取(~30-60 秒)...")

    stc = StrToComposition(target_col_id="composition")
    stc.set_n_jobs(n_jobs)
    df = stc.featurize_dataframe(df, formula_col, ignore_errors=True)
    df = df.dropna(subset=["composition"]).reset_index(drop=True)

    ep = ElementProperty.from_preset("magpie")
    ep.set_n_jobs(n_jobs)
    df = ep.featurize_dataframe(df, "composition", ignore_errors=True)
    feature_cols = ep.feature_labels()
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    if verbose:
        log(f"  特征维度: {len(feature_cols)}, 剩余样本: {len(df)}")

    return df, feature_cols


# ============================================================
# 3. Dominant-TM 归类:论文 §2.1 应当使用的方式
# ============================================================
def compute_dominant_tm(
    formula: str,
    tm_list: List[str] = TM_LIST,
) -> Optional[str]:
    """
    返回 formula 中 molar fraction 最大的过渡金属.
    无 TM 则返回 None;平局按字母序取第一个.

    例:
      Na3Fe(PO4)(CO3)  -> 'Fe'
      Na2TiFe(PO4)3    -> Ti 和 Fe 等分,字母序优先 -> 'Fe'
      Na3V2(PO4)3      -> 'V'
    """
    try:
        comp = Composition(formula)
        frac = comp.fractional_composition.as_dict()
        tm_fracs = {tm: float(frac[tm]) for tm in tm_list if tm in frac}
        if not tm_fracs:
            return None
        max_frac = max(tm_fracs.values())
        tied = sorted([tm for tm, f in tm_fracs.items() if f == max_frac])
        return tied[0]
    except Exception:
        return None


def assign_dominant_tm(
    df: pd.DataFrame,
    formula_col: str = "formula_discharge",
    tm_list: List[str] = TM_LIST,
    col_name: str = "dominant_tm",
) -> pd.DataFrame:
    """添加 dominant_tm 列;无 TM 的样本得到 NaN."""
    df = df.copy()
    df[col_name] = df[formula_col].apply(
        lambda f: compute_dominant_tm(f, tm_list)
    )
    return df


def report_subset_counts(
    df: pd.DataFrame,
    tm_list: List[str] = TM_LIST,
    col_name: str = "dominant_tm",
) -> Dict[str, int]:
    """
    报告每个 TM 子集的样本数.返回 dict {tm: n}.
    """
    if col_name not in df.columns:
        df = assign_dominant_tm(df)
    counts = df[col_name].value_counts()
    out = {}
    log("\nDominant-TM subset counts:")
    sum_n = 0
    for tm in tm_list:
        n = int(counts.get(tm, 0))
        out[tm] = n
        log(f"  {tm:>3s}: n = {n}")
        sum_n += n
    log(f"  Sum (single-TM-assigned): {sum_n}")
    log(f"  Total dataset: {len(df)}")
    log(f"  No-TM (Na with non-TM redox center): {len(df) - sum_n}")
    return out


# ============================================================
# 4. 跨折评估(per-fold StandardScaler, 无数据泄漏)
# ============================================================
def cv_predict_with_per_fold_scaler(
    X: np.ndarray,
    y: np.ndarray,
    model_factory,
    n_splits: int = 5,
    random_state: int = 42,
    scale_X: bool = True,
    scale_y: bool = False,
) -> np.ndarray:
    """
    手动 k 折 CV,确保 StandardScaler 在每折训练集上 fit.

    避免了 sklearn.cross_val_predict + StandardScaler.fit_transform(X) 全集 fit
    导致的特征均值/方差泄漏(对 RF 几乎无影响,但对深度模型/线性模型有影响,
    且审稿人一定会问).

    Parameters
    ----------
    X : (N, D)
    y : (N,) — assumed univariate target
    model_factory : callable that returns a fresh sklearn-like estimator
    n_splits : int
    random_state : int
    scale_X : 是否对特征做 standardization (per-fold)
    scale_y : 是否对目标做 standardization (per-fold);仅用于深度模型场景

    Returns
    -------
    y_pred : (N,) — out-of-fold predictions in the original target scale
    """
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_pred = np.empty_like(y, dtype=float)

    for train_idx, val_idx in kf.split(X):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        if scale_X:
            scaler_X = StandardScaler().fit(X_tr)
            X_tr = scaler_X.transform(X_tr)
            X_va = scaler_X.transform(X_va)

        if scale_y:
            y_tr_mean = float(np.mean(y_tr))
            y_tr_std = float(np.std(y_tr)) or 1.0
            y_tr_use = (y_tr - y_tr_mean) / y_tr_std
        else:
            y_tr_use = y_tr

        model = model_factory()
        model.fit(X_tr, y_tr_use)
        pred_va = model.predict(X_va)

        if scale_y:
            pred_va = pred_va * y_tr_std + y_tr_mean

        y_pred[val_idx] = pred_va

    return y_pred


# ============================================================
# 5. 评估指标(论文 Eq. 4 风格)
# ============================================================
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma_y: Optional[float] = None,
) -> Dict[str, float]:
    """
    计算论文报告的三个指标:
      MAE, R², MAE/σ_y.

    sigma_y: 若未提供,默认用 y_true 的 std (这与论文 Eq. 4 的定义一致:
             σ_y is dataset std).
    """
    from sklearn.metrics import mean_absolute_error, r2_score

    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    if sigma_y is None:
        sigma_y = float(np.std(y_true))
    rel = mae / sigma_y if sigma_y > 0 else float("nan")
    return {
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "MAE_over_std": round(rel, 4),
        "sigma_y": round(sigma_y, 4),
    }


# ============================================================
# 6. JSON 安全保存
# ============================================================
def save_json(obj, path: Path | str, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    log(f"  保存 {path}")


# ============================================================
# 自检
# ============================================================
if __name__ == "__main__":
    # 简单 smoke test
    log("utils.py self-check")
    assert compute_dominant_tm("Na3Fe(PO4)(CO3)") == "Fe"
    assert compute_dominant_tm("Na3V2(PO4)3") == "V"
    assert compute_dominant_tm("Na2TiFe(PO4)3") == "Fe"  # tied -> alphabetical first
    assert compute_dominant_tm("NaCl") is None
    log("  All dominant_tm assertions passed ✓")
