"""
Leave-One-Chemistry-Out (LOCO) generalisation test for the RF+Magpie baseline.
Run:  python loco_baseline.py       (same env as baseline_na_v2.py)

Why: the manuscript's headline numbers come from KFold(5, shuffle) CV. A
standard reviewer objection at npj CM / CEJ is that random splits leak
chemically similar compounds across train/test, inflating apparent
generalisation. LOCO answers the sharper question actually relevant to
screening: "trained on all OTHER transition-metal families, how well does the
model predict a chemistry it has NEVER seen?" This is the honest upper bound on
extrapolative discovery and directly stress-tests the Fe-Ni reversal claim.

Protocol (mirrors the benchmark except for the split):
  * same load_and_filter + magpie_featurize + dominant-TM assignment as utils;
  * for each of the 8 TM families, hold that entire family out as the test set,
    train one RF per target on the remaining families, predict the held-out
    family; report per-family, per-target R2 / MAE / MAE-over-sigma;
  * also report a MACRO summary (mean over families) and, for reference, the
    in-distribution random-CV numbers on the same feature matrix, so the
    LOCO-vs-CV gap is explicit.

Output: mp_baseline/loco_baseline_per_family.csv
        mp_baseline/loco_baseline_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "paches")
from utils import (log, load_and_filter, magpie_featurize, assign_dominant_tm,
                   compute_metrics, cv_predict_with_per_fold_scaler,
                   DEFAULT_TARGETS, TM_LIST)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

ELECTRODES = "mp_baseline/na_electrodes.csv"
OUT_DIR = Path("mp_baseline")
TARGETS = DEFAULT_TARGETS
N_JOBS = 2

# RF config identical to the benchmark (baseline_na_v2 / Table S1)
RF_KW = dict(n_estimators=150, max_depth=20, min_samples_leaf=2,
             random_state=42, n_jobs=N_JOBS)


def rf():
    return RandomForestRegressor(**RF_KW)


def sigma(y):
    return float(np.std(y)) if np.std(y) > 0 else 1.0


def loco_eval(df, feature_cols):
    """One RF per (held-out family, target). Trains on all other families,
    predicts the held-out family. Feature scaler fitted on TRAIN only."""
    X = df[feature_cols].values.astype(np.float64)
    fam = df["dominant_tm"].values
    families = [t for t in TM_LIST if (fam == t).sum() > 0]
    rows = []
    for held in families:
        te = fam == held
        tr = ~te & pd.notna(fam)          # train = other assigned families
        n_te, n_tr = int(te.sum()), int(tr.sum())
        sc = StandardScaler().fit(X[tr])   # tree-insensitive, kept for parity
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for t in TARGETS:
            y = df[t].values.astype(np.float64)
            model = rf().fit(Xtr, y[tr])
            yp = model.predict(Xte)
            yt = y[te]
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
            mae = float(np.mean(np.abs(yt - yp)))
            rows.append(dict(family=held, n_train=n_tr, n_test=n_te,
                             target=t, r2=round(r2, 4), mae=round(mae, 4),
                             mae_over_sigma=round(mae / sigma(yt), 4)))
            log(f"  hold-out {held:3s} ({n_te:2d}) {t:18s} "
                f"R2={r2:+.3f}  MAE/sigma={mae/sigma(yt):.3f}")
    return pd.DataFrame(rows), families


def cv_reference(df, feature_cols):
    """In-distribution random-5-fold numbers on the SAME feature matrix, so
    the LOCO-vs-CV degradation is quantified rather than asserted."""
    out = {}
    X = df[feature_cols].values.astype(np.float64)
    for t in TARGETS:
        y = df[t].values.astype(np.float64)
        yp = cv_predict_with_per_fold_scaler(       # returns OOF preds array
            X, y, model_factory=rf, n_splits=5, random_state=42)
        m = compute_metrics(y, yp)
        out[t] = {"r2": round(float(m["R2"]), 4),
                  "mae": round(float(m["MAE"]), 4)}
    return out


def main():
    log("LOCO generalisation test for RF+Magpie")
    df = load_and_filter(ELECTRODES, targets=TARGETS)
    df, feature_cols = magpie_featurize(df, n_jobs=N_JOBS)
    df = assign_dominant_tm(df)
    n_assigned = int(pd.notna(df["dominant_tm"]).sum())
    log(f"assigned {n_assigned}/{len(df)} pairs to 8 TM families "
        f"({len(df)-n_assigned} non-TM excluded from LOCO)")

    per_fam, families = loco_eval(df, feature_cols)
    per_fam.to_csv(OUT_DIR / "loco_baseline_per_family.csv", index=False)

    # try the CV reference; if the utils signature differs, skip gracefully
    try:
        cv_ref = cv_reference(df, feature_cols)
    except Exception as e:
        log(f"CV reference skipped ({type(e).__name__}: {e})")
        cv_ref = None

    summary = {"macro": {}, "cv_reference_random5fold": cv_ref,
               "families": families, "n_assigned": n_assigned}
    print("\n=== LOCO per-target MACRO (mean over held-out families) ===")
    for t in TARGETS:
        sub = per_fam[per_fam["target"] == t]
        macro_r2 = round(float(sub["r2"].mean()), 4)
        macro_ms = round(float(sub["mae_over_sigma"].mean()), 4)
        summary["macro"][t] = {"loco_r2_macro_mean": macro_r2,
                               "loco_r2_macro_std": round(float(sub["r2"].std(ddof=1)), 4),
                               "loco_mae_over_sigma_macro": macro_ms}
        cvtxt = (f"  |  random-5fold R2={cv_ref[t]['r2']:+.3f}"
                 if cv_ref else "")
        print(f"  {t:18s} LOCO macro-R2 = {macro_r2:+.3f} "
              f"(std {sub['r2'].std(ddof=1):.3f}){cvtxt}")

    with open(OUT_DIR / "loco_baseline_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nwritten: {OUT_DIR/'loco_baseline_per_family.csv'}")
    print(f"written: {OUT_DIR/'loco_baseline_summary.json'}")
    print("\nInterpretation: LOCO R2 << random-CV R2 is EXPECTED and is the "
          "honest extrapolation bound; the Fe/Ni contrast under LOCO is the "
          "key thing to read for the reversal claim.")


if __name__ == "__main__":
    main()
