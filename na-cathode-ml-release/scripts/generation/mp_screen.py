"""
MP held-out pool construction + confidence-aware screening (Windows-safe).

Pipeline
--------
1. Load the 365 training electrodes (utils.load_and_filter) + Magpie features.
2. Build the candidate pool:
     - REAL mode (MP_API_KEY set): query Materials Project for ALL Na insertion
       electrodes, drop the training battery_ids, apply the same stability filter.
       These carry charge/discharge structures (id_charge/id_discharge) for later
       MLIP validation.
     - MOCK mode (no key): hold out 18% of the 365 as pseudo-candidates so the
       whole scoring chain can be verified offline. Prints retrospective accuracy.
3. Train a conformal-NGBoost ENSEMBLE on the training set (per target):
     mean, total sigma, epistemic, aleatoric, and split-conformal 90% bounds.
4. Confidence-aware score (materials-correct directions):
       voltage  : higher better  -> use lower conf. bound  (LCB = mu - q*sigma)
       capacity : higher better  -> LCB
       max_delta_volume (strain) : LOWER better -> use upper bound (UCB = mu + q*sigma)
   Composite = w_V*z(LCB_V) + w_C*z(LCB_C) - w_DV*z(UCB_DV).
   A candidate ranks high only if it looks good even under the pessimistic,
   calibrated estimate -> uncertainty is penalised automatically.
5. Output ranked shortlist CSV (+ mp-ids for MLIP) and print top-k.

Usage
-----
    set MP_API_KEY=...        (Windows)   /   export MP_API_KEY=...  (Linux/mac)
    python mp_screen.py
Edit CONFIG below to change targets, weights, hard gates, top_k, ensemble size.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "paches")
import utils
from sklearn.preprocessing import StandardScaler
from ngboost import NGBRegressor
from ngboost.distns import Normal

# ----------------------------- CONFIG -----------------------------
CSV = "mp_baseline/na_electrodes.csv"
CANDIDATE_SOURCE = "csv"     # "csv" (generated pool) | "mp" (held-out MP) | "mock" (self-test)
CANDIDATE_CSV = "mp_baseline/generated_candidates.csv"   # used when CANDIDATE_SOURCE == "csv"
TARGETS = ["average_voltage", "capacity_grav", "max_delta_volume"]
DIRECTION = {"average_voltage": +1, "capacity_grav": +1, "max_delta_volume": -1}  # +1 higher better, -1 lower better
WEIGHTS = {"average_voltage": 1.0, "capacity_grav": 1.0, "max_delta_volume": 1.0}
EHULL_MAX = 0.1          # stability_discharge gate (eV/atom), same as training
M_ENSEMBLE = 10          # ensemble members (epistemic estimate); 5 is faster
TOP_K = 25
RNG = 42
OUT_CSV = "mp_baseline/screening_shortlist.csv"
# ------------------------------------------------------------------


def ngb(seed):
    return NGBRegressor(Dist=Normal, n_estimators=300, learning_rate=0.04,
                        minibatch_frac=0.8, verbose=False, random_state=seed)


def fit_conformal_ensemble(Xtr, ytr):
    """Train M-member bootstrap NGBoost ensemble on a fit split; calibrate q90 on a holdout.
       Returns a predictor closure: X -> (mu, sigma_total, epi, ale, q90)."""
    rs = np.random.RandomState(RNG)
    perm = rs.permutation(len(ytr))
    n_cal = max(15, int(0.2 * len(ytr)))
    cal = perm[:n_cal]; fit = perm[n_cal:]
    xs = StandardScaler().fit(Xtr[fit]); ys = StandardScaler().fit(ytr[fit].reshape(-1, 1))
    sy = ys.scale_[0]
    Xf = xs.transform(Xtr[fit]); yf = ys.transform(ytr[fit].reshape(-1, 1)).ravel()
    models = []
    for m in range(M_ENSEMBLE):
        bs = np.random.RandomState(1000 + m).choice(len(fit), len(fit), replace=True)
        models.append(ngb(1000 + m).fit(Xf[bs], yf[bs]))

    def predict(X):
        Xz = xs.transform(X)
        mus = np.zeros((M_ENSEMBLE, len(X))); sgs = np.zeros((M_ENSEMBLE, len(X)))
        for i, mdl in enumerate(models):
            d = mdl.pred_dist(Xz)
            mus[i] = ys.inverse_transform(d.loc.reshape(-1, 1)).ravel()
            sgs[i] = d.scale * sy
        mu = mus.mean(0); ale = (sgs ** 2).mean(0); epi = mus.var(0, ddof=1)
        return mu, np.sqrt(ale + epi), epi, ale
    # conformal q90 on calibration split
    mu_c, sg_c, _, _ = predict(Xtr[cal])
    q90 = float(np.quantile(np.abs(ytr[cal] - mu_c) / np.maximum(sg_c, 1e-9), 0.90))
    return predict, q90


def fetch_mp_candidates(train_ids):
    """Query MP for Na insertion electrodes; drop training ids; apply stability gate.
       Returns dataframe with the same columns as na_electrodes.csv (+ id_charge/id_discharge).

       Note: use_document_model=False returns raw dicts, bypassing the pydantic
       InsertionElectrodeDoc validation that fails on non-integer MPIDs
       (e.g. id_charge='mp-aaaaaaeu'), a known mp-api client bug."""
    from mp_api.client import MPRester
    key = os.environ.get("MP_API_KEY")
    fields = ["battery_id", "formula_charge", "formula_discharge", "average_voltage",
              "capacity_grav", "capacity_vol", "energy_grav", "energy_vol",
              "max_voltage_step", "stability_charge", "stability_discharge",
              "fracA_charge", "fracA_discharge", "max_delta_volume", "num_steps",
              "id_charge", "id_discharge"]
    with MPRester(key) as mpr:
        try:
            res = mpr.materials.insertion_electrodes
        except AttributeError:
            res = mpr.insertion_electrodes
        res.use_document_model = False          # <-- return raw dicts, skip validation
        docs = res.search(working_ion="Na", fields=fields)

    def get(d, f):
        return d.get(f) if isinstance(d, dict) else getattr(d, f, None)
    rows = [{f: get(d, f) for f in fields} for d in docs]
    cand = pd.DataFrame(rows)
    cand = cand[~cand["battery_id"].isin(train_ids)]
    cand = cand[cand["average_voltage"] > 0]
    cand = cand[cand["stability_discharge"] < EHULL_MAX]
    cand = cand.dropna(subset=["formula_discharge"] + TARGETS).reset_index(drop=True)
    return cand


def main():
    # ---- training set ----
    tr = utils.load_and_filter(CSV)
    tr, feat_cols = utils.magpie_featurize(tr, n_jobs=1, verbose=True)
    Xtr = tr[feat_cols].values.astype(float)
    print(f"[screen] training N={len(tr)}  features={len(feat_cols)}")

    # known MP formulas (raw 416 + cleaned training) for leakage-free exclusion
    from pymatgen.core import Composition
    def red(f):
        try: return Composition(f).reduced_formula
        except Exception: return None
    raw = pd.read_csv(CSV)
    known_red = set(raw["formula_discharge"].apply(red).dropna()) | set(tr["formula_discharge"].apply(red).dropna())

    # ---- candidate pool ----
    src = CANDIDATE_SOURCE
    if src == "mock":
        print("[screen] MOCK self-test (hold out 18% of training as candidates)")
        rs = np.random.RandomState(RNG); idx = rs.permutation(len(tr))
        ncand = int(0.18 * len(tr))
        cand = tr.iloc[idx[:ncand]].reset_index(drop=True)
        tr = tr.iloc[idx[ncand:]].reset_index(drop=True)
        Xtr = tr[feat_cols].values.astype(float)
    elif src == "csv":
        cand = pd.read_csv(CANDIDATE_CSV)
        print(f"[screen] candidate CSV: {CANDIDATE_CSV}  rows={len(cand)}")
    elif src == "mp":
        cand = fetch_mp_candidates(set())            # exclusion done below by composition
    else:
        raise ValueError(f"unknown CANDIDATE_SOURCE: {src}")

    # leakage-free: drop candidates whose composition is already known, dedup by composition
    if src != "mock":
        cand = cand.copy()
        cand["_red"] = cand["formula_discharge"].apply(red)
        before = len(cand)
        cand = cand[~cand["_red"].isin(known_red)]
        cand = cand.dropna(subset=["_red"]).drop_duplicates("_red").reset_index(drop=True)
        print(f"[screen] candidates after novelty filter + dedup: {len(cand)} (from {before})")
        cand, _ = utils.magpie_featurize(cand, n_jobs=1, verbose=True)
    Xc = cand[feat_cols].values.astype(float)

    # ---- per-target conformal-ensemble predictions on candidates ----
    pred = {}
    for tgt in TARGETS:
        predict, q90 = fit_conformal_ensemble(Xtr, tr[tgt].values.astype(float))
        mu, sig, epi, ale = predict(Xc)
        pred[tgt] = dict(mu=mu, sig=sig, epi=epi, ale=ale, q90=q90,
                         lcb=mu - q90 * sig, ucb=mu + q90 * sig)

    # ---- confidence-aware composite score ----
    def z(v):
        v = np.asarray(v, float); s = v.std()
        return (v - v.mean()) / s if s > 0 else v * 0.0
    composite = np.zeros(len(cand))
    for tgt in TARGETS:
        cons = pred[tgt]["lcb"] if DIRECTION[tgt] > 0 else pred[tgt]["ucb"]  # conservative estimate
        composite += WEIGHTS[tgt] * DIRECTION[tgt] * z(cons)
    mean_norm_sig = np.mean([pred[t]["sig"] / tr[t].std() for t in TARGETS], axis=0)

    # ---- assemble output ----
    out = pd.DataFrame({
        "formula_discharge": cand["formula_discharge"],
        "composite_score": composite,
        "confidence_rank_sigma": mean_norm_sig,  # lower = more confident
    })
    for tgt in TARGETS:
        out[f"{tgt}_pred"] = pred[tgt]["mu"]
        out[f"{tgt}_sigma"] = pred[tgt]["sig"]
        out[f"{tgt}_lcb"] = pred[tgt]["lcb"]
        out[f"{tgt}_ucb"] = pred[tgt]["ucb"]
        out[f"{tgt}_epi_frac"] = pred[tgt]["epi"] / np.maximum(pred[tgt]["epi"] + pred[tgt]["ale"], 1e-12)
    if src == "mp":
        out["id_charge"] = cand.get("id_charge")
        out["id_discharge"] = cand.get("id_discharge")
    if src == "csv":
        for c in ("framework", "substitution"):
            if c in cand.columns:
                out[c] = cand[c].values
    if src == "mock":  # ground truth available for retrospective validation
        for tgt in TARGETS:
            out[f"{tgt}_true"] = cand[tgt].values

    out = out.sort_values("composite_score", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[screen] wrote {OUT_CSV}  ({len(out)} candidates)")

    cols = ["formula_discharge", "composite_score",
            "average_voltage_pred", "capacity_grav_pred", "max_delta_volume_pred",
            "confidence_rank_sigma"]
    print(f"\n=== TOP {min(TOP_K, len(out))} confidence-aware candidates ===")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(out[cols].head(TOP_K).to_string(index=False,
              formatters={c: (lambda x: f"{x:.3f}") for c in cols if c != "formula_discharge"}))

    if src == "mock":
        from scipy.stats import spearmanr
        print("\n=== MOCK retrospective check (predicted vs true on candidates) ===")
        for tgt in TARGETS:
            mu = pred[tgt]["mu"]; tv = cand[tgt].values
            mae = float(np.mean(np.abs(mu - tv))); rho, _ = spearmanr(mu, tv)
            print(f"  {tgt:18s} MAE={mae:.3f}  spearman(pred,true)={rho:+.2f}")
        # does confidence-aware top-k actually have low true strain?
        k = min(TOP_K, len(out))
        top_true_dv = out["max_delta_volume_true"].head(k).mean()
        all_true_dv = out["max_delta_volume_true"].mean()
        print(f"  top-{k} mean TRUE delta_volume = {top_true_dv:.4f}  vs pool mean = {all_true_dv:.4f}  "
              f"({'lower strain selected OK' if top_true_dv < all_true_dv else 'check weights'})")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
