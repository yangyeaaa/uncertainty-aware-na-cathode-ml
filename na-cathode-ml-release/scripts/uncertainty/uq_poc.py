"""
UQ proof-of-concept for the AMI upgrade.

Question: does *calibrated* predictive uncertainty automatically flag the
chemistry/target cells that the deterministic study found unpredictable
(Fe-voltage, Ni-DeltaV)? If yes, the Fe-Ni "reversal" stops being an
anecdote and becomes a model-introspective, screening-usable signal.

Pipeline (faithful to the paper):
  - same load_and_filter (utils.py)
  - same 132-dim Magpie ElementProperty features
  - same dominant-TM subset assignment
Added:
  - NGBoost heteroscedastic regressor (mean + sigma)
  - split-conformal normalization -> distribution-free 90% intervals
  - per-sample normalized uncertainty; aggregate per TM-family x target
  - calibration (empirical coverage) + error-uncertainty rank correlation
"""
import sys, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "paches")
import utils
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, norm
from ngboost import NGBRegressor
from ngboost.distns import Normal

RNG = 42
TARGETS = ["average_voltage", "capacity_grav", "max_delta_volume"]
TLAB = {"average_voltage": "Voltage", "capacity_grav": "Capacity", "max_delta_volume": "DeltaV"}

# ---- data + features (identical to paper) ----
df = utils.load_and_filter("mp_baseline/na_electrodes.csv")
df, feat_cols = utils.magpie_featurize(df, n_jobs=2, verbose=True)
df = utils.assign_dominant_tm(df, formula_col="formula_discharge")
X = df[feat_cols].values.astype(float)
tm = df["dominant_tm"].values
print(f"[poc] N={len(df)}  features={len(feat_cols)}")

def ngb():
    return NGBRegressor(Dist=Normal, n_estimators=300, learning_rate=0.04,
                        minibatch_frac=0.8, verbose=False, random_state=RNG)

results = {}
for tgt in TARGETS:
    y = df[tgt].values.astype(float)
    n = len(y)
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    mu = np.zeros(n); sig = np.zeros(n); ql = np.zeros(n); qu = np.zeros(n)
    for tr, te in kf.split(X):
        # inner split of train -> fit + conformal-calibration
        rs = np.random.RandomState(RNG)
        perm = rs.permutation(len(tr))
        n_cal = max(12, int(0.20*len(tr)))   # 20% calibration (proper-train 80%)
        cal_idx = tr[perm[:n_cal]]; fit_idx = tr[perm[n_cal:]]
        xs = StandardScaler().fit(X[fit_idx]); ys = StandardScaler().fit(y[fit_idx].reshape(-1,1))
        Xf = xs.transform(X[fit_idx]); yf = ys.transform(y[fit_idx].reshape(-1,1)).ravel()
        m = ngb().fit(Xf, yf)
        # calibration residuals (normalized by predicted sigma) -> conformal score
        d_cal = m.pred_dist(xs.transform(X[cal_idx]))
        mu_c = ys.inverse_transform(d_cal.loc.reshape(-1,1)).ravel()
        sg_c = d_cal.scale * ys.scale_[0]
        EPS = 1e-9
        score = np.abs(y[cal_idx]-mu_c)/np.maximum(sg_c, EPS)
        # finite-sample corrected conformal quantile (Angelopoulos & Bates)
        alpha = 0.10
        q_level = min(1.0, np.ceil((n_cal+1)*(1-alpha))/n_cal)
        q90 = np.quantile(score, q_level)
        # predict test
        d_te = m.pred_dist(xs.transform(X[te]))
        mu[te] = ys.inverse_transform(d_te.loc.reshape(-1,1)).ravel()
        sig[te] = d_te.scale * ys.scale_[0]
        sig_te_floored = np.maximum(sig[te], EPS)
        ql[te] = mu[te]-q90*sig_te_floored; qu[te] = mu[te]+q90*sig_te_floored
    err = np.abs(y-mu)
    cover = float(np.mean((y>=ql)&(y<=qu)))
    rho, p = spearmanr(sig, err)
    nrm_u = sig/np.std(y)                       # uncertainty normalized to target spread
    results[tgt] = dict(mu=mu, sig=sig, err=err, y=y, nrm_u=nrm_u,
                        cover=cover, rho=float(rho), p=float(p))
    print(f"[{TLAB[tgt]:8s}] conformal-90 coverage={cover:.2f}  "
          f"spearman(sigma,|err|)={rho:+.2f} (p={p:.1e})")

# ---- per-TM-family normalized uncertainty table ----
fams = ["Mn","V","Fe","Ni","Cr","Cu","Co","Ti"]
print("\n=== mean normalized uncertainty  (high = model is unsure = low predictability) ===")
print(f"{'TM':>4} {'n':>4} | " + " ".join(f"{TLAB[t]:>9}" for t in TARGETS))
tbl = {}
for f in fams:
    mask = tm==f; row=[]
    for t in TARGETS: row.append(float(np.mean(results[t]['nrm_u'][mask])))
    tbl[f]=dict(n=int(mask.sum()), u=row)
    print(f"{f:>4} {int(mask.sum()):>4} | " + " ".join(f"{v:9.3f}" for v in row))

# rank Fe and Ni within each target
for t in TARGETS:
    order = sorted(fams, key=lambda f: tbl[f]['u'][TARGETS.index(t)], reverse=True)
    fe_rank = order.index("Fe")+1; ni_rank = order.index("Ni")+1
    print(f"  {TLAB[t]:8s}: Fe uncertainty rank {fe_rank}/8, Ni rank {ni_rank}/8  (1=most uncertain)")

json.dump({"families":tbl,
           "calibration":{TLAB[t]:{"coverage":results[t]['cover'],
                                   "spearman_sig_err":results[t]['rho']} for t in TARGETS}},
          open("mp_baseline/uq_poc_summary.json","w"), indent=2)

# stash arrays for plotting
np.savez("mp_baseline/uq_poc_arrays.npz",
         tm=tm,
         **{f"{TLAB[t]}_sig":results[t]['sig'] for t in TARGETS},
         **{f"{TLAB[t]}_err":results[t]['err'] for t in TARGETS},
         **{f"{TLAB[t]}_nrmu":results[t]['nrm_u'] for t in TARGETS})
print("\n[poc] wrote uq_poc_summary.json + uq_poc_arrays.npz")
