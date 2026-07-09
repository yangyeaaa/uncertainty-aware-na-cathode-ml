"""
Epistemic / aleatoric decomposition for Section 3.4 (Windows-safe).
Run:  python uq_ea_decomposition.py

Method (deep-ensemble style, Lakshminarayanan 2017 applied to NGBoost):
  Train an ensemble of M NGBoost models per CV fold (bootstrap of the
  training fold + distinct seeds). Each member m gives a predictive Normal
  N(mu_m, sigma_m^2); sigma_m^2 is that member's *aleatoric* estimate.
    aleatoric_var(x) = mean_m sigma_m(x)^2          # irreducible data noise
    epistemic_var(x) = var_m  mu_m(x)               # model disagreement (reducible)
    total_var(x)     = aleatoric_var + epistemic_var
  Question answered: for the Fe-voltage cell (high total uncertainty), is it
  epistemic-dominated (more/denser data would help) or aleatoric-dominated
  (composition information has hit its ceiling -> structure is required)?

Outputs (in launch dir):
  - uq_ea_figure.png / .pdf
  - mp_baseline/uq_ea_summary.json
"""
import sys, json, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "paches")
import utils
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from ngboost import NGBRegressor
from ngboost.distns import Normal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = 42
M_ENSEMBLE = 5          # raise to 10 on your machine for smoother epistemic estimates
TARGETS = ["average_voltage", "capacity_grav", "max_delta_volume"]
TLAB = {"average_voltage": "Voltage", "capacity_grav": "Capacity", "max_delta_volume": "DeltaV"}
FAMS = ["Mn", "V", "Fe", "Ni", "Cr", "Cu", "Co", "Ti"]


def ngb(seed):
    return NGBRegressor(Dist=Normal, n_estimators=300, learning_rate=0.04,
                        minibatch_frac=0.8, verbose=False, random_state=seed)


def decompose(X, y):
    """5-fold CV; per fold an M-member bootstrap ensemble of NGBoost.
       Returns per-sample mu, aleatoric_var, epistemic_var, total_var, conformal coverage."""
    n = len(y)
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG)
    mu = np.zeros(n); ale = np.zeros(n); epi = np.zeros(n)
    ql = np.zeros(n); qu = np.zeros(n)
    for tr, te in kf.split(X):
        rs = np.random.RandomState(RNG)
        perm = rs.permutation(len(tr))
        n_cal = max(12, int(0.25 * len(tr)))
        cal_idx = tr[perm[:n_cal]]; fit_idx = tr[perm[n_cal:]]
        xs = StandardScaler().fit(X[fit_idx]); ys = StandardScaler().fit(y[fit_idx].reshape(-1, 1))
        sy = ys.scale_[0]
        Xf_all = xs.transform(X[fit_idx]); yf_all = ys.transform(y[fit_idx].reshape(-1, 1)).ravel()
        Xc = xs.transform(X[cal_idx]); Xt = xs.transform(X[te])

        mus_t = np.zeros((M_ENSEMBLE, len(te))); sgs_t = np.zeros((M_ENSEMBLE, len(te)))
        mus_c = np.zeros((M_ENSEMBLE, len(cal_idx))); sgs_c = np.zeros((M_ENSEMBLE, len(cal_idx)))
        for m in range(M_ENSEMBLE):
            bs = np.random.RandomState(1000 + m).choice(len(fit_idx), len(fit_idx), replace=True)
            model = ngb(seed=1000 + m).fit(Xf_all[bs], yf_all[bs])
            dt = model.pred_dist(Xt); dc = model.pred_dist(Xc)
            mus_t[m] = ys.inverse_transform(dt.loc.reshape(-1, 1)).ravel(); sgs_t[m] = dt.scale * sy
            mus_c[m] = ys.inverse_transform(dc.loc.reshape(-1, 1)).ravel(); sgs_c[m] = dc.scale * sy

        # test-fold decomposition
        mu[te] = mus_t.mean(0)
        ale[te] = (sgs_t ** 2).mean(0)
        epi[te] = mus_t.var(0, ddof=1)
        tot_t = np.sqrt(ale[te] + epi[te])
        # conformal multiplier from calibration set, normalized by total sigma
        mu_c = mus_c.mean(0); tot_c = np.sqrt((sgs_c ** 2).mean(0) + mus_c.var(0, ddof=1))
        score = np.abs(y[cal_idx] - mu_c) / np.maximum(tot_c, 1e-9)
        q90 = np.quantile(score, 0.90)
        ql[te] = mu[te] - q90 * tot_t; qu[te] = mu[te] + q90 * tot_t

    tot = ale + epi
    cover = float(np.mean((y >= ql) & (y <= qu)))
    return dict(mu=mu, ale=ale, epi=epi, tot=tot, y=y, cover=cover,
                vary=float(np.var(y)))


def main():
    df = utils.load_and_filter("mp_baseline/na_electrodes.csv")
    df, feat_cols = utils.magpie_featurize(df, n_jobs=1, verbose=True)
    df = utils.assign_dominant_tm(df, formula_col="formula_discharge")
    X = df[feat_cols].values.astype(float)
    tm = df["dominant_tm"].values
    print(f"[ea] N={len(df)}  features={len(feat_cols)}  ensemble M={M_ENSEMBLE}")

    res = {}
    for tgt in TARGETS:
        res[tgt] = decompose(X, df[tgt].values.astype(float))
        print(f"[{TLAB[tgt]:8s}] conformal-90 coverage={res[tgt]['cover']:.2f}")

    # per-family decomposition table: report variance normalized by target variance
    print("\n=== per-family uncertainty decomposition (share of target variance) ===")
    summary = {}
    for tgt in TARGETS:
        r = res[tgt]; vary = r['vary']; summary[TLAB[tgt]] = {}
        print(f"\n-- {TLAB[tgt]} --   (epi% = epistemic share of total; high epi% = more data helps)")
        print(f"{'TM':>4} {'n':>4} | {'aleatoric':>10} {'epistemic':>10} {'epi%':>6}")
        for f in FAMS:
            mask = tm == f
            a = float(np.mean(r['ale'][mask]) / vary)
            e = float(np.mean(r['epi'][mask]) / vary)
            epct = 100 * e / (a + e) if (a + e) > 0 else 0.0
            summary[TLAB[tgt]][f] = dict(n=int(mask.sum()), aleatoric=a, epistemic=e, epi_pct=epct)
            print(f"{f:>4} {int(mask.sum()):>4} | {a:>10.3f} {e:>10.3f} {epct:>5.0f}%")

    # verdicts
    def verdict(tgt, fam):
        s = summary[TLAB[tgt]][fam]
        kind = "EPISTEMIC-dominated (reducible: more/denser data helps)" if s['epi_pct'] >= 50 \
               else "ALEATORIC-dominated (irreducible from composition: structure needed)"
        return f"{fam} {TLAB[tgt]}: epi%={s['epi_pct']:.0f}% -> {kind}"
    print("\n=== VERDICTS ===")
    print(" ", verdict("average_voltage", "Fe"))
    print(" ", verdict("max_delta_volume", "Fe"))
    print(" ", verdict("average_voltage", "Ni"))
    print(" ", verdict("max_delta_volume", "Ni"))

    json.dump(summary, open("mp_baseline/uq_ea_summary.json", "w"), indent=2)
    print("\n[ea] wrote mp_baseline/uq_ea_summary.json")

    # figure: stacked aleatoric+epistemic per family, Voltage and DeltaV
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for ax, tgt in zip(axes, ["average_voltage", "max_delta_volume"]):
        r = res[tgt]; vary = r['vary']
        a = [np.mean(r['ale'][tm == f]) / vary for f in FAMS]
        e = [np.mean(r['epi'][tm == f]) / vary for f in FAMS]
        x = np.arange(len(FAMS))
        ax.bar(x, a, color="#5a9367", label="aleatoric (irreducible)")
        ax.bar(x, e, bottom=a, color="#c0603a", label="epistemic (reducible)")
        for i, f in enumerate(FAMS):
            if f in ("Fe", "Ni"):
                ax.axvspan(i - 0.5, i + 0.5, color="#ffe9a8", alpha=0.4, zorder=0)
        ax.set_xticks(x); ax.set_xticklabels(FAMS)
        ax.set_ylabel("uncertainty variance / target variance")
        ax.set_title(f"{TLAB[tgt]}: aleatoric vs epistemic")
        ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig("uq_ea_figure.png", dpi=200, bbox_inches="tight")
    plt.savefig("uq_ea_figure.pdf", bbox_inches="tight")
    print("[ea] saved uq_ea_figure.png / .pdf")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
