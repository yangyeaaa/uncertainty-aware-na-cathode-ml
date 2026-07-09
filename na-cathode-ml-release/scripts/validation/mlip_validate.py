"""
MLIP validation of screened Na-cathode candidates (CHGNet, GPU).
Run:  python mlip_validate.py

Pipeline
--------
1. Rebuild framework -> source-structure map from na_structures.pkl (365 discharged
   training structures) using the same template logic as the generator.
2. Select candidates from screening_shortlist.csv, STRATIFIED by framework family
   (TOP_PER_FAMILY each) so the validation is not dominated by phosphates, capped
   at N_VALIDATE.
3. For each candidate: substitute its TM(s) into the framework's source structure
   -> discharged initial structure; remove Na -> charged initial structure.
4. Relax both with CHGNet; record:
     - relaxed energy per atom (CHGNet, MP-PBE-compatible)
     - volume change |V_charged - V_discharged| / V_discharged  (strain proxy)
     - relaxation sanity (converged, init->relaxed volume drift)
5. Write mp_baseline/mlip_validation.csv and compare the MLIP strain to the model's
   predicted max_delta_volume (does the "low-strain" ranking survive MLIP?).

If chgnet is not installed, the script runs in DRY_RUN: it builds all structures
and reports initial volumes only, so the structure-prep can be verified first.
    pip install chgnet      (then re-run for real relaxation)

Optional second stage (not here): proper e_above_hull via MP phase diagram on the
MLIP survivors (needs MP_API_KEY); cheap relaxation + strain is enough for triage.
"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "paches")
import utils
from pymatgen.core import Composition, Element

# ----------------------------- CONFIG -----------------------------
SHORTLIST = "mp_baseline/screening_shortlist.csv"
STRUCTS = "mp_baseline/na_structures.pkl"
OUT = "mp_baseline/mlip_validation.csv"
DEVICE = "cuda"            # "cuda" (you have GPU) or "cpu"
FMAX = 0.1                 # relaxation force tolerance (eV/A)
STEPS = 200               # max relaxation steps
TOP_PER_FAMILY = 3         # stratified: keep this many top candidates per framework
N_VALIDATE = 30            # overall cap
RNG = 42
TM = ["Mn", "V", "Fe", "Ni", "Cr", "Cu", "Co", "Ti"]
ALLOWED = {"Na", "O", "P", "S", "F", "Si", "As", "B"}
# ------------------------------------------------------------------


def to_template(formula):
    try:
        d = dict(Composition(formula).reduced_composition.get_el_amt_dict())
    except Exception:
        return None
    tms = [t for t in TM if d.get(t, 0) > 0]
    others = {e: a for e, a in d.items() if e not in TM}
    if len(tms) != 1 or any(e not in ALLOWED for e in others) or not others:
        return None
    return (frozenset(others.items()), d[tms[0]])


def build_template_map(structs):
    tr = utils.load_and_filter("mp_baseline/na_electrodes.csv")
    tmap = {}
    for _, row in tr.iterrows():
        f, bid = row["formula_discharge"], row["battery_id"]
        t = to_template(f)
        if t is None or bid not in structs:
            continue
        d = Composition(f).get_el_amt_dict()
        dom = max([(x, d[x]) for x in TM if d.get(x, 0) > 0], key=lambda z: z[1])[0]
        tmap.setdefault(t, []).append((bid, dom))
    return tmap


def make_structures(formula, structs, tmap):
    """Return (discharged_struct, charged_struct) or (None, None) if no template."""
    t = to_template(formula)
    comp = Composition(formula).get_el_amt_dict()
    tms = [(x, comp[x]) for x in TM if comp.get(x, 0) > 0]
    # binary-mix candidates have len(tms)==2 and template detection (single-TM) fails;
    # recover template by collapsing both TMs into one slot
    if t is None and len(tms) == 2:
        d = dict(Composition(formula).reduced_composition.get_el_amt_dict())
        others = {e: a for e, a in d.items() if e not in TM}
        slot = sum(comp[x] for x, _ in tms) / (sum(comp.values()) / sum(d.values()))
        t = (frozenset(others.items()), d.get(tms[0][0], 0) + d.get(tms[1][0], 0))
    if t not in tmap:
        return None, None
    src_bid, src_dom = tmap[t][0]
    S = structs[src_bid].copy()
    # substitute the source dominant TM sites with the candidate TM(s)
    if len(tms) == 1:
        S.replace_species({Element(src_dom): Element(tms[0][0])})
    else:
        # split src_dom sites between the two target TMs by molar ratio
        idx = [i for i, s in enumerate(S) if s.specie.symbol == src_dom]
        frac0 = tms[0][1] / (tms[0][1] + tms[1][1])
        n0 = max(1, int(round(len(idx) * frac0)))
        for j, i in enumerate(idx):
            S[i] = Element(tms[0][0] if j < n0 else tms[1][0])
    S_charged = S.copy(); S_charged.remove_species(["Na"])
    return S, S_charged


def stratified_select(shortlist):
    df = pd.read_csv(shortlist)
    if "framework" not in df.columns:
        df["framework"] = "all"
    picks = []
    for fam, g in df.sort_values("composite_score", ascending=False).groupby("framework", sort=False):
        picks.append(g.head(TOP_PER_FAMILY))
    sel = pd.concat(picks).sort_values("composite_score", ascending=False).head(N_VALIDATE)
    return sel.reset_index(drop=True)


def main():
    structs = pickle.load(open(STRUCTS, "rb"))
    tmap = build_template_map(structs)
    print(f"[mlip] templates with source structures: {len(tmap)}")
    sel = stratified_select(SHORTLIST)
    print(f"[mlip] stratified candidates selected: {len(sel)} "
          f"(<= {TOP_PER_FAMILY}/framework, cap {N_VALIDATE})")

    # Load CHGNet as an ASE calculator and relax with the CURRENT ase API directly.
    # Reason: chgnet 0.3.0's built-in StructOptimizer uses ExpCellFilter, removed in
    # new ase (3.26); driving ase ourselves (FrechetCellFilter + FIRE) is version-proof.
    # NVML fix: replace the free-mem GPU query (nvml.dll missing) so it uses cuda:0.
    calc = None; FILT = None; FIRE = None
    try:
        import torch
        import chgnet.utils.common_utils as _cu
        _cu.cuda_devices_sorted_by_free_mem = lambda: [0]      # bypass NVML
        from chgnet.model.dynamics import CHGNetCalculator
        from ase.optimize import FIRE
        try:
            from ase.filters import FrechetCellFilter as FILT
        except Exception:
            from ase.constraints import ExpCellFilter as FILT
        dev = "cuda" if (DEVICE.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        calc = CHGNetCalculator(use_device=dev)
        print(f"[mlip] CHGNet calculator ready on {dev} (driving ase {FILT.__name__} + FIRE)")
    except ImportError as e:
        print(f"[mlip] CHGNet/ase not available ({e}) -> DRY-RUN (structures only)")
    relaxer = calc  # non-None means "do relaxation"

    from pymatgen.io.ase import AseAtomsAdaptor

    def relax(struct):
        atoms = AseAtomsAdaptor.get_atoms(struct)
        atoms.calc = calc
        dyn = FIRE(FILT(atoms), logfile=None)
        dyn.run(fmax=FMAX, steps=STEPS)
        e = float(atoms.get_potential_energy())
        relaxed = AseAtomsAdaptor.get_structure(atoms)
        return relaxed, e, int(dyn.nsteps)

    rows = []
    for _, c in sel.iterrows():
        f = c["formula_discharge"]
        Sd, Sc = make_structures(f, structs, tmap)
        rec = dict(formula_discharge=f,
                   framework=c.get("framework"),
                   composite_score=c.get("composite_score"),
                   pred_delta_volume=c.get("max_delta_volume_pred"),
                   pred_voltage=c.get("average_voltage_pred"))
        if Sd is None:
            rec["status"] = "no_template_structure"; rows.append(rec); continue
        rec["V_discharged_init"] = float(Sd.volume)
        rec["V_charged_init"] = float(Sc.volume)
        if relaxer is None:
            rec["status"] = "dry_run"; rows.append(rec); continue
        try:
            Sd_r, Ed, nd = relax(Sd)
            Sc_r, Ec, nc = relax(Sc)
            vdv = abs(Sc_r.volume - Sd_r.volume) / Sd_r.volume
            rec.update(
                e_per_atom_discharged=Ed / len(Sd_r),
                V_discharged_relaxed=float(Sd_r.volume),
                V_charged_relaxed=float(Sc_r.volume),
                mlip_delta_volume=vdv,
                relax_steps_dis=nd, relax_steps_cha=nc,
                vol_drift_dis=abs(Sd_r.volume - Sd.volume) / Sd.volume,
                status="ok")
        except Exception as e:
            rec["status"] = f"relax_fail:{type(e).__name__}"
            rec["error"] = str(e)[:200]
            if not any("error" in r and r.get("status", "").startswith("relax_fail") for r in rows):
                import traceback; print("[mlip] first relax failure:\n", traceback.format_exc()[-800:])
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\n[mlip] wrote {OUT}  ({len(out)} rows)")
    ok = out[out["status"] == "ok"] if "status" in out else out
    if len(ok) and "mlip_delta_volume" in ok:
        from scipy.stats import spearmanr
        rho, p = spearmanr(ok["pred_delta_volume"], ok["mlip_delta_volume"])
        print(f"[mlip] predicted vs MLIP delta_volume: spearman={rho:+.2f} (p={p:.1e}, n={len(ok)})")
        print(f"[mlip] mean MLIP strain of validated set = {ok['mlip_delta_volume'].mean():.3f}")
        cols = ["formula_discharge", "framework", "pred_delta_volume", "mlip_delta_volume",
                "pred_voltage", "e_per_atom_discharged"]
        print("\n=== validated candidates (sorted by MLIP strain) ===")
        print(ok.sort_values("mlip_delta_volume")[cols].to_string(index=False))
    else:
        print("[mlip] DRY-RUN summary (initial volumes built; install chgnet for relaxation):")
        print(out[["formula_discharge", "framework", "V_discharged_init", "V_charged_init", "status"]]
              .head(30).to_string(index=False))


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
