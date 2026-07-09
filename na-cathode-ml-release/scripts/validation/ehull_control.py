"""
POSITIVE CONTROL for the self-consistent CHGNet hull pipeline.
Run:  python ehull_control.py     (same env; reuses ehull_relax_cache.json)

Rationale: the main run reported 14/15 candidates at E_hull -0.10 to -0.50
eV/atom -- far below the ~0.05 scale of genuine new-stable-phase discoveries,
i.e. a systematic-bias fingerprint. This script feeds REAL Materials Project
discharged cathodes (the very templates the candidate generator used) through
the IDENTICAL machinery. Expected outcome for MP-stable controls:
CHGNet E_hull ~ 0 +/- 0.05.

Verdict logic:
  controls also strongly negative  -> pipeline bug (hull side);
  controls ~0, candidates -0.3     -> bias lives in the GENERATED structures
                                      (substitution artefact / CHGNet
                                      pathology on substituted cells).

Reference relaxations are shared with the main run via the same cache file,
so only the few control structures themselves need relaxing (fast).
"""
import json
import os
import sys
import time
import traceback
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "paches")
import utils
from pymatgen.core import Composition

# reuse the module-level helpers from the main script (same directory)
from ehull_validate import (clean_structure, to_primitive,
                            select_reference_phases, fetch_gga_entries, log,
                            RELAX_CACHE_FILE, FMAX, STEPS, DEVICE,
                            NEAR_STABLE, EHULL_STABLE, TM)

STRUCTS = "mp_baseline/na_structures.pkl"
OUT = "mp_baseline/ehull_control.csv"
N_CONTROL = 6
CONTROL_TMS = {"Ni", "Co", "Fe", "Cu"}   # chemsys already relaxed in main run


def main():
    import pickle
    tr = utils.load_and_filter("mp_baseline/na_electrodes.csv")
    structs = pickle.load(open(STRUCTS, "rb"))

    # MP-side stability column (name-robust)
    stab_col = next(c for c in tr.columns if "stability" in c.lower()
                    and "discharge" in c.lower())

    # controls: single-TM Na-TM-P-O discharged phases, most MP-stable first,
    # restricted to chemsys the main run has already relaxed (cache hits)
    def eligible(row):
        try:
            els = {e.symbol for e in Composition(row["formula_discharge"]).elements}
        except Exception:
            return False
        tms = els & set(TM)
        return (len(tms) == 1 and next(iter(tms)) in CONTROL_TMS
                and els <= (tms | {"Na", "P", "O"})
                and row["battery_id"] in structs)

    cand = tr[tr.apply(eligible, axis=1)].sort_values(stab_col)
    # one per TM first, then fill up to N_CONTROL by stability
    seen_tm, rows = set(), []
    for _, r in cand.iterrows():
        tm = next(iter({e.symbol for e in
                        Composition(r["formula_discharge"]).elements} & set(TM)))
        if tm not in seen_tm or len(rows) < N_CONTROL:
            if tm in seen_tm and len(rows) >= N_CONTROL:
                continue
            rows.append(r); seen_tm.add(tm)
        if len(rows) >= N_CONTROL and seen_tm >= {"Ni", "Co"}:
            break
    controls = rows[:N_CONTROL]
    log(f"controls: {[r['formula_discharge'] for r in controls]}")

    # ---- identical CHGNet machinery (mirrors ehull_validate.main) -------
    import torch
    import chgnet.utils.common_utils as _cu
    _cu.cuda_devices_sorted_by_free_mem = lambda: [0]
    from chgnet.model.dynamics import CHGNetCalculator
    from ase.optimize import FIRE
    try:
        from ase.filters import FrechetCellFilter as FILT
    except Exception:
        from ase.constraints import ExpCellFilter as FILT
    from pymatgen.io.ase import AseAtomsAdaptor
    from mp_api.client import MPRester
    from pymatgen.entries.computed_entries import ComputedEntry
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    dev = "cuda" if (DEVICE.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    calc = CHGNetCalculator(use_device=dev)
    key = os.environ.get("pQ9JHaAgk01LuS58n8Zgen7ty0QhDp9J")

    relax_cache = {}
    if os.path.exists(RELAX_CACHE_FILE):
        try:
            relax_cache = json.load(open(RELAX_CACHE_FILE))
            log(f"resumed {len(relax_cache)} relaxed phases from cache")
        except Exception:
            log("cache corrupt -- starting fresh")
            os.replace(RELAX_CACHE_FILE, RELAX_CACHE_FILE + ".corrupt")

    def _dump():
        with open(RELAX_CACHE_FILE, "w") as fh:
            json.dump(relax_cache, fh)

    def relax_energy(struct):
        s = clean_structure(struct)
        atoms = AseAtomsAdaptor.get_atoms(s)
        atoms.calc = calc
        FIRE(FILT(atoms), logfile=None).run(fmax=FMAX, steps=STEPS)
        return AseAtomsAdaptor.get_structure(atoms), float(atoms.get_potential_energy())

    def relaxed_entry(struct, cache_key, label):
        if cache_key in relax_cache:
            r = relax_cache[cache_key]
            return ComputedEntry(Composition(r["comp"]), r["E"],
                                 entry_id=cache_key)
        s0 = to_primitive(struct)
        t0 = time.time()
        Sr, Etot = relax_energy(s0)
        log(f"      {label} ({len(s0)} atoms) relaxed in {time.time()-t0:.0f}s "
            f"[{Etot/len(s0):+.3f} eV/atom]")
        relax_cache[cache_key] = {"comp": str(Sr.composition),
                                  "E": float(Etot), "fallback": 0}
        _dump()
        return ComputedEntry(Sr.composition, float(Etot), entry_id=cache_key)

    pd_cache = {}

    def get_cg_pd(elements):
        chemsys = "-".join(sorted(elements))
        if chemsys in pd_cache:
            return pd_cache[chemsys]
        with MPRester(key) as mpr:
            entries = fetch_gga_entries(mpr, elements,
                                        logf=lambda m: print(m, flush=True))
        near = select_reference_phases(entries, set(elements),
                                       logf=lambda m: print(m, flush=True))
        cg = []
        for i, e in enumerate(near, 1):
            eid = str(getattr(e, "entry_id", None)
                      or str(e.composition) + str(round(e.energy, 3)))
            cg.append(relaxed_entry(e.structure, eid,
                                    f"({i}/{len(near)}) {eid} "
                                    f"{e.composition.reduced_formula}"))
        phased = PhaseDiagram(cg)
        pd_cache[chemsys] = phased
        return phased

    # ---- run controls ----------------------------------------------------
    recs = []
    for i, r in enumerate(controls, 1):
        f, bid = r["formula_discharge"], r["battery_id"]
        mp_stab = float(r[stab_col])
        log(f"[{i}/{len(controls)}] control {f}  (MP E_hull = {mp_stab:+.4f})")
        rec = dict(formula=f, battery_id=bid, mp_e_hull=mp_stab)
        try:
            S = structs[bid]
            elems = {str(e) for e in S.composition.elements}
            phased = get_cg_pd(elems)
            ce = relaxed_entry(S, f"ctrl::{bid}", f"control {f}")
            dec, eh = phased.get_decomp_and_e_above_hull(ce, allow_negative=True)
            rec["decomposition"] = ", ".join(
                f"{p.composition.reduced_formula}:{x:.2f}"
                for p, x in sorted(dec.items(), key=lambda t: -t[1]))
            rec.update(chgnet_e_hull=round(float(eh), 4),
                       delta=round(float(eh) - mp_stab, 4), status="ok")
            log(f"   -> CHGNet E_hull = {eh:+.4f}  (MP {mp_stab:+.4f}, "
                f"delta {eh - mp_stab:+.4f})")
        except Exception:
            rec.update(status="fail", error=traceback.format_exc()[-400:])
            traceback.print_exc()
        recs.append(rec)

    out = pd.DataFrame(recs)
    out.to_csv(OUT, index=False)
    print("\n=== POSITIVE CONTROL: MP-stable discharged cathodes ===")
    print(out.to_string(index=False))
    ok = out[out["status"] == "ok"]
    if len(ok):
        bias = ok["delta"].mean()
        print(f"\nmean(CHGNet E_hull - MP E_hull) = {bias:+.4f} eV/atom")
        print("interpretation: |mean delta| < ~0.05 -> pipeline sane, bias "
              "lives in the GENERATED candidates;")
        print("                strongly negative      -> hull-side pipeline "
              "bug, candidates were never trustworthy.")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
