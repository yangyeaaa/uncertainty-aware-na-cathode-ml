"""
Stage-2 stability screening with a SELF-CONSISTENT CHGNet convex hull.  (v3.1)
Run:  python ehull_validate.py     (mlip env; needs MP_API_KEY; CHGNet on cpu)

v3.1 CRITICAL FIX (found via positive control + cache forensics):
  get_entries_in_chemsys returned GGA, GGA+U AND r2SCAN entries; their raw
  energies live on incompatible scales, so the MP pre-hull used for reference
  SELECTION was dominated by low-raw-energy r2SCAN entries and silently
  excluded most GGA/GGA+U compounds -- e.g. the Na-Ni-P-O reference set
  contained NO Ni-O/Ni-P-O/quaternary phase at all, and oxide candidates
  "decomposed" into metallic Ni + O2. Result: hull under-covered, everything
  ~-0.3 to -0.5 eV/atom "stable(new)" (controls included). Fix: restrict to
  the GGA/GGA+U corrected scale (CHGNet's own training scale) at the API
  level, with a defensive local post-filter; plus a decomposition-sanity
  warning and a `decomposition` csv column so under-coverage can never hide.

v3 = v2 (tiered reference selection) + hardening after a silent all-fail run:
  P1  PREFLIGHT self-test: before touching MP, relax a local 2-atom Na cell,
      round-trip the json cache, and build a toy hull with allow_negative.
      Any environment breakage surfaces with a full traceback in <1 minute.
  P2  Structures are CLEANED before relaxation (oxidation states and ALL site
      properties stripped). MP thermo-entry structures often carry `magmom`
      (sometimes vector-valued) which can crash the ASE conversion, while the
      stage-1 candidate structures from na_structures.pkl do not -- the prime
      suspect for "candidates relax fine, every reference fails".
  P3  Errors are never swallowed: the first failure prints a FULL traceback;
      every failure stores a 600-char traceback tail in the csv `error` column.
  P4  Per-reference progress lines with atom counts and wall time, so a hang
      is visible and attributable.
  P5  Reference-level static-energy fallback: one stubborn phase degrades to a
      single-point CHGNet energy (flagged, counted) instead of killing the
      whole chemical system. NOTE: an unrelaxed reference sits above its
      relaxed energy, locally raising the hull -> candidate looks slightly
      MORE stable; systems with n_static_fallback > 0 deserve a re-check.
  P6  Candidate relaxation energies are cached too (the all-fail run threw
      away ~a day of candidate relaxations; never again), and the reference
      hull is built BEFORE the candidate relax so failures are cheap.

Self-consistency rationale (unchanged): CHGNet energies are not on MP's
per-element zero, so BOTH the hull references and the candidate use CHGNet
energies (GNoME / MatterGen style). Screening-grade E_hull; not a substitute
for corrected DFT hull energies (state in the paper; gaseous O2 reference is
approximate but interior oxide candidates never decompose to elemental O).

Output: mp_baseline/ehull_validation.csv + printed shortlist.
"""
import json
import os
import pickle
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "paches")
import utils
from pymatgen.core import Composition, Element

# ----------------------------- CONFIG -----------------------------
os.environ.setdefault("MP_API_KEY", "pQ9JHaAgk01LuS58n8Zgen7ty0QhDp9J")   # or hardcode as before
VALIDATION_CSV = "mp_baseline/mlip_validation.csv"
STRUCTS = "mp_baseline/na_structures.pkl"
OUT = "mp_baseline/ehull_validation.csv"
RELAX_CACHE_FILE = "mp_baseline/ehull_relax_cache.json"
DEVICE = "cpu"
FMAX = 0.08            # screening-grade; tighten to 0.05 for the final table
STEPS = 300
N_HULL = 15            # lowest-MLIP-strain survivors to check
NEAR_STABLE = 0.05     # eV/atom: Tier-2 window on the MP pre-hull
CAP_PHASES = 50        # Tier-2 cap (T0/T1 are exempt by design)
SITE_CAP = 60          # Tier-2 primitive-cell size limit
EHULL_STABLE = 0.05    # verdict: <0 stable(new), <= stable, <=0.10 metastable
TERM_E_WIN = 0.03      # eV/atom window for size-aware terminal polymorph pick
STABLE_TOL = 1e-6      # "on the MP pre-hull"
TM = ["Mn", "V", "Fe", "Ni", "Cr", "Cu", "Co", "Ti"]
ALLOWED = {"Na", "O", "P", "S", "F", "Si", "As", "B"}
# ------------------------------------------------------------------


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ================= candidate reconstruction (unchanged from v1) =====
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


def make_discharged(formula, structs, tmap):
    t = to_template(formula)
    comp = Composition(formula).get_el_amt_dict()
    tms = [(x, comp[x]) for x in TM if comp.get(x, 0) > 0]
    if t is None and len(tms) == 2:
        d = dict(Composition(formula).reduced_composition.get_el_amt_dict())
        others = {e: a for e, a in d.items() if e not in TM}
        t = (frozenset(others.items()), d.get(tms[0][0], 0) + d.get(tms[1][0], 0))
    if t not in tmap:
        return None
    src_bid, src_dom = tmap[t][0]
    S = structs[src_bid].copy()
    if len(tms) == 1:
        S.replace_species({Element(src_dom): Element(tms[0][0])})
    else:
        idx = [i for i, s in enumerate(S) if s.specie.symbol == src_dom]
        n0 = max(1, int(round(len(idx) * tms[0][1] / (tms[0][1] + tms[1][1]))))
        for j, i in enumerate(idx):
            S[i] = Element(tms[0][0] if j < n0 else tms[1][0])
    return S


# ================= structure hygiene [v3/P2] ========================
def clean_structure(struct):
    """Strip oxidation states and ALL site properties before ASE/CHGNet.

    MP thermo entries frequently carry `magmom` (occasionally vector-valued)
    or other decorations; these can crash AseAtomsAdaptor / the calculator.
    CHGNet infers magnetism itself, so nothing of value is lost.
    """
    s = struct.copy()
    try:
        s.remove_oxidation_states()
    except Exception:
        pass
    for p in list(s.site_properties):
        try:
            s.remove_site_property(p)
        except Exception:
            pass
    return s


def to_primitive(struct):
    try:
        return struct.get_primitive_structure()
    except Exception:
        return struct


# ================= MP entry fetch, single energy scale [v3.1] =======
def fetch_gga_entries(mpr, elements, logf=print):
    """Chemsys entries restricted to the GGA/GGA+U corrected scale.

    CHGNet is trained on MPtrj (GGA/GGA+U); r2SCAN entries sit on a different
    absolute energy scale and, if mixed into the selection pre-hull, push
    GGA/GGA+U facet phases out of the near-stable window (the v3 failure).
    Filter at the API when supported, and ALWAYS post-filter locally.
    """
    try:
        entries = mpr.get_entries_in_chemsys(
            list(elements),
            additional_criteria={"thermo_types": ["GGA_GGA+U"]})
    except TypeError:                      # older mp-api: no such kwarg
        entries = mpr.get_entries_in_chemsys(list(elements))
    n0 = len(entries)
    entries = [e for e in entries
               if getattr(e, "structure", None) is not None
               and "r2SCAN" not in str(getattr(e, "entry_id", ""))
               and "R2SCAN" not in str(getattr(e, "entry_id", ""))]
    logf(f"      entries: {len(entries)} on GGA/GGA+U scale "
         f"(dropped {n0 - len(entries)} r2SCAN/structureless)")
    return entries


# ================= tiered reference selection [v2] ==================
def _prim_len(e):
    try:
        return len(e.structure.get_primitive_structure())
    except Exception:
        return len(e.structure)


def select_reference_phases(entries, elems, logf=print):
    """Tiered selection; see v2 notes. T0 terminals and T1 stable COMPOUND
    facets are mandatory and cap-exempt (old caps could drop stable facets,
    making candidates look MORE stable -- a paper-level over-count); only T2
    near-stable extras are trimmed by CAP_PHASES/SITE_CAP."""
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    pd_mp = PhaseDiagram(entries)          # MP pre-hull: selection only
    ehs = {}
    for e in entries:
        try:
            ehs[id(e)] = pd_mp.get_e_above_hull(e)
        except Exception:
            continue

    picked, tier_of = [], {}

    for sym in sorted(elems):                                  # Tier 0
        polys = [e for e in entries
                 if e.composition.is_element
                 and e.composition.elements[0].symbol == sym]
        if not polys:
            raise RuntimeError(f"no elemental MP entry for {sym} in chemsys")
        e_min = min(p.energy_per_atom for p in polys)
        near = [p for p in polys if p.energy_per_atom - e_min <= TERM_E_WIN]
        pick = min(near, key=_prim_len)
        picked.append(pick); tier_of[id(pick)] = "T0"

    for e in entries:                                          # Tier 1
        if (id(e) in ehs and ehs[id(e)] <= STABLE_TOL
                and not e.composition.is_element and id(e) not in tier_of):
            picked.append(e); tier_of[id(e)] = "T1"

    extras = sorted((e for e in entries                        # Tier 2
                     if id(e) in ehs and STABLE_TOL < ehs[id(e)] <= NEAR_STABLE
                     and id(e) not in tier_of),
                    key=lambda e: ehs[id(e)])
    n2 = 0
    for e in extras:
        if n2 >= CAP_PHASES:
            break
        if _prim_len(e) > SITE_CAP:
            continue
        picked.append(e); tier_of[id(e)] = "T2"; n2 += 1

    n0 = sum(1 for v in tier_of.values() if v == "T0")
    n1 = sum(1 for v in tier_of.values() if v == "T1")
    logf(f"      refs {len(picked)}/{len(entries)} "
         f"(T0 terminals {n0}, T1 stable facets {n1}, T2 near-stable {n2})")
    return picked


# ============================== main ================================
def main():
    v = pd.read_csv(VALIDATION_CSV)
    if "mlip_delta_volume" not in v.columns:
        print("[ehull] no mlip_delta_volume in stage-1 csv (dry-run?)"); return
    surv = (v[v["status"] == "ok"].sort_values("mlip_delta_volume")
            .head(N_HULL).reset_index(drop=True))
    log(f"survivors to hull-check: {len(surv)}")

    structs = pickle.load(open(STRUCTS, "rb"))
    tmap = build_template_map(structs)

    # ---- CHGNet calculator + modern-ase relaxation (as in stage-1) ----
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
    dev = "cuda" if (DEVICE.startswith("cuda") and torch.cuda.is_available()) else "cpu"
    calc = CHGNetCalculator(use_device=dev)
    log(f"CHGNet on {dev}")

    def relax_energy(struct):
        s = clean_structure(struct)                    # [v3/P2]
        atoms = AseAtomsAdaptor.get_atoms(s)
        atoms.calc = calc
        FIRE(FILT(atoms), logfile=None).run(fmax=FMAX, steps=STEPS)
        return AseAtomsAdaptor.get_structure(atoms), float(atoms.get_potential_energy())

    def static_energy(struct):
        s = clean_structure(struct)
        atoms = AseAtomsAdaptor.get_atoms(s)
        atoms.calc = calc
        return float(atoms.get_potential_energy())

    from mp_api.client import MPRester
    from pymatgen.entries.computed_entries import ComputedEntry
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    key = os.environ.get("MP_API_KEY")

    # ---- persistent relax cache [v2/P6]: refs AND candidates ----------
    relax_cache = {}   # key -> {"comp": str, "E": float, "fallback": 0/1}
    if os.path.exists(RELAX_CACHE_FILE):
        relax_cache = json.load(open(RELAX_CACHE_FILE))
        log(f"resumed {len(relax_cache)} relaxed phases from cache")

    def _dump_cache():
        with open(RELAX_CACHE_FILE, "w") as fh:
            json.dump(relax_cache, fh)

    _first_tb = [True]

    def _report_failure(tag, exc):                     # [v3/P3]
        if _first_tb[0]:
            print(f"\n===== FIRST FAILURE ({tag}) -- full traceback =====",
                  flush=True)
            traceback.print_exc()
            print("=" * 55, flush=True)
            _first_tb[0] = False
        return traceback.format_exc()[-600:]

    def relaxed_entry(struct, cache_key, label):
        """Relax (cached) -> (ComputedEntry, fallback_flag). Static-energy
        fallback if relaxation fails; raises only if even static fails."""
        if cache_key in relax_cache:
            r = relax_cache[cache_key]
            return (ComputedEntry(Composition(r["comp"]), r["E"],
                                  entry_id=cache_key), r.get("fallback", 0))
        s0 = to_primitive(struct)
        t0 = time.time()
        fb = 0
        try:
            Sr, Etot = relax_energy(s0)
            comp = Sr.composition
        except Exception as exc:                       # [v3/P5]
            _report_failure(f"relax {label}", exc)
            log(f"      relax FAILED for {label}; static-energy fallback")
            Etot = static_energy(s0)
            comp = clean_structure(s0).composition
            fb = 1
        log(f"      {label} ({len(s0)} atoms) "
            f"{'static' if fb else 'relaxed'} in {time.time()-t0:.0f}s "
            f"[{Etot/len(s0):+.3f} eV/atom]")
        relax_cache[cache_key] = {"comp": str(comp), "E": float(Etot),
                                  "fallback": fb}
        _dump_cache()
        return ComputedEntry(comp, float(Etot), entry_id=cache_key), fb

    pd_cache = {}   # chemsys -> (PhaseDiagram, n_static_fallback)

    def get_cg_pd(elements):
        chemsys = "-".join(sorted(elements))
        if chemsys in pd_cache:
            return pd_cache[chemsys]
        with MPRester(key) as mpr:
            entries = fetch_gga_entries(mpr, elements,
                                        logf=lambda m: print(m, flush=True))
        near = select_reference_phases(entries, set(elements),
                                       logf=lambda m: print(m, flush=True))
        cg, n_fb = [], 0
        for i, e in enumerate(near, 1):
            eid = getattr(e, "entry_id", None) or \
                str(e.composition) + str(round(e.energy, 3))
            lab = f"({i}/{len(near)}) {eid} {e.composition.reduced_formula}"
            ce, fb = relaxed_entry(e.structure, str(eid), lab)
            cg.append(ce); n_fb += fb
        phased = PhaseDiagram(cg)
        pd_cache[chemsys] = (phased, n_fb)
        return phased, n_fb

    # ---- preflight self-test [v3/P1] ----------------------------------
    def preflight():
        log("preflight: relax local Na cell + cache round-trip + toy hull ...")
        from pymatgen.core import Structure, Lattice
        from pymatgen.analysis.phase_diagram import PDEntry
        na = Structure(Lattice.cubic(4.29), ["Na", "Na"],
                       [[0, 0, 0], [0.5, 0.5, 0.5]])
        ce, fb = relaxed_entry(na, "preflight::Na-bcc", "preflight Na (2 atoms)")
        assert not fb, "preflight relaxation fell back to static -- inspect traceback above"
        rt = json.load(open(RELAX_CACHE_FILE))
        assert "preflight::Na-bcc" in rt, "cache round-trip failed"
        toy = PhaseDiagram([PDEntry(Composition("Na"), 0.0),
                            PDEntry(Composition("O"), 0.0),
                            PDEntry(Composition("Na2O"), -6.0)])
        _, eh = toy.get_decomp_and_e_above_hull(
            PDEntry(Composition("NaO"), -4.0), allow_negative=True)
        assert eh < 0, "allow_negative path broken in this pymatgen version"
        log("preflight OK")

    try:
        preflight()
    except Exception:
        print("\npreflight FAILED -- environment problem, aborting before "
              "the long run:", flush=True)
        traceback.print_exc()
        sys.exit(1)

    # ---- candidate loop ------------------------------------------------
    rows = []
    for i, c in surv.iterrows():
        f = c["formula_discharge"]
        log(f"[{i+1}/{len(surv)}] {f}")
        rec = dict(formula_discharge=f, framework=c.get("framework"),
                   mlip_delta_volume=c.get("mlip_delta_volume"),
                   pred_voltage=c.get("pred_voltage"))
        S = make_discharged(f, structs, tmap)
        if S is None:
            rec["status"] = "no_structure"; rows.append(rec); continue
        try:
            elems = {str(e) for e in S.composition.elements}
            phased, n_fb = get_cg_pd(elems)            # refs first [v3/P6]
            cand_ce, cand_fb = relaxed_entry(S, f"cand::{f}", f"candidate {f}")
            if cand_fb:
                raise RuntimeError("candidate relaxation fell back to static; "
                                   "E_hull would not be trustworthy")
            dec, ehull = phased.get_decomp_and_e_above_hull(
                cand_ce, allow_negative=True)          # [v2] stable(new)
            ehull = float(ehull)
            dec_str = ", ".join(
                f"{p.composition.reduced_formula}:{x:.2f}"
                for p, x in sorted(dec.items(), key=lambda t: -t[1]))
            rec["decomposition"] = dec_str
            # [v3.1] under-coverage detector: an O-containing TM candidate
            # decomposing into elemental TM or O2 means facet phases are
            # missing from the reference set -- E_hull is NOT trustworthy
            cels = {el.symbol for el in cand_ce.composition.elements}
            if "O" in cels and cels & set(TM):
                bad = [p.composition.reduced_formula for p in dec
                       if p.composition.is_element
                       and str(p.composition.elements[0]) in set(TM) | {"O"}]
                if bad:
                    log(f"   !! UNDER-COVERAGE WARNING: decomposition uses "
                        f"elemental {bad} -- reference set likely missing "
                        f"facet phases; treat this E_hull as suspect")
                    rec["coverage_warning"] = ";".join(bad)
            rec.update(e_above_hull=ehull,
                       stability=("stable(new)" if ehull < 0
                                  else "stable" if ehull <= EHULL_STABLE
                                  else "metastable" if ehull <= 0.10
                                  else "unstable"),
                       n_static_fallback=n_fb, status="ok")
            log(f"   -> E_hull = {ehull:+.4f} eV/atom  [{rec['stability']}]"
                + (f"  ({n_fb} static-fallback refs)" if n_fb else ""))
        except Exception as exc:
            rec["status"] = f"fail:{type(exc).__name__}"
            rec["error"] = _report_failure(f"candidate {f}", exc)
            log(f"   -> FAILED: {type(exc).__name__}: {exc}")
        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\n[ehull] wrote {OUT}")
    ok = out[out["status"] == "ok"]
    if len(ok):
        ok = ok.sort_values("e_above_hull")
        cols = ["formula_discharge", "framework", "e_above_hull", "stability",
                "n_static_fallback", "coverage_warning",
                "mlip_delta_volume", "pred_voltage"]
        cols = [c for c in cols if c in ok.columns]
        print("\n=== self-consistent CHGNet hull (sorted by E_above_hull) ===")
        print(ok[cols].to_string(index=False))
        print(f"\n[ehull] stable(new): {(ok['e_above_hull'] < 0).sum()}/{len(ok)}; "
              f"stable (<= {EHULL_STABLE}): {(ok['e_above_hull'] <= EHULL_STABLE).sum()}/{len(ok)}; "
              f"metastable (<=0.10): {(ok['e_above_hull'] <= 0.10).sum()}/{len(ok)}")
    else:
        print("[ehull] no successful candidates -- see the FIRST FAILURE "
              "traceback above and the `error` column in the csv")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
