"""
Tiered reference-phase selection for the self-consistent CHGNet convex hull.

Fixes the multi-element (e.g. Na-Fe-Co-P-O) failure mode of ehull_validate.py,
where the cost-control filters (near-stable cutoff + CAP_PHASES + MAX_ATOMS)
could silently drop elemental terminals or stable sub-system phases, making
PhaseDiagram construction fail ("missing terminal entries") or -- worse --
succeed with an under-covered hull that misreports E_hull.

Selection tiers:
  Tier 0 (mandatory, exempt from ALL caps):
      one elemental terminal per element. Chosen as the smallest-cell polymorph
      among those within TERMINAL_E_WINDOW of the elemental minimum, so that a
      huge ground-state cell (e.g. some P allotropes) never forces us to relax
      hundreds of atoms, while the energy penalty stays negligible.
  Tier 1 (mandatory, exempt from CAP_PHASES; size handled via primitive cell):
      every phase on the MP pre-hull (mp_e_hull <= STABLE_TOL). Because
      get_entries_in_chemsys returns all sub-chemsys entries, this tier
      automatically covers stable binaries/ternaries -- the actual hull facets
      that decompose an interior candidate.
  Tier 2 (optional, capped):
      near-stable phases with STABLE_TOL < mp_e_hull <= NEAR_STABLE, sorted by
      mp_e_hull, truncated to CAP_PHASES, size-limited by MAX_ATOMS.

The MP pre-hull is built locally from *uncorrected* MP energies (consistent
with CHGNet's MPtrj training reference), so no extra API calls are needed.
"""

from __future__ import annotations

from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.entries.computed_entries import ComputedEntry

STABLE_TOL = 1e-6        # eV/atom: "on the MP pre-hull"
TERMINAL_E_WINDOW = 0.03  # eV/atom: acceptable penalty when swapping a huge
                          # elemental ground-state cell for a smaller polymorph


def _uncorrected_epa(entry: ComputedEntry) -> float:
    """Uncorrected energy per atom (CHGNet-consistent reference)."""
    return entry.uncorrected_energy / entry.composition.num_atoms


def _natoms(entry) -> int:
    """Atom count of the (primitive, if available) structure."""
    s = getattr(entry, "structure", None)
    if s is None:
        return entry.composition.num_atoms
    try:
        return len(s.get_primitive_structure())
    except Exception:
        return len(s)


def mp_pre_hull(entries) -> dict:
    """Local pre-hull from uncorrected MP energies.

    Returns {entry_id: mp_e_hull}. Uses PDEntry copies so that MP's applied
    corrections never leak into the geometry of the pre-hull.
    """
    pde = [PDEntry(e.composition, e.uncorrected_energy,
                   attribute=e.entry_id) for e in entries]
    pd = PhaseDiagram(pde)
    out = {}
    for p in pde:
        out[p.attribute] = pd.get_e_above_hull(p)
    return out


def select_reference_entries(entries,
                             near_stable: float = 0.05,
                             cap_phases: int = 60,
                             max_atoms: int = 80,
                             log=print):
    """Tiered selection. Returns (selected_entries, info_dict).

    `entries`: ComputedStructureEntry list for one chemical system
               (output of MPRester.get_entries_in_chemsys).
    """
    by_id = {e.entry_id: e for e in entries}
    ehull = mp_pre_hull(entries)

    selected: dict[str, str] = {}   # entry_id -> tier label

    # ---- Tier 0: one terminal per element ------------------------------
    elements = sorted({el.symbol for e in entries
                       for el in e.composition.elements})
    for sym in elements:
        polys = [e for e in entries
                 if e.composition.is_element
                 and e.composition.elements[0].symbol == sym]
        if not polys:
            raise RuntimeError(
                f"chemsys has no elemental MP entry for {sym}; "
                f"cannot anchor the hull")
        e_min = min(_uncorrected_epa(p) for p in polys)
        near_ground = [p for p in polys
                       if _uncorrected_epa(p) - e_min <= TERMINAL_E_WINDOW]
        pick = min(near_ground, key=_natoms)     # smallest cell wins
        selected[pick.entry_id] = "T0-terminal"

    # ---- Tier 1: MP-stable COMPOUND phases (hull facets) ----------------
    # Elemental entries are excluded here: terminals are handled exclusively
    # by Tier 0's size-aware pick, so a huge elemental ground-state cell can
    # never re-enter through the stable-phase door. The <=TERMINAL_E_WINDOW
    # penalty on a swapped terminal only lifts the hull at the elemental
    # corner, which interior (oxide) candidates never decompose into.
    for eid, eh in ehull.items():
        if (eh <= STABLE_TOL and eid not in selected
                and not by_id[eid].composition.is_element):
            selected[eid] = "T1-stable"

    # ---- Tier 2: near-stable extras (capped, size-limited) -------------
    extras = sorted(
        (eid for eid, eh in ehull.items()
         if STABLE_TOL < eh <= near_stable and eid not in selected),
        key=lambda eid: ehull[eid])
    n_extra = 0
    for eid in extras:
        if n_extra >= cap_phases:
            break
        if _natoms(by_id[eid]) > max_atoms:
            continue
        selected[eid] = "T2-near-stable"
        n_extra += 1

    picked = [by_id[eid] for eid in selected]
    info = {
        "n_total": len(entries),
        "n_selected": len(picked),
        "n_t0": sum(1 for v in selected.values() if v == "T0-terminal"),
        "n_t1": sum(1 for v in selected.values() if v == "T1-stable"),
        "n_t2": n_extra,
        "tiers": selected,
        "mp_e_hull": ehull,
    }
    log(f"  [select] {info['n_selected']}/{info['n_total']} refs "
        f"(T0 terminals {info['n_t0']}, T1 stable {info['n_t1']}, "
        f"T2 near-stable {info['n_t2']})")

    # ---- Sanity: the selected set must span all elements ---------------
    covered = {el.symbol for e in picked for el in e.composition.elements
               if e.composition.is_element}
    missing = set(elements) - covered
    if missing:
        raise RuntimeError(f"internal error: terminals missing for {missing}")
    return picked, info
