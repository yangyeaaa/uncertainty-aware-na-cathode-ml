"""
Substitution-based candidate generator for Na cathodes (Windows-safe).
Run:  python generate_candidates.py

Why: the MP Na insertion-electrode database, after the stability filter, is
essentially the 365 training set (only ~45 unique held-out formulas, all of
them previously filtered out). There is no meaningful in-database pool to
screen, so genuinely novel candidates must be generated.

How (grounded, not random):
  1. From each clean single-TM training compound, extract its structural
     FRAMEWORK by removing the redox TM, keeping only frameworks built from
     {Na,O,P,S,F,Si,As,B} (drops exotic C/H/Mo/Ge/K entries).
  2. Re-instantiate every framework with each of the 8 redox TMs, plus a few
     priority binary mixes (Fe-Mn, Fe-Ni, Mn-Ni, Ni-Co, Fe-Co) on slots >= 2.
  3. Drop any composition already known to MP (training + raw 416) -> novelty.
  4. Write mp_baseline/generated_candidates.csv (formula_discharge + provenance).

These carry NO labels (voltage/capacity/ΔV); they are scored by mp_screen.py
and the top-k get their ground truth from MLIP relaxation.
"""
import sys, warnings
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "paches")
import utils
from pymatgen.core import Composition

TM = ["Mn", "V", "Fe", "Ni", "Cr", "Cu", "Co", "Ti"]
ALLOWED_FRAME = {"Na", "O", "P", "S", "F", "Si", "As", "B"}
PRIORITY_PAIRS = [("Fe", "Mn"), ("Fe", "Ni"), ("Mn", "Ni"), ("Ni", "Co"), ("Fe", "Co")]
OUT = "mp_baseline/generated_candidates.csv"


def to_template(formula):
    """Return (frozenset(frame.items()), slot_amount) for clean single-TM frameworks, else None."""
    try:
        d = dict(Composition(formula).reduced_composition.get_el_amt_dict())
    except Exception:
        return None
    tms = [t for t in TM if d.get(t, 0) > 0]
    others = {e: a for e, a in d.items() if e not in TM}
    if len(tms) != 1:
        return None
    if any(e not in ALLOWED_FRAME for e in others):
        return None
    if not others:                      # need a real framework, not a binary oxide alone
        return None
    return (frozenset(others.items()), d[tms[0]])


def instantiate(template, comp_dict_override=None):
    frame, slot = template
    d = dict(frame)
    if comp_dict_override is None:
        return None
    d.update(comp_dict_override)
    try:
        return Composition(d).reduced_formula
    except Exception:
        return None


def main():
    # known MP formulas (raw 416 + cleaned training) -> novelty filter
    raw = pd.read_csv("mp_baseline/na_electrodes.csv")
    def red(f):
        try: return Composition(f).reduced_formula
        except Exception: return None
    known = set(raw["formula_discharge"].apply(red).dropna())

    tr = utils.load_and_filter("mp_baseline/na_electrodes.csv")
    known |= set(tr["formula_discharge"].apply(red).dropna())

    # extract unique frameworks from clean training compounds
    templates = {}
    for f in tr["formula_discharge"]:
        t = to_template(f)
        if t is not None:
            templates[t] = templates.get(t, 0) + 1
    print(f"[gen] clean single-TM frameworks extracted: {len(templates)}")

    rows = []
    for t, count in templates.items():
        slot = t[1]
        frame_str = "".join(f"{e}{int(a) if float(a).is_integer() else a}" for e, a in sorted(t[0]))
        # single-TM substitutions
        for tm in TM:
            rf = instantiate(t, {tm: slot})
            if rf and rf not in known:
                rows.append(dict(formula_discharge=rf, framework=frame_str,
                                 substitution=f"{tm}->slot{slot}", n_template_support=count))
        # binary mixes on integer slots >= 2
        if float(slot).is_integer() and slot >= 2:
            for a, b in PRIORITY_PAIRS:
                k = slot / 2 if (slot % 2 == 0) else (slot - 1)
                d = {a: k, b: slot - k} if (slot % 2 == 0) else {a: slot - 1, b: 1}
                rf = instantiate(t, d)
                if rf and rf not in known:
                    rows.append(dict(formula_discharge=rf, framework=frame_str,
                                     substitution=f"{a}/{b} mix slot{slot}", n_template_support=count))

    cand = pd.DataFrame(rows).drop_duplicates("formula_discharge").reset_index(drop=True)
    cand.to_csv(OUT, index=False)
    print(f"[gen] generated novel candidates: {len(cand)}  -> {OUT}")
    print("\nsample:")
    print(cand.head(15).to_string(index=False))
    # quick chemistry breakdown
    def dom(f):
        d = Composition(f).get_el_amt_dict()
        t = [(x, d.get(x, 0)) for x in TM if d.get(x, 0) > 0]
        return max(t, key=lambda z: z[1])[0] if t else "mix"
    cand["dom"] = cand["formula_discharge"].apply(dom)
    print("\nby dominant TM:", cand["dom"].value_counts().to_dict())


if __name__ == "__main__":
    main()
