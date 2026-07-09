# Uncertainty-aware ML for low-strain sodium-ion cathodes

Code, data, and figures for the paper *"Uncertainty-aware machine learning for
the discovery and physical validation of low-strain sodium-ion cathodes"*.

## Overview

An uncertainty-aware framework for discovering low-strain sodium-ion cathodes,
integrating calibrated property prediction, generative candidate design, and
two-stage physical validation on a single sodium chemistry.

## Repository structure

```
data/         Cleaned dataset, generated candidates, and all result tables
figures/      Main-text and Supporting Information figures (PDF + PNG)
scripts/
  models/       Four benchmark models (RF+Magpie, MiniRoost, MiniCGCNN, ALIGNN)
  uncertainty/  Conformal NGBoost and epistemic-aleatoric decomposition
  generation/   Framework-substitution candidate generation and screening
  validation/   CHGNet strain relaxation and self-consistent convex-hull analysis
  analysis/     Per-chemistry subset analysis and leave-one-chemistry-out
```

## Data files

| File | Description |
|------|-------------|
| `data/training_set_365.csv` | 365 cleaned Na electrode pairs (Materials Project) |
| `data/generated_candidates_813.csv` | 813 candidates from framework substitution |
| `data/screening_shortlist.csv` | Full uncertainty-aware ranking with intervals |
| `data/mlip_strain_30.csv` | 30 CHGNet strain-relaxed candidates |
| `data/hull_validation_15.csv` | 15 candidates with convex-hull energies |
| `data/positive_control_6.csv` | Six known oxides used as the hull positive control |
| `data/loco_per_family.csv` | Leave-one-chemistry-out results per family |

## Pipeline

1. `scripts/models/` — benchmark the four models under 5-fold cross-validation
2. `scripts/analysis/` — per-transition-metal and leave-one-chemistry-out analysis
3. `scripts/uncertainty/` — conformal calibration and uncertainty decomposition
4. `scripts/generation/` — generate 813 candidates and rank by composite score
5. `scripts/validation/` — CHGNet strain screen and self-consistent convex hull

## Environment

Two conda environments were used:

- **Prediction / screening**: pymatgen, matminer, scikit-learn, PyTorch, NGBoost
- **Interatomic potential**: CHGNet, ASE, pymatgen

ALIGNN is installed from its pip package. See `requirements.txt` for core
dependencies. The positive control (`positive_control_6.csv`) can be regenerated
as an installation check: a correct setup reproduces a mean energy above hull of
about -0.0001 eV/atom for the six reference oxides.

## Reproducibility

All random operations use fixed seeds. MiniCGCNN uses seeds 42-47 and ALIGNN uses
three independent seeds. The convex-hull construction is deterministic given the
same reference phases and CHGNet version.

## Citation

Please cite the associated paper if you use this code or data.
