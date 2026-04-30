# CompPhys Final Project Code

## Repository layout

- `restart_studies/`: main codebase for the final-project restart pipeline (training, evaluation, aggregation, sanity checks, and figures).
- `restart_studies/sbatch/`: all `.sh` launcher scripts (single jobs, arrays, submit helpers, and aggregators) moved here to keep the root code folder uncluttered.
- `jetclass_transformer/`: supporting JetClass transformer code and earlier baseline utilities.

## Main files in `restart_studies/`

- `reimplement_preliminary_studies.py`
  - Re-runs the baseline/preliminary study pipeline (clean runs, corruptions, shift metrics, interpretability effectiveness, sanity outputs).
- `aggregate_preliminary_studies_multi_seed.py`
  - Aggregates multi-seed outputs from the preliminary reimplementation.
- `evaluate_aspen_shift_calibration.py`
  - Calibrates unlabeled shift metrics on JetClass corruptions and applies them to AspenOpenJets inference outputs.
- `aspen_shift_sanity_checks_5seeds.py`
  - Cross-seed integrity checks for Aspen shift outputs and calibration behavior.
- `probe_aspen_openjets_h5.py`
  - HDF5 schema/data probe for AspenOpenJets files and loader metadata generation.
- `train_rrr_find_another_single.py`
  - One full RRR/find-another configuration run (iterative masks, retraining, JetClass/Aspen evaluation each iteration).
- `aggregate_rrr_find_another_sweep.py`
  - Aggregates completed RRR sweep runs and produces ranking/summary tables.
- `make_rrr_sweep_figures.py`
  - Generates figure assets for sweep tradeoff/heatmap/trajectory plots.
- `train_eval_jetclass_canonical_aspen.py`
  - Canonical feature/label loading path for JetClass-to-Aspen evaluation.
- `backends/`
  - Backend loading/model/data utilities used by canonical and restart scripts.
- `config/`
  - Configuration assets used by restart scripts.

## What the `.sh` / `sbatch` scripts are for

All shell launchers are in `restart_studies/sbatch/`:

- `sbatch_*.sh`
  - SLURM job scripts for single jobs or arrays (training, Aspen eval, probing, aggregation, sanity checks).
- `submit_*.sh`
  - Convenience submit wrappers that queue one or more `sbatch_*.sh` jobs with dependencies.
- `run_*.sh`
  - Local/interactive convenience run scripts (non-SLURM).

## Authorship / provenance note

All code I wrote for this final project is in:

- `/home/ryan/Documents/School/CompPhys/new_final_project/new_github/CompPhys_Final/restart_studies`

Not all code in `restart_studies/` is fully original to this project. Some parts were inherited or adapted from prior projects and research-group work.

In particular, both of the files below contain substantial code I wrote for this project plus inherited/adapted components from previous studies:

- `restart_studies/train_eval_jetclass_canonical_aspen.py`
- `restart_studies/reimplement_preliminary_studies.py`
