#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ryan/Documents/School/CompPhys/new_final_project"
SCRIPT="${ROOT}/restart_studies/reimplement_preliminary_studies.py"

source ~/.bashrc

conda run -n base python3 "${SCRIPT}" \
  --data_dir "/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/data/jetclass_part0" \
  --output_root "${ROOT}/restart_studies/results" \
  --run_name "prelim_reimpl_seed52" \
  --seed 52 \
  --device cpu \
  --feature_mode full \
  --n_train_jets 12000 \
  --n_val_jets 3000 \
  --n_test_jets 12000 \
  --epochs 8 \
  --patience 3 \
  --batch_size 256 \
  --embed_dim 96 \
  --num_layers 4 \
  --ff_dim 384 \
  --num_workers 2 \
  --explain_subset_size 2000 \
  --explain_batch_size 128 \
  --ig_steps 12 \
  --smoothgrad_samples 8 \
  --smoothgrad_sigma 0.10 \
  --random_mask_repeats 3
