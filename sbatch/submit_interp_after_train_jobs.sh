#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 TRAIN_JOB_ID0 TRAIN_JOB_ID1 TRAIN_JOB_ID2 [more TRAIN_JOB_ID...]" >&2
  echo "Env vars you can override:" >&2
  echo "  TRAIN_OUTPUT_ROOT (default: /home/ryreu/atlas/CompPhys_Final/runs/jetclass_part0_shiftstudy)" >&2
  echo "  TRAIN_RUN_TAG     (default: part0_kinpid_seed2_train150k50k_test150k)" >&2
  echo "  INTERP_RUN_TAG_PREFIX (default: part0_kinpid_interp_after_train)" >&2
  echo "  SEED_BASE         (default: 0; seed increments by index)" >&2
  exit 1
fi

TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-/home/ryreu/atlas/CompPhys_Final/runs/jetclass_part0_shiftstudy}"
TRAIN_RUN_TAG="${TRAIN_RUN_TAG:-part0_kinpid_seed2_train150k50k_test150k}"
INTERP_RUN_TAG_PREFIX="${INTERP_RUN_TAG_PREFIX:-part0_kinpid_interp_after_train}"
SEED_BASE="${SEED_BASE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERP_SBATCH="${SCRIPT_DIR}/run_jetclass_part0_interpretability_tier3.sh"

if [[ ! -f "${INTERP_SBATCH}" ]]; then
  echo "ERROR: missing interpretability sbatch script: ${INTERP_SBATCH}" >&2
  exit 1
fi

for i in "$@"; do
  if ! [[ "${i}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: job id must be numeric, got '${i}'" >&2
    exit 1
  fi
done

idx=0
for train_job_id in "$@"; do
  seed=$((SEED_BASE + idx))
  checkpoint="${TRAIN_OUTPUT_ROOT}/${TRAIN_RUN_TAG}_job${train_job_id}/saved_model.pt"
  run_tag="${INTERP_RUN_TAG_PREFIX}_seed${seed}_from${train_job_id}"

  echo "Submitting interpretability job dependent on train job ${train_job_id}"
  echo "  checkpoint: ${checkpoint}"
  echo "  run_tag:    ${run_tag}"

  sbatch \
    --dependency="afterok:${train_job_id}" \
    --export="ALL,CHECKPOINT=${checkpoint},SEED=${seed},RUN_TAG=${run_tag}" \
    "${INTERP_SBATCH}"

  idx=$((idx + 1))
done
