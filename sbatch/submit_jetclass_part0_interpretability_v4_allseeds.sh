#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="${ROOT_DIR}/sbatch/run_jetclass_interpretability_tier3.sh"

if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: runner not found: ${RUNNER}" >&2
  exit 1
fi

# Default paths for your current cluster layout.
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ryreu/atlas/CompPhys_Final/runs/jetclass_part0_interpretability}"
CKPT_ROOT="${CKPT_ROOT:-/home/ryreu/atlas/PracticeTagging/runs/jetclass_part0_shiftstudy}"
FEATURE_SET="${FEATURE_SET:-kinpid}"
RUN_SUFFIX="${RUN_SUFFIX:-interp_v4}"
PARTITION="${PARTITION:-debug}"

# Keep settings aligned with fixed interpretability benchmark setup.
TARGET_MODE="${TARGET_MODE:-true}"
EXPLAIN_SAMPLING="${EXPLAIN_SAMPLING:-stratified}"
MAX_EVAL_JETS="${MAX_EVAL_JETS:-120000}"
MAX_EXPLAIN_JETS="${MAX_EXPLAIN_JETS:-20000}"
METHODS="${METHODS:-input_gradients,integrated_gradients,smoothgrad}"
MASK_FRACTIONS="${MASK_FRACTIONS:-0.02,0.05,0.1,0.2}"
RANDOM_REPEATS="${RANDOM_REPEATS:-3}"
IG_STEPS="${IG_STEPS:-16}"
SMOOTHGRAD_SAMPLES="${SMOOTHGRAD_SAMPLES:-16}"
SMOOTHGRAD_SIGMA="${SMOOTHGRAD_SIGMA:-0.02}"

declare -A CKPT_BY_SEED=(
  [0]="${CKPT_ROOT}/part0_kinpid_seed0_job21156942/saved_model.pt"
  [2]="${CKPT_ROOT}/part0_kinpid_seed2_job21157144/saved_model.pt"
  [3]="${CKPT_ROOT}/part0_kinpid_seed3_job21157145/saved_model.pt"
)

SEEDS=(0 2 3)

echo "Submitting interpretability v4 jobs for seeds: ${SEEDS[*]}"
echo "Runner: ${RUNNER}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Checkpoint root: ${CKPT_ROOT}"
echo

for SEED in "${SEEDS[@]}"; do
  CHECKPOINT="${CKPT_BY_SEED[${SEED}]}"
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "WARN: missing checkpoint for seed ${SEED}, skipping: ${CHECKPOINT}" >&2
    continue
  fi

  RUN_TAG="part0_${FEATURE_SET}_seed${SEED}_${RUN_SUFFIX}"

  echo "Submitting seed ${SEED}"
  echo "  checkpoint: ${CHECKPOINT}"
  echo "  run_tag:    ${RUN_TAG}"
  sbatch \
    --partition="${PARTITION}" \
    --export=ALL,OUTPUT_ROOT="${OUTPUT_ROOT}",FEATURE_SET="${FEATURE_SET}",SEED="${SEED}",RUN_SUFFIX="${RUN_SUFFIX}",RUN_TAG="${RUN_TAG}",CHECKPOINT="${CHECKPOINT}",TARGET_MODE="${TARGET_MODE}",EXPLAIN_SAMPLING="${EXPLAIN_SAMPLING}",MAX_EVAL_JETS="${MAX_EVAL_JETS}",MAX_EXPLAIN_JETS="${MAX_EXPLAIN_JETS}",METHODS="${METHODS}",MASK_FRACTIONS="${MASK_FRACTIONS}",RANDOM_REPEATS="${RANDOM_REPEATS}",IG_STEPS="${IG_STEPS}",SMOOTHGRAD_SAMPLES="${SMOOTHGRAD_SAMPLES}",SMOOTHGRAD_SIGMA="${SMOOTHGRAD_SIGMA}" \
    "${RUNNER}"
done

