#!/usr/bin/env bash
#SBATCH --job-name=jcpart0_interp
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_jcpart0_interp_%j.out
#SBATCH --error=slurm_jcpart0_interp_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATASET_DIR="${DATASET_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ryreu/atlas/CompPhys_Final/runs/jetclass_part0_interpretability}"

# Required: checkpoint from completed training run.
CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "ERROR: CHECKPOINT is required. Set CHECKPOINT=/path/to/saved_model.pt" >&2
  exit 1
fi
TRAINER_LOG="${TRAINER_LOG:-}"

FEATURE_SET="${FEATURE_SET:-kinpid}"
LABEL_SOURCE="${LABEL_SOURCE:-filename}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-part0_${FEATURE_SET}_seed${SEED}_interp}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_TAG}_job${SLURM_JOB_ID:-manual}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/slurm_logs/${RUN_TAG}}"

# Interpretability config
DEVICE="${DEVICE:-cuda}"
METHODS="${METHODS:-input_gradients,integrated_gradients,smoothgrad}"
TARGET_MODE="${TARGET_MODE:-true}"           # true | pred
EXPLAIN_SAMPLING="${EXPLAIN_SAMPLING:-stratified}"
MASK_FRACTIONS="${MASK_FRACTIONS:-0.02,0.05,0.1,0.2}"
RANDOM_REPEATS="${RANDOM_REPEATS:-3}"
MAX_EVAL_JETS="${MAX_EVAL_JETS:-120000}"
MAX_EXPLAIN_JETS="${MAX_EXPLAIN_JETS:-20000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
ATTR_BATCH_SIZE="${ATTR_BATCH_SIZE:-96}"
IG_STEPS="${IG_STEPS:-16}"
SMOOTHGRAD_SAMPLES="${SMOOTHGRAD_SAMPLES:-16}"
SMOOTHGRAD_SIGMA="${SMOOTHGRAD_SIGMA:-0.02}"
MAX_NUM_PARTICLES="${MAX_NUM_PARTICLES:-128}"

# Split by file index within each class
TRAIN_INDICES="${TRAIN_INDICES:-0-7}"
VAL_INDICES="${VAL_INDICES:-8}"
TEST_INDICES="${TEST_INDICES:-9}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

RUNTIME_LOG_STDOUT="${LOG_DIR}/job${SLURM_JOB_ID:-manual}.out"
RUNTIME_LOG_STDERR="${LOG_DIR}/job${SLURM_JOB_ID:-manual}.err"
exec > >(tee -a "${RUNTIME_LOG_STDOUT}") 2> >(tee -a "${RUNTIME_LOG_STDERR}" >&2)

echo "RUNTIME_LOG_STDOUT=${RUNTIME_LOG_STDOUT}"
echo "RUNTIME_LOG_STDERR=${RUNTIME_LOG_STDERR}"
echo "RUNTIME_RUN_DIR=${RUN_DIR}"
echo "============================================================"
echo "JetClass interpretability benchmark"
echo "Run dir: ${RUN_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Dataset: ${DATASET_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Trainer log: ${TRAINER_LOG:-[auto]}"
echo "Feature set: ${FEATURE_SET}"
echo "Label source: ${LABEL_SOURCE}"
echo "Methods: ${METHODS}"
echo "Target mode: ${TARGET_MODE}"
echo "Explain sampling: ${EXPLAIN_SAMPLING}"
echo "Mask fractions: ${MASK_FRACTIONS}"
echo "Eval jets: ${MAX_EVAL_JETS}"
echo "Explain jets: ${MAX_EXPLAIN_JETS}"
echo "Conda env: atlas_kd"
echo "============================================================"
echo

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
export PATH="$HOME/.local/bin:$PATH"

cd "${REPO_ROOT}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ -z "${TRAINER_LOG}" ]]; then
  CANDIDATE_LOG="$(dirname "${CHECKPOINT}")/train.log"
  if [[ -f "${CANDIDATE_LOG}" ]]; then
    TRAINER_LOG="${CANDIDATE_LOG}"
  fi
fi

CMD=(
  python3 run_jetclass_interpretability_benchmark.py
  --dataset-dir "${DATASET_DIR}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${RUN_DIR}"
  --feature-set "${FEATURE_SET}"
  --label-source "${LABEL_SOURCE}"
  --device "${DEVICE}"
  --train-indices "${TRAIN_INDICES}"
  --val-indices "${VAL_INDICES}"
  --test-indices "${TEST_INDICES}"
  --max-num-particles "${MAX_NUM_PARTICLES}"
  --max-eval-jets "${MAX_EVAL_JETS}"
  --max-explain-jets "${MAX_EXPLAIN_JETS}"
  --explain-sampling "${EXPLAIN_SAMPLING}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --attr-batch-size "${ATTR_BATCH_SIZE}"
  --methods "${METHODS}"
  --target-mode "${TARGET_MODE}"
  --mask-fractions "${MASK_FRACTIONS}"
  --random-repeats "${RANDOM_REPEATS}"
  --ig-steps "${IG_STEPS}"
  --smoothgrad-samples "${SMOOTHGRAD_SAMPLES}"
  --smoothgrad-sigma "${SMOOTHGRAD_SIGMA}"
  --seed "${SEED}"
)

if [[ -n "${TRAINER_LOG}" ]]; then
  CMD+=(--trainer-log "${TRAINER_LOG}")
fi

printf ' %q' "${CMD[@]}"
echo
echo
"${CMD[@]}"

echo
echo "Done. Key outputs:"
echo "  ${RUN_DIR}/interpretability_summary.json"
echo "  ${RUN_DIR}/method_effectiveness_summary.csv"
echo "  ${RUN_DIR}/masking_perturbation_results.csv"
echo "  ${RUN_DIR}/feature_attribution_summary.csv"
echo "  ${RUNTIME_LOG_STDOUT}"
echo "  ${RUNTIME_LOG_STDERR}"
