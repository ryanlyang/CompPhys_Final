#!/usr/bin/env bash
#SBATCH --job-name=jcpart0_evalinterp
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_jcpart0_evalinterp_%j.out
#SBATCH --error=slurm_jcpart0_evalinterp_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATASET_DIR="${DATASET_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ryreu/atlas/CompPhys_Final/runs/jetclass_part0_evalinterp}"

CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "ERROR: CHECKPOINT is required. Set CHECKPOINT=/path/to/saved_model.pt" >&2
  exit 1
fi

TRAINER_LOG="${TRAINER_LOG:-}"
TRAINER_SUMMARY="${TRAINER_SUMMARY:-/dev/null}"

FEATURE_SET="${FEATURE_SET:-kinpid}"
LABEL_SOURCE="${LABEL_SOURCE:-filename}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-part0_${FEATURE_SET}_seed${SEED}_evalinterp_from_trained}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_TAG}_job${SLURM_JOB_ID:-manual}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/slurm_logs/${RUN_TAG}}"

DEVICE="${DEVICE:-cuda}"
TRAIN_INDICES="${TRAIN_INDICES:-0-7}"
VAL_INDICES="${VAL_INDICES:-8}"
TEST_INDICES="${TEST_INDICES:-9}"

# Analysis (corruption + correlation) config
EPOCHS="${EPOCHS:-12}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-150000}"
VAL_SAMPLES="${VAL_SAMPLES:-50000}"
TEST_JETS="${TEST_JETS:-150000}"
NUM_WORKERS="${NUM_WORKERS:-6}"
FETCH_STEP="${FETCH_STEP:-0.01}"
START_LR="${START_LR:-1e-3}"
GPUS="${GPUS:-0}"
NOISE_LEVELS="${NOISE_LEVELS:-0.01,0.03,0.05,0.1}"
DROPOUT_LEVELS="${DROPOUT_LEVELS:-0.05,0.1,0.2,0.3}"
MASK_LEVELS="${MASK_LEVELS:-0.05,0.1,0.2,0.3}"
JITTER_LEVELS="${JITTER_LEVELS:-0.01,0.03,0.05,0.1}"

# Interpretability config
METHODS="${METHODS:-input_gradients,integrated_gradients,smoothgrad}"
TARGET_MODE="${TARGET_MODE:-true}"           # true | pred
EXPLAIN_SAMPLING="${EXPLAIN_SAMPLING:-stratified}"
MASK_FRACTIONS="${MASK_FRACTIONS:-0.02,0.05,0.1,0.2}"
RANDOM_REPEATS="${RANDOM_REPEATS:-3}"
MAX_EVAL_JETS="${MAX_EVAL_JETS:-120000}"
MAX_EXPLAIN_JETS="${MAX_EXPLAIN_JETS:-20000}"
ATTR_BATCH_SIZE="${ATTR_BATCH_SIZE:-96}"
IG_STEPS="${IG_STEPS:-16}"
SMOOTHGRAD_SAMPLES="${SMOOTHGRAD_SAMPLES:-16}"
SMOOTHGRAD_SIGMA="${SMOOTHGRAD_SIGMA:-0.02}"
MAX_NUM_PARTICLES="${MAX_NUM_PARTICLES:-128}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

RUNTIME_LOG_STDOUT="${LOG_DIR}/job${SLURM_JOB_ID:-manual}.out"
RUNTIME_LOG_STDERR="${LOG_DIR}/job${SLURM_JOB_ID:-manual}.err"
exec > >(tee -a "${RUNTIME_LOG_STDOUT}") 2> >(tee -a "${RUNTIME_LOG_STDERR}" >&2)

echo "RUNTIME_LOG_STDOUT=${RUNTIME_LOG_STDOUT}"
echo "RUNTIME_LOG_STDERR=${RUNTIME_LOG_STDERR}"
echo "RUNTIME_RUN_DIR=${RUN_DIR}"
echo "============================================================"
echo "JetClass part0 eval+interp from pretrained checkpoint"
echo "Run dir: ${RUN_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Dataset: ${DATASET_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Trainer log: ${TRAINER_LOG:-[auto]}"
echo "Trainer summary: ${TRAINER_SUMMARY}"
echo "Feature set: ${FEATURE_SET}"
echo "Label source: ${LABEL_SOURCE}"
echo "Seed: ${SEED}"
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

# Guardrail: this script is intended for baseline checkpoints, not RRR checkpoints.
if [[ "${CHECKPOINT}" == *"/jetclass_part0_rrr/"* ]]; then
  echo "ERROR: checkpoint appears to come from jetclass_part0_rrr. " \
       "Use baseline checkpoint from jetclass_part0_shiftstudy." >&2
  exit 1
fi

if [[ -z "${TRAINER_LOG}" ]]; then
  CANDIDATE_LOG="$(dirname "${CHECKPOINT}")/train.log"
  if [[ -f "${CANDIDATE_LOG}" ]]; then
    TRAINER_LOG="${CANDIDATE_LOG}"
  fi
fi

ANALYSIS_DIR="${RUN_DIR}/analysis"
INTERP_DIR="${RUN_DIR}/interpretability"
mkdir -p "${ANALYSIS_DIR}" "${INTERP_DIR}"

ANALYSIS_CMD=(
  python3 run_jetclass_part0_baseline_and_shift.py
  --dataset-dir "${DATASET_DIR}"
  --output-dir "${ANALYSIS_DIR}"
  --feature-set "${FEATURE_SET}"
  --label-source "${LABEL_SOURCE}"
  --train-indices "${TRAIN_INDICES}"
  --val-indices "${VAL_INDICES}"
  --test-indices "${TEST_INDICES}"
  --epochs "${EPOCHS}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --samples-per-epoch "${TRAIN_SAMPLES}"
  --samples-per-epoch-val "${VAL_SAMPLES}"
  --max-test-jets "${TEST_JETS}"
  --num-workers "${NUM_WORKERS}"
  --fetch-step "${FETCH_STEP}"
  --start-lr "${START_LR}"
  --gpus "${GPUS}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --tensorboard-name "JetClass_part0_${FEATURE_SET}_seed${SEED}_evalinterp"
  --noise-levels "${NOISE_LEVELS}"
  --dropout-levels "${DROPOUT_LEVELS}"
  --mask-levels "${MASK_LEVELS}"
  --jitter-levels "${JITTER_LEVELS}"
  --skip-train
  --checkpoint "${CHECKPOINT}"
  --trainer-summary "${TRAINER_SUMMARY}"
)
if [[ -n "${TRAINER_LOG}" ]]; then
  ANALYSIS_CMD+=(--trainer-log "${TRAINER_LOG}")
fi

echo "--- Analysis stage ---"
printf ' %q' "${ANALYSIS_CMD[@]}"
echo
"${ANALYSIS_CMD[@]}"

INTERP_CMD=(
  python3 run_jetclass_interpretability_benchmark.py
  --dataset-dir "${DATASET_DIR}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${INTERP_DIR}"
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
  --trainer-summary "${TRAINER_SUMMARY}"
)
if [[ -n "${TRAINER_LOG}" ]]; then
  INTERP_CMD+=(--trainer-log "${TRAINER_LOG}")
fi

echo
echo "--- Interpretability stage ---"
printf ' %q' "${INTERP_CMD[@]}"
echo
"${INTERP_CMD[@]}"

echo
echo "Done. Outputs:"
echo "  ${ANALYSIS_DIR}/summary.json"
echo "  ${ANALYSIS_DIR}/correlations.csv"
echo "  ${INTERP_DIR}/interpretability_summary.json"
echo "  ${INTERP_DIR}/method_effectiveness_summary.csv"
echo "  ${RUNTIME_LOG_STDOUT}"
echo "  ${RUNTIME_LOG_STDERR}"
