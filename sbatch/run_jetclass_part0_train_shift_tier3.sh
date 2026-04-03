#!/usr/bin/env bash
#SBATCH --job-name=jcpart0_shift
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_jcpart0_shift_%j.out
#SBATCH --error=slurm_jcpart0_shift_%j.err

set -euo pipefail

# Core paths
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATASET_DIR="${DATASET_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ryreu/atlas/CompPhys_Final/runs/jetclass_part0_shiftstudy}"

# Run identity
FEATURE_SET="${FEATURE_SET:-kinpid}"
LABEL_SOURCE="${LABEL_SOURCE:-filename}" # filename aligns with class prefixes used in training command
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-part0_${FEATURE_SET}_seed${SEED}_train150k50k_test150k}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_TAG}_job${SLURM_JOB_ID:-manual}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/slurm_logs/${RUN_TAG}}"

# Train / eval controls
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
DEVICE="${DEVICE:-cuda}"

# Split by file index within each class
TRAIN_INDICES="${TRAIN_INDICES:-0-7}"
VAL_INDICES="${VAL_INDICES:-8}"
TEST_INDICES="${TEST_INDICES:-9}"

# Corruption study settings
NOISE_LEVELS="${NOISE_LEVELS:-0.01,0.03,0.05,0.1}"
DROPOUT_LEVELS="${DROPOUT_LEVELS:-0.05,0.1,0.2,0.3}"
MASK_LEVELS="${MASK_LEVELS:-0.05,0.1,0.2,0.3}"
JITTER_LEVELS="${JITTER_LEVELS:-0.01,0.03,0.05,0.1}"

# Optional eval-only mode
SKIP_TRAIN="${SKIP_TRAIN:-0}"
CHECKPOINT="${CHECKPOINT:-}"
TRAINER_LOG="${TRAINER_LOG:-}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

RUNTIME_LOG_STDOUT="${LOG_DIR}/job${SLURM_JOB_ID:-manual}.out"
RUNTIME_LOG_STDERR="${LOG_DIR}/job${SLURM_JOB_ID:-manual}.err"
exec > >(tee -a "${RUNTIME_LOG_STDOUT}") 2> >(tee -a "${RUNTIME_LOG_STDERR}" >&2)

echo "RUNTIME_LOG_STDOUT=${RUNTIME_LOG_STDOUT}"
echo "RUNTIME_LOG_STDERR=${RUNTIME_LOG_STDERR}"
echo "RUNTIME_RUN_DIR=${RUN_DIR}"
echo "============================================================"
echo "JetClass part0 train + shift-analysis run"
echo "Run dir: ${RUN_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Dataset: ${DATASET_DIR}"
echo "Feature set: ${FEATURE_SET}"
echo "Label source: ${LABEL_SOURCE}"
echo "Seed: ${SEED}"
echo "Epochs: ${EPOCHS}"
echo "Train/Val/Test targets: ${TRAIN_SAMPLES} / ${VAL_SAMPLES} / ${TEST_JETS}"
echo "GPU arg: ${GPUS}, Device: ${DEVICE}"
echo "Skip train: ${SKIP_TRAIN}"
echo "Checkpoint: ${CHECKPOINT:-[none]}"
echo "Trainer log: ${TRAINER_LOG:-[auto]}"
echo "============================================================"
echo

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
export PATH="$HOME/.local/bin:$PATH"

cd "${REPO_ROOT}"

CMD=(
  python3 run_jetclass_part0_baseline_and_shift.py
  --dataset-dir "${DATASET_DIR}"
  --output-dir "${RUN_DIR}"
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
  --tensorboard-name "JetClass_part0_${FEATURE_SET}_seed${SEED}"
  --noise-levels "${NOISE_LEVELS}"
  --dropout-levels "${DROPOUT_LEVELS}"
  --mask-levels "${MASK_LEVELS}"
  --jitter-levels "${JITTER_LEVELS}"
)

if [[ "${SKIP_TRAIN}" == "1" ]]; then
  if [[ -z "${CHECKPOINT}" ]]; then
    echo "ERROR: SKIP_TRAIN=1 requires CHECKPOINT=/path/to/saved_model.pt" >&2
    exit 1
  fi
  if [[ -z "${TRAINER_LOG}" ]]; then
    CANDIDATE_LOG="$(dirname "${CHECKPOINT}")/train.log"
    if [[ -f "${CANDIDATE_LOG}" ]]; then
      TRAINER_LOG="${CANDIDATE_LOG}"
    fi
  fi
  CMD+=(--skip-train --checkpoint "${CHECKPOINT}")
fi

if [[ -n "${TRAINER_LOG}" ]]; then
  CMD+=(--trainer-log "${TRAINER_LOG}")
fi

printf ' %q' "${CMD[@]}"
echo
echo
"${CMD[@]}"

echo
echo "Done. Key outputs:"
echo "  ${RUN_DIR}/saved_model.pt"
echo "  ${RUN_DIR}/summary.json"
echo "  ${RUN_DIR}/corruption_metrics.csv"
echo "  ${RUN_DIR}/correlations.csv"
echo "  ${RUNTIME_LOG_STDOUT}"
echo "  ${RUNTIME_LOG_STDERR}"
