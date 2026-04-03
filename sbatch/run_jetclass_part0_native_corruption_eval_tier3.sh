#!/usr/bin/env bash
#SBATCH --job-name=jcpart0_native
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_jcpart0_native_%j.out
#SBATCH --error=slurm_jcpart0_native_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATASET_DIR="${DATASET_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ryreu/atlas/CompPhys_Final/runs/jetclass_part0_native_shiftstudy}"

CHECKPOINT="${CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
  echo "ERROR: CHECKPOINT is required. Set CHECKPOINT=/path/to/saved_model.pt" >&2
  exit 1
fi

FEATURE_SET="${FEATURE_SET:-kinpid}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-part0_${FEATURE_SET}_seed${SEED}_nativecorrupt}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_TAG}_job${SLURM_JOB_ID:-manual}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/slurm_logs/${RUN_TAG}}"

TRAIN_INDICES="${TRAIN_INDICES:-0-7}"
VAL_INDICES="${VAL_INDICES:-8}"
TEST_INDICES="${TEST_INDICES:-9}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-6}"
FETCH_STEP="${FETCH_STEP:-0.01}"
GPUS="${GPUS:-0}"

NOISE_LEVELS="${NOISE_LEVELS:-0.01,0.03,0.05,0.1}"
DROPOUT_LEVELS="${DROPOUT_LEVELS:-0.05,0.1,0.2,0.3}"
MASK_LEVELS="${MASK_LEVELS:-0.05,0.1,0.2,0.3}"
JITTER_LEVELS="${JITTER_LEVELS:-0.01,0.03,0.05,0.1}"

KEEP_CORRUPTED_FILES="${KEEP_CORRUPTED_FILES:-0}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

RUNTIME_LOG_STDOUT="${LOG_DIR}/job${SLURM_JOB_ID:-manual}.out"
RUNTIME_LOG_STDERR="${LOG_DIR}/job${SLURM_JOB_ID:-manual}.err"
exec > >(tee -a "${RUNTIME_LOG_STDOUT}") 2> >(tee -a "${RUNTIME_LOG_STDERR}" >&2)

echo "RUNTIME_LOG_STDOUT=${RUNTIME_LOG_STDOUT}"
echo "RUNTIME_LOG_STDERR=${RUNTIME_LOG_STDERR}"
echo "RUNTIME_RUN_DIR=${RUN_DIR}"
echo "============================================================"
echo "JetClass part0 native corruption evaluation"
echo "Run dir: ${RUN_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Dataset: ${DATASET_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Feature set: ${FEATURE_SET}"
echo "Seed: ${SEED}"
echo "Train/Val/Test file indices: ${TRAIN_INDICES} / ${VAL_INDICES} / ${TEST_INDICES}"
echo "GPU arg: ${GPUS}"
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

CMD=(
  python3 native_corruption_eval.py
  --dataset-dir "${DATASET_DIR}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${RUN_DIR}"
  --feature-set "${FEATURE_SET}"
  --train-indices "${TRAIN_INDICES}"
  --val-indices "${VAL_INDICES}"
  --test-indices "${TEST_INDICES}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --fetch-step "${FETCH_STEP}"
  --gpus "${GPUS}"
  --noise-levels "${NOISE_LEVELS}"
  --dropout-levels "${DROPOUT_LEVELS}"
  --mask-levels "${MASK_LEVELS}"
  --jitter-levels "${JITTER_LEVELS}"
  --seed "${SEED}"
)

if [[ "${KEEP_CORRUPTED_FILES}" == "1" ]]; then
  CMD+=(--keep-corrupted-files)
fi

printf ' %q' "${CMD[@]}"
echo
echo
"${CMD[@]}"

echo
echo "Done. Key outputs:"
echo "  ${RUN_DIR}/summary.json"
echo "  ${RUN_DIR}/corruption_metrics.csv"
echo "  ${RUN_DIR}/correlations.csv"
echo "  ${RUNTIME_LOG_STDOUT}"
echo "  ${RUNTIME_LOG_STDERR}"

