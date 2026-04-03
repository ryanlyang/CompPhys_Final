#!/usr/bin/env bash
#SBATCH --job-name=jcpart0_rrr
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --output=slurm_jcpart0_rrr_%j.out
#SBATCH --error=slurm_jcpart0_rrr_%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
DATASET_DIR="${DATASET_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ryreu/atlas/CompPhys_Final/runs/jetclass_part0_rrr}"

FEATURE_SET="${FEATURE_SET:-kinpid}"
LABEL_SOURCE="${LABEL_SOURCE:-filename}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-part0_${FEATURE_SET}_seed${SEED}_rrr50k10k50k}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_TAG}_job${SLURM_JOB_ID:-manual}}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/slurm_logs/${RUN_TAG}}"

# Data volumes
TRAIN_JETS="${TRAIN_JETS:-50000}"
VAL_JETS="${VAL_JETS:-10000}"
TEST_JETS="${TEST_JETS:-50000}"

# Split by per-class file index
TRAIN_INDICES="${TRAIN_INDICES:-0-7}"
VAL_INDICES="${VAL_INDICES:-8}"
TEST_INDICES="${TEST_INDICES:-9}"

# Training hyperparameters
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-192}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
MAX_NUM_PARTICLES="${MAX_NUM_PARTICLES:-128}"
DEVICE="${DEVICE:-cuda}"

# RRR controls
LAMBDA_RRR="${LAMBDA_RRR:-8.0}"
RRR_START_EPOCH="${RRR_START_EPOCH:-2}"
RRR_SCORE_MODE="${RRR_SCORE_MODE:-sum_log_probs}"   # sum_log_probs | true_log_prob | pred_log_prob
RRR_MASK_MODE="${RRR_MASK_MODE:-adaptive_topk}"     # all | adaptive_topk
ADAPTIVE_TOPK_FEATURES="${ADAPTIVE_TOPK_FEATURES:-5}"
ADAPTIVE_MASK_FLOOR="${ADAPTIVE_MASK_FLOOR:-0.15}"
ADAPTIVE_REFRESH_EPOCHS="${ADAPTIVE_REFRESH_EPOCHS:-1}"
ADAPTIVE_PROBE_JETS="${ADAPTIVE_PROBE_JETS:-20000}"

# Post-train shift analysis (same metrics pipeline as baseline)
RUN_SHIFT_ANALYSIS="${RUN_SHIFT_ANALYSIS:-1}"
NOISE_LEVELS="${NOISE_LEVELS:-0.01,0.03,0.05,0.1}"
DROPOUT_LEVELS="${DROPOUT_LEVELS:-0.05,0.1,0.2,0.3}"
MASK_LEVELS="${MASK_LEVELS:-0.05,0.1,0.2,0.3}"
JITTER_LEVELS="${JITTER_LEVELS:-0.01,0.03,0.05,0.1}"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"

RUNTIME_LOG_STDOUT="${LOG_DIR}/job${SLURM_JOB_ID:-manual}.out"
RUNTIME_LOG_STDERR="${LOG_DIR}/job${SLURM_JOB_ID:-manual}.err"
exec > >(tee -a "${RUNTIME_LOG_STDOUT}") 2> >(tee -a "${RUNTIME_LOG_STDERR}" >&2)

echo "RUNTIME_LOG_STDOUT=${RUNTIME_LOG_STDOUT}"
echo "RUNTIME_LOG_STDERR=${RUNTIME_LOG_STDERR}"
echo "RUNTIME_RUN_DIR=${RUN_DIR}"
echo "============================================================"
echo "JetClass part0 RRR training + shift analysis"
echo "Run dir: ${RUN_DIR}"
echo "Dataset: ${DATASET_DIR}"
echo "Feature set: ${FEATURE_SET}"
echo "Label source: ${LABEL_SOURCE}"
echo "Seed: ${SEED}"
echo "Jets train/val/test: ${TRAIN_JETS}/${VAL_JETS}/${TEST_JETS}"
echo "Epochs: ${EPOCHS}"
echo "RRR lambda: ${LAMBDA_RRR}"
echo "RRR mode: ${RRR_MASK_MODE} (score=${RRR_SCORE_MODE})"
echo "Run shift analysis: ${RUN_SHIFT_ANALYSIS}"
echo "============================================================"
echo

set +u
source ~/.bashrc
set -u
conda activate atlas_kd
export PATH="$HOME/.local/bin:$PATH"

cd "${REPO_ROOT}"

TRAIN_CMD=(
  python3 train_jetclass_part0_rrr.py
  --dataset-dir "${DATASET_DIR}"
  --output-dir "${RUN_DIR}"
  --feature-set "${FEATURE_SET}"
  --label-source "${LABEL_SOURCE}"
  --train-indices "${TRAIN_INDICES}"
  --val-indices "${VAL_INDICES}"
  --test-indices "${TEST_INDICES}"
  --max-train-jets "${TRAIN_JETS}"
  --max-val-jets "${VAL_JETS}"
  --max-test-jets "${TEST_JETS}"
  --max-num-particles "${MAX_NUM_PARTICLES}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --grad-clip "${GRAD_CLIP}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --lambda-rrr "${LAMBDA_RRR}"
  --rrr-start-epoch "${RRR_START_EPOCH}"
  --rrr-score-mode "${RRR_SCORE_MODE}"
  --rrr-mask-mode "${RRR_MASK_MODE}"
  --adaptive-topk-features "${ADAPTIVE_TOPK_FEATURES}"
  --adaptive-mask-floor "${ADAPTIVE_MASK_FLOOR}"
  --adaptive-refresh-epochs "${ADAPTIVE_REFRESH_EPOCHS}"
  --adaptive-probe-jets "${ADAPTIVE_PROBE_JETS}"
)

echo "--- Training (RRR) ---"
printf ' %q' "${TRAIN_CMD[@]}"
echo
"${TRAIN_CMD[@]}"

CHECKPOINT="${RUN_DIR}/saved_model.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: expected checkpoint missing: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ "${RUN_SHIFT_ANALYSIS}" == "1" ]]; then
  ANALYSIS_DIR="${RUN_DIR}/shift_eval"
  mkdir -p "${ANALYSIS_DIR}"

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
    --train-batch-size "${BATCH_SIZE}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --samples-per-epoch "${TRAIN_JETS}"
    --samples-per-epoch-val "${VAL_JETS}"
    --max-test-jets "${TEST_JETS}"
    --num-workers 6
    --fetch-step 0.01
    --start-lr "${LR}"
    --gpus 0
    --device "${DEVICE}"
    --seed "${SEED}"
    --tensorboard-name "JetClass_part0_${FEATURE_SET}_seed${SEED}_rrr_eval"
    --noise-levels "${NOISE_LEVELS}"
    --dropout-levels "${DROPOUT_LEVELS}"
    --mask-levels "${MASK_LEVELS}"
    --jitter-levels "${JITTER_LEVELS}"
    --skip-train
    --checkpoint "${CHECKPOINT}"
  )

  echo
  echo "--- Post-train shift analysis ---"
  printf ' %q' "${ANALYSIS_CMD[@]}"
  echo
  "${ANALYSIS_CMD[@]}"
fi

echo
echo "Done. Outputs:"
echo "  ${RUN_DIR}/saved_model.pt"
echo "  ${RUN_DIR}/summary.json"
echo "  ${RUN_DIR}/history.csv"
if [[ "${RUN_SHIFT_ANALYSIS}" == "1" ]]; then
  echo "  ${RUN_DIR}/shift_eval/summary.json"
  echo "  ${RUN_DIR}/shift_eval/correlations.csv"
fi
echo "  ${RUNTIME_LOG_STDOUT}"
echo "  ${RUNTIME_LOG_STDERR}"
