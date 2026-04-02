#!/usr/bin/env bash
#SBATCH --job-name=jcpart0_shift
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=sbatch/slurm_logs/slurm_jcpart0_shift_%j.out
#SBATCH --error=sbatch/slurm_logs/slurm_jcpart0_shift_%j.err

set -euo pipefail

# ---- User-tunable defaults ----
CONDA_ENV="${CONDA_ENV:-atlas_kd}"
DATASET_DIR="${DATASET_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ryreu/atlas/CompPhys_Final/runs/jetclass_part0_shiftstudy}"
FEATURE_SET="${FEATURE_SET:-kinpid}"           # kin | kinpid | full
SEED="${SEED:-1}"
RUN_TAG="${RUN_TAG:-part0_${FEATURE_SET}_seed${SEED}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/slurm_logs}"

# File-index split inside each class. Defaults assume *_000..*_009 exist.
TRAIN_INDICES="${TRAIN_INDICES:-0-7}"
VAL_INDICES="${VAL_INDICES:-8}"
TEST_INDICES="${TEST_INDICES:-9}"

# Requested sample counts.
TRAIN_JETS_PER_EPOCH="${TRAIN_JETS_PER_EPOCH:-150000}"
VAL_JETS_PER_EPOCH="${VAL_JETS_PER_EPOCH:-50000}"
MAX_TEST_JETS="${MAX_TEST_JETS:-300000}"

EPOCHS="${EPOCHS:-12}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-6}"
FETCH_STEP="${FETCH_STEP:-0.01}"
START_LR="${START_LR:-1e-3}"
GPUS="${GPUS:-0}"
DEVICE="${DEVICE:-cuda}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
CHECKPOINT="${CHECKPOINT:-}"

# Corruption severities.
NOISE_LEVELS="${NOISE_LEVELS:-0.01,0.03,0.05,0.1}"
DROPOUT_LEVELS="${DROPOUT_LEVELS:-0.05,0.1,0.2,0.3}"
MASK_LEVELS="${MASK_LEVELS:-0.05,0.1,0.2,0.3}"
JITTER_LEVELS="${JITTER_LEVELS:-0.01,0.03,0.05,0.1}"

# Set AUTO_INSTALL_DEPS=1 to install missing packages into current env.
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-0}"
# Set MIRROR_TO_SLURM_STDIO=0 to disable runtime mirroring to Slurm stdout/stderr.
# Default keeps logs in LOG_DIR and also mirrors to Slurm for easier debugging.
MIRROR_TO_SLURM_STDIO="${MIRROR_TO_SLURM_STDIO:-1}"

set +u
source ~/.bashrc
set -u
conda activate "${CONDA_ENV}"
export PATH="${HOME}/.local/bin:${PATH}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_TAG}_job${JOB_ID}"
LOG_DIR="${LOG_ROOT}/${RUN_TAG}"
STDOUT_LOG="${LOG_DIR}/job${JOB_ID}.out"
STDERR_LOG="${LOG_DIR}/job${JOB_ID}.err"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

# Print deterministic log locations early (before stdio redirection) so they
# are always visible in Slurm's own stdout/stderr files.
echo "RUNTIME_LOG_STDOUT=${STDOUT_LOG}"
echo "RUNTIME_LOG_STDERR=${STDERR_LOG}"
echo "RUNTIME_RUN_DIR=${RUN_DIR}"

# Route all script logs to a stable folder. By default do not mirror runtime
# logs back to Slurm stdout/stderr (to avoid clutter in submit directory logs).
if [[ "${MIRROR_TO_SLURM_STDIO}" == "1" ]]; then
  exec > >(tee -a "${STDOUT_LOG}") 2> >(tee -a "${STDERR_LOG}" >&2)
else
  exec > >(tee -a "${STDOUT_LOG}" >/dev/null) 2> >(tee -a "${STDERR_LOG}" >/dev/null)
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "ERROR: DATASET_DIR does not exist: ${DATASET_DIR}" >&2
  exit 1
fi

SCRIPT_PATH="run_jetclass_part0_baseline_and_shift.py"
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: cannot find ${SCRIPT_PATH} in $(pwd)" >&2
  exit 1
fi

missing_modules=()
for mod in torch awkward vector uproot; do
  if ! python3 -c "import ${mod}" >/dev/null 2>&1; then
    missing_modules+=("${mod}")
  fi
done

if ! python3 -c "import weaver.train" >/dev/null 2>&1; then
  missing_modules+=("weaver-core")
fi

if [[ "${#missing_modules[@]}" -gt 0 ]]; then
  if [[ "${AUTO_INSTALL_DEPS}" == "1" ]]; then
    echo "Installing missing deps: ${missing_modules[*]}"
    python3 -m pip install --upgrade pip
    # weaver-core pulls related training dependencies.
    python3 -m pip install --no-user 'weaver-core>=0.4' awkward vector uproot
    export PATH="${HOME}/.local/bin:${PATH}"
  else
    echo "ERROR: Missing dependencies: ${missing_modules[*]}" >&2
    echo "Set AUTO_INSTALL_DEPS=1 to install automatically, or pre-install manually." >&2
    exit 1
  fi
fi

TENSORBOARD_NAME="${TENSORBOARD_NAME:-JetClass_part0_${FEATURE_SET}_seed${SEED}}"

CMD=(
  python3 "${SCRIPT_PATH}"
  --dataset-dir "${DATASET_DIR}"
  --output-dir "${RUN_DIR}"
  --feature-set "${FEATURE_SET}"
  --train-indices "${TRAIN_INDICES}"
  --val-indices "${VAL_INDICES}"
  --test-indices "${TEST_INDICES}"
  --epochs "${EPOCHS}"
  --train-batch-size "${TRAIN_BATCH_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --samples-per-epoch "${TRAIN_JETS_PER_EPOCH}"
  --samples-per-epoch-val "${VAL_JETS_PER_EPOCH}"
  --max-test-jets "${MAX_TEST_JETS}"
  --num-workers "${NUM_WORKERS}"
  --fetch-step "${FETCH_STEP}"
  --start-lr "${START_LR}"
  --gpus "${GPUS}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --tensorboard-name "${TENSORBOARD_NAME}"
  --noise-levels "${NOISE_LEVELS}"
  --dropout-levels "${DROPOUT_LEVELS}"
  --mask-levels "${MASK_LEVELS}"
  --jitter-levels "${JITTER_LEVELS}"
)

if [[ "${SKIP_TRAIN}" == "1" ]]; then
  if [[ -z "${CHECKPOINT}" ]]; then
    echo "ERROR: SKIP_TRAIN=1 requires CHECKPOINT=/abs/path/to/saved_model.pt" >&2
    exit 1
  fi
  CMD+=(--skip-train --checkpoint "${CHECKPOINT}")
fi

echo "============================================================"
echo "JetClass part0 baseline + shift-correlation run"
echo "Run dir: ${RUN_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Stdout log: ${STDOUT_LOG}"
echo "Stderr log: ${STDERR_LOG}"
echo "Dataset: ${DATASET_DIR}"
echo "Feature set: ${FEATURE_SET}"
echo "Seed: ${SEED}"
echo "Epochs: ${EPOCHS}"
echo "Train/Val/Test targets: ${TRAIN_JETS_PER_EPOCH} / ${VAL_JETS_PER_EPOCH} / ${MAX_TEST_JETS}"
echo "GPU arg: ${GPUS}, Device: ${DEVICE}"
echo "Skip train: ${SKIP_TRAIN}"
if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "Checkpoint: ${CHECKPOINT}"
fi
echo "Conda env: ${CONDA_ENV}"
echo "============================================================"
echo
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
echo "  ${STDOUT_LOG}"
echo "  ${STDERR_LOG}"
