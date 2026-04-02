#!/usr/bin/env bash
#SBATCH --job-name=aspen_eval
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=sbatch/slurm_logs/slurm_aspen_eval_%j.out
#SBATCH --error=sbatch/slurm_logs/slurm_aspen_eval_%j.err

set -euo pipefail

# ---- User-tunable defaults ----
CONDA_ENV="${CONDA_ENV:-atlas_kd}"
ASPEN_DIR="${ASPEN_DIR:-/home/ryreu/atlas/CompPhys_Final/data/AspenOpenJets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ryreu/atlas/CompPhys_Final/runs/aspen_eval_shiftstudy}"
FEATURE_SET="${FEATURE_SET:-kinpid}"            # kin | kinpid | full
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-aspen_${FEATURE_SET}_seed${SEED}}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/slurm_logs}"

CHECKPOINT="${CHECKPOINT:-}"
REFERENCE_SUMMARY="${REFERENCE_SUMMARY:-}"

MAX_JETS="${MAX_JETS:-300000}"
MAX_FILES="${MAX_FILES:--1}"
MAX_NUM_PARTICLES="${MAX_NUM_PARTICLES:-128}"
READ_CHUNK_SIZE="${READ_CHUNK_SIZE:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
DEVICE="${DEVICE:-cuda}"
SKIP_CORRUPTIONS="${SKIP_CORRUPTIONS:-0}"

NOISE_LEVELS="${NOISE_LEVELS:-0.01,0.03,0.05,0.1}"
DROPOUT_LEVELS="${DROPOUT_LEVELS:-0.05,0.1,0.2,0.3}"
MASK_LEVELS="${MASK_LEVELS:-0.05,0.1,0.2,0.3}"
JITTER_LEVELS="${JITTER_LEVELS:-0.01,0.03,0.05,0.1}"

AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-0}"
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

if [[ -z "${CHECKPOINT}" ]]; then
  echo "ERROR: CHECKPOINT must be set to a valid saved_model.pt path." >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: CHECKPOINT does not exist: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -d "${ASPEN_DIR}" ]]; then
  echo "ERROR: ASPEN_DIR does not exist: ${ASPEN_DIR}" >&2
  exit 1
fi

SCRIPT_PATH="evaluate_aspen_openjets_shift.py"
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: cannot find ${SCRIPT_PATH} in $(pwd)" >&2
  exit 1
fi

JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_TAG}_job${JOB_ID}"
LOG_DIR="${LOG_ROOT}/${RUN_TAG}"
STDOUT_LOG="${LOG_DIR}/job${JOB_ID}.out"
STDERR_LOG="${LOG_DIR}/job${JOB_ID}.err"
mkdir -p "${RUN_DIR}" "${LOG_DIR}" "sbatch/slurm_logs"

echo "RUNTIME_LOG_STDOUT=${STDOUT_LOG}"
echo "RUNTIME_LOG_STDERR=${STDERR_LOG}"
echo "RUNTIME_RUN_DIR=${RUN_DIR}"

if [[ "${MIRROR_TO_SLURM_STDIO}" == "1" ]]; then
  exec > >(tee -a "${STDOUT_LOG}") 2> >(tee -a "${STDERR_LOG}" >&2)
else
  exec > >(tee -a "${STDOUT_LOG}" >/dev/null) 2> >(tee -a "${STDERR_LOG}" >/dev/null)
fi

missing_modules=()
for mod in torch h5py numpy; do
  if ! python3 -c "import ${mod}" >/dev/null 2>&1; then
    missing_modules+=("${mod}")
  fi
done
if ! python3 -c "import weaver.nn.model.ParticleTransformer" >/dev/null 2>&1; then
  missing_modules+=("weaver-core")
fi

if [[ "${#missing_modules[@]}" -gt 0 ]]; then
  if [[ "${AUTO_INSTALL_DEPS}" == "1" ]]; then
    echo "Installing missing deps: ${missing_modules[*]}"
    python3 -m pip install --upgrade pip
    python3 -m pip install --no-user 'weaver-core>=0.4' h5py
  else
    echo "ERROR: Missing dependencies: ${missing_modules[*]}" >&2
    echo "Set AUTO_INSTALL_DEPS=1 to install automatically." >&2
    exit 1
  fi
fi

CMD=(
  python3 "${SCRIPT_PATH}"
  --aspen-dir "${ASPEN_DIR}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${RUN_DIR}"
  --feature-set "${FEATURE_SET}"
  --max-num-particles "${MAX_NUM_PARTICLES}"
  --max-jets "${MAX_JETS}"
  --max-files "${MAX_FILES}"
  --read-chunk-size "${READ_CHUNK_SIZE}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --noise-levels "${NOISE_LEVELS}"
  --dropout-levels "${DROPOUT_LEVELS}"
  --mask-levels "${MASK_LEVELS}"
  --jitter-levels "${JITTER_LEVELS}"
)

if [[ -n "${REFERENCE_SUMMARY}" ]]; then
  CMD+=(--reference-summary "${REFERENCE_SUMMARY}")
fi
if [[ "${SKIP_CORRUPTIONS}" == "1" ]]; then
  CMD+=(--skip-corruptions)
fi

echo "============================================================"
echo "AspenOpenJets shift evaluation"
echo "Run dir: ${RUN_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Stdout log: ${STDOUT_LOG}"
echo "Stderr log: ${STDERR_LOG}"
echo "Aspen dir: ${ASPEN_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Reference summary: ${REFERENCE_SUMMARY:-[none]}"
echo "Feature set: ${FEATURE_SET}"
echo "Seed: ${SEED}"
echo "Max jets/files: ${MAX_JETS} / ${MAX_FILES}"
echo "Skip corruptions: ${SKIP_CORRUPTIONS}"
echo "Device: ${DEVICE}"
echo "Conda env: ${CONDA_ENV}"
echo "============================================================"
echo
printf ' %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo
echo "Done. Key outputs:"
echo "  ${RUN_DIR}/aspen_summary.json"
echo "  ${RUN_DIR}/aspen_corruption_metrics.csv"
echo "  ${RUN_DIR}/aspen_corruption_metrics.json"
echo "  ${RUN_DIR}/run_config.json"
echo "  ${STDOUT_LOG}"
echo "  ${STDERR_LOG}"
