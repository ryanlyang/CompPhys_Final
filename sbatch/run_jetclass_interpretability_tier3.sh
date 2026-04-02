#!/usr/bin/env bash
#SBATCH --job-name=jcpart0_interp
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=sbatch/slurm_logs/slurm_jcpart0_interp_%j.out
#SBATCH --error=sbatch/slurm_logs/slurm_jcpart0_interp_%j.err

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-atlas_kd}"
DATASET_DIR="${DATASET_DIR:-/home/ryreu/atlas/PracticeTagging/data/jetclass_part0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ryreu/atlas/CompPhys_Final/runs/jetclass_part0_interpretability}"
FEATURE_SET="${FEATURE_SET:-kinpid}"
SEED="${SEED:-1}"
RUN_TAG="${RUN_TAG:-part0_${FEATURE_SET}_seed${SEED}_interp}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/slurm_logs}"

# Required: path to trained checkpoint (saved_model.pt)
CHECKPOINT="${CHECKPOINT:-}"

TRAIN_INDICES="${TRAIN_INDICES:-0-7}"
VAL_INDICES="${VAL_INDICES:-8}"
TEST_INDICES="${TEST_INDICES:-9}"

MAX_EVAL_JETS="${MAX_EVAL_JETS:-120000}"
MAX_EXPLAIN_JETS="${MAX_EXPLAIN_JETS:-20000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
ATTR_BATCH_SIZE="${ATTR_BATCH_SIZE:-96}"
DEVICE="${DEVICE:-cuda}"

METHODS="${METHODS:-input_gradients,integrated_gradients,smoothgrad}"
TARGET_MODE="${TARGET_MODE:-pred}"  # pred | true
MASK_FRACTIONS="${MASK_FRACTIONS:-0.02,0.05,0.1,0.2}"
RANDOM_REPEATS="${RANDOM_REPEATS:-3}"
IG_STEPS="${IG_STEPS:-16}"
SMOOTHGRAD_SAMPLES="${SMOOTHGRAD_SAMPLES:-16}"
SMOOTHGRAD_SIGMA="${SMOOTHGRAD_SIGMA:-0.02}"

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

JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_TAG}_job${JOB_ID}"
LOG_DIR="${LOG_ROOT}/${RUN_TAG}"
STDOUT_LOG="${LOG_DIR}/job${JOB_ID}.out"
STDERR_LOG="${LOG_DIR}/job${JOB_ID}.err"
mkdir -p "${RUN_DIR}" "${LOG_DIR}"

echo "RUNTIME_LOG_STDOUT=${STDOUT_LOG}"
echo "RUNTIME_LOG_STDERR=${STDERR_LOG}"
echo "RUNTIME_RUN_DIR=${RUN_DIR}"

if [[ "${MIRROR_TO_SLURM_STDIO}" == "1" ]]; then
  exec > >(tee -a "${STDOUT_LOG}") 2> >(tee -a "${STDERR_LOG}" >&2)
else
  exec > >(tee -a "${STDOUT_LOG}" >/dev/null) 2> >(tee -a "${STDERR_LOG}" >/dev/null)
fi

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "ERROR: DATASET_DIR does not exist: ${DATASET_DIR}" >&2
  exit 1
fi
if [[ -z "${CHECKPOINT}" ]]; then
  echo "ERROR: CHECKPOINT must be set to /abs/path/to/saved_model.pt" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: CHECKPOINT not found: ${CHECKPOINT}" >&2
  exit 1
fi

SCRIPT_PATH="run_jetclass_interpretability_benchmark.py"
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
    python3 -m pip install --no-user 'weaver-core>=0.4' awkward vector uproot
    export PATH="${HOME}/.local/bin:${PATH}"
  else
    echo "ERROR: Missing dependencies: ${missing_modules[*]}" >&2
    echo "Set AUTO_INSTALL_DEPS=1 to install automatically, or pre-install manually." >&2
    exit 1
  fi
fi

CMD=(
  python3 "${SCRIPT_PATH}"
  --dataset-dir "${DATASET_DIR}"
  --checkpoint "${CHECKPOINT}"
  --output-dir "${RUN_DIR}"
  --feature-set "${FEATURE_SET}"
  --device "${DEVICE}"
  --train-indices "${TRAIN_INDICES}"
  --val-indices "${VAL_INDICES}"
  --test-indices "${TEST_INDICES}"
  --max-eval-jets "${MAX_EVAL_JETS}"
  --max-explain-jets "${MAX_EXPLAIN_JETS}"
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

echo "============================================================"
echo "JetClass interpretability benchmark"
echo "Run dir: ${RUN_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Stdout log: ${STDOUT_LOG}"
echo "Stderr log: ${STDERR_LOG}"
echo "Dataset: ${DATASET_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Feature set: ${FEATURE_SET}"
echo "Methods: ${METHODS}"
echo "Target mode: ${TARGET_MODE}"
echo "Mask fractions: ${MASK_FRACTIONS}"
echo "Eval jets: ${MAX_EVAL_JETS}"
echo "Explain jets: ${MAX_EXPLAIN_JETS}"
echo "Conda env: ${CONDA_ENV}"
echo "============================================================"
echo
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
echo "  ${STDOUT_LOG}"
echo "  ${STDERR_LOG}"
