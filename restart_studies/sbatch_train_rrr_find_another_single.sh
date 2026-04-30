#!/usr/bin/env bash
#SBATCH --job-name=rrrFindAnother
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=restart_studies/logs/rrr_findanother_%j.out
#SBATCH --error=restart_studies/logs/rrr_findanother_%j.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/atlas/CompPhys_FinalProject}"

SCRIPT_CANDIDATES=(
  "${WORKDIR}/restart_studies/train_rrr_find_another_single.py"
  "${PROJECT_ROOT}/restart_studies/train_rrr_find_another_single.py"
  "/home/ryreu/atlas/CompPhys_FinalProject/restart_studies/train_rrr_find_another_single.py"
  "/home/ryan/Documents/School/CompPhys/new_final_project/restart_studies/train_rrr_find_another_single.py"
)
SCRIPT_PATH=""
for c in "${SCRIPT_CANDIDATES[@]}"; do
  if [[ -f "${c}" ]]; then
    SCRIPT_PATH="${c}"
    break
  fi
done

if [[ -z "${SCRIPT_PATH}" ]]; then
  echo "ERROR: train_rrr_find_another_single.py not found." >&2
  echo "Checked: ${SCRIPT_CANDIDATES[*]}" >&2
  exit 2
fi

PROJECT_ROOT_RESOLVED="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT_RESOLVED}/restart_studies/results}"

to_tag() {
  local x="${1}"
  echo "${x}" | sed -e 's/-/m/g' -e 's/\./p/g' -e 's/+//g'
}

RUN_BASENAME="${RUN_BASENAME:-rrr_findanother_seed52}"
A_SOURCE="${A_SOURCE:-input_grad}"
LAMBDA_RRR="${LAMBDA_RRR:-10}"
MASK_FRAC="${MASK_FRAC:-0.10}"
LAMBDA_TAG="$(to_tag "${LAMBDA_RRR}")"
MASK_TAG="$(to_tag "${MASK_FRAC}")"
RUN_NAME="${RUN_NAME:-${RUN_BASENAME}_${A_SOURCE}_lam${LAMBDA_TAG}_mask${MASK_TAG}}"
SEED="${SEED:-52}"

MAX_ITERATIONS="${MAX_ITERATIONS:-5}"

FEATURE_MODE="${FEATURE_MODE:-full}"
MAX_CONSTITS="${MAX_CONSTITS:-128}"
TRAIN_FILES_PER_CLASS="${TRAIN_FILES_PER_CLASS:-8}"
VAL_FILES_PER_CLASS="${VAL_FILES_PER_CLASS:-1}"
TEST_FILES_PER_CLASS="${TEST_FILES_PER_CLASS:-1}"
N_TRAIN_JETS="${N_TRAIN_JETS:-150000}"
N_VAL_JETS="${N_VAL_JETS:-50000}"
N_TEST_JETS="${N_TEST_JETS:-150000}"

BATCH_SIZE="${BATCH_SIZE:-512}"
RRR_BATCH_SIZE="${RRR_BATCH_SIZE:-128}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-7e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
EMBED_DIM="${EMBED_DIM:-128}"
NUM_HEADS="${NUM_HEADS:-8}"
NUM_LAYERS="${NUM_LAYERS:-6}"
FF_DIM="${FF_DIM:-512}"
DROPOUT="${DROPOUT:-0.1}"
TARGET_CLASS="${TARGET_CLASS:-HToBB}"
BACKGROUND_CLASS="${BACKGROUND_CLASS:-ZJetsToNuNu}"

ATTR_BATCH_SIZE="${ATTR_BATCH_SIZE:-128}"
IG_STEPS="${IG_STEPS:-16}"
SMOOTHGRAD_SAMPLES="${SMOOTHGRAD_SAMPLES:-16}"
SMOOTHGRAD_SIGMA="${SMOOTHGRAD_SIGMA:-0.10}"

ASPEN_GLOB="${ASPEN_GLOB:-Run*.h5}"
ASPEN_N_JETS="${ASPEN_N_JETS:-1000000}"
ASPEN_CHUNK_JETS="${ASPEN_CHUNK_JETS:-50000}"

DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SHUFFLE_FILES="${SHUFFLE_FILES:-0}"

if [[ -n "${DATA_DIR:-}" ]]; then
  DATA_CANDIDATES=(
    "${DATA_DIR}"
    "/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"
    "/home/ryreu/atlas/ATLAS-top-tagging-open-data/data/jetclass_part0"
  )
else
  DATA_CANDIDATES=(
    "/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"
    "/home/ryreu/atlas/ATLAS-top-tagging-open-data/data/jetclass_part0"
  )
fi

DATA_DIR_RESOLVED=""
for d in "${DATA_CANDIDATES[@]}"; do
  if [[ -d "${d}" ]]; then
    DATA_DIR_RESOLVED="${d}"
    break
  fi
done

if [[ -z "${DATA_DIR_RESOLVED}" ]]; then
  echo "ERROR: JetClass data directory not found." >&2
  echo "Checked: ${DATA_CANDIDATES[*]}" >&2
  exit 2
fi

if [[ -n "${ASPEN_DATA_DIR:-}" ]]; then
  ASPEN_CANDIDATES=(
    "${ASPEN_DATA_DIR}"
    "/home/ryreu/atlas/CompPhys_Final/data/AspenOpenJets"
  )
else
  ASPEN_CANDIDATES=(
    "/home/ryreu/atlas/CompPhys_Final/data/AspenOpenJets"
  )
fi

ASPEN_DIR_RESOLVED=""
for d in "${ASPEN_CANDIDATES[@]}"; do
  if [[ -d "${d}" ]]; then
    ASPEN_DIR_RESOLVED="${d}"
    break
  fi
done

if [[ -z "${ASPEN_DIR_RESOLVED}" ]]; then
  echo "ERROR: AspenOpenJets directory not found." >&2
  echo "Checked: ${ASPEN_CANDIDATES[*]}" >&2
  exit 2
fi

CONDA_ENV="${CONDA_ENV:-atlas_kd}"
CONDA_SH_CANDIDATES=(
  "${CONDA_SH:-}"
  "/home/ryreu/miniconda3/etc/profile.d/conda.sh"
  "$HOME/miniforge3/etc/profile.d/conda.sh"
  "$HOME/miniconda3/etc/profile.d/conda.sh"
)
CONDA_SH_RESOLVED=""
for c in "${CONDA_SH_CANDIDATES[@]}"; do
  if [[ -n "${c}" && -f "${c}" ]]; then
    CONDA_SH_RESOLVED="${c}"
    break
  fi
done

if [[ -n "${CONDA_SH_RESOLVED}" ]]; then
  source "${CONDA_SH_RESOLVED}"
  conda activate "${CONDA_ENV}"
else
  echo "[preflight] WARNING: no conda.sh found; using current shell Python." >&2
fi

cd "${WORKDIR}"
mkdir -p "${PROJECT_ROOT_RESOLVED}/restart_studies/logs"

PYTHON_BIN="${PYTHON_BIN:-python3}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONHASHSEED="${SEED}"
export PYTHONWARNINGS="ignore"

echo "[preflight] WORKDIR=${WORKDIR}"
echo "[preflight] PROJECT_ROOT=${PROJECT_ROOT_RESOLVED}"
echo "[preflight] SCRIPT=${SCRIPT_PATH}"
echo "[preflight] RUN_NAME=${RUN_NAME}"
echo "[preflight] A_SOURCE=${A_SOURCE}"
echo "[preflight] LAMBDA_RRR=${LAMBDA_RRR}"
echo "[preflight] MASK_FRAC=${MASK_FRAC}"
echo "[preflight] DATA_DIR=${DATA_DIR_RESOLVED}"
echo "[preflight] ASPEN_DIR=${ASPEN_DIR_RESOLVED}"
echo "[preflight] python: $(command -v "${PYTHON_BIN}" || true)"

"${PYTHON_BIN}" - << 'PY'
import importlib.util as iu
mods = ["torch", "numpy", "scipy", "sklearn", "h5py", "uproot", "awkward", "tqdm"]
missing = [m for m in mods if iu.find_spec(m) is None]
if missing:
    raise SystemExit(f"[preflight] Missing required Python modules: {missing}")
print("[preflight] module check OK")
PY

CMD=(
  "${PYTHON_BIN}" -u "${SCRIPT_PATH}"
  --data_dir "${DATA_DIR_RESOLVED}"
  --aspen_data_dir "${ASPEN_DIR_RESOLVED}"
  --aspen_glob "${ASPEN_GLOB}"
  --aspen_n_jets "${ASPEN_N_JETS}"
  --aspen_chunk_jets "${ASPEN_CHUNK_JETS}"
  --output_root "${OUTPUT_ROOT}"
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --feature_mode "${FEATURE_MODE}"
  --max_constits "${MAX_CONSTITS}"
  --train_files_per_class "${TRAIN_FILES_PER_CLASS}"
  --val_files_per_class "${VAL_FILES_PER_CLASS}"
  --test_files_per_class "${TEST_FILES_PER_CLASS}"
  --n_train_jets "${N_TRAIN_JETS}"
  --n_val_jets "${N_VAL_JETS}"
  --n_test_jets "${N_TEST_JETS}"
  --batch_size "${BATCH_SIZE}"
  --rrr_batch_size "${RRR_BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --embed_dim "${EMBED_DIM}"
  --num_heads "${NUM_HEADS}"
  --num_layers "${NUM_LAYERS}"
  --ff_dim "${FF_DIM}"
  --dropout "${DROPOUT}"
  --target_class "${TARGET_CLASS}"
  --background_class "${BACKGROUND_CLASS}"
  --a_source "${A_SOURCE}"
  --lambda_rrr "${LAMBDA_RRR}"
  --mask_frac "${MASK_FRAC}"
  --max_iterations "${MAX_ITERATIONS}"
  --attr_batch_size "${ATTR_BATCH_SIZE}"
  --ig_steps "${IG_STEPS}"
  --smoothgrad_samples "${SMOOTHGRAD_SAMPLES}"
  --smoothgrad_sigma "${SMOOTHGRAD_SIGMA}"
)

if [[ "${SHUFFLE_FILES}" == "1" ]]; then
  CMD+=(--shuffle_files)
fi

echo "============================================================"
echo "RRR Find-Another Training (Single Config)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Output: ${OUTPUT_ROOT}/${RUN_NAME}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "[done] outputs:"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/config.json"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/iteration_summary.csv"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/summary.json"

