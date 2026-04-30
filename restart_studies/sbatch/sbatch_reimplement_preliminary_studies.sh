#!/usr/bin/env bash
#SBATCH --job-name=prelimReimpl
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=restart_studies/logs/prelim_reimpl_%j.out
#SBATCH --error=restart_studies/logs/prelim_reimpl_%j.err

set -euo pipefail


WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/atlas/CompPhys_FinalProject}"

RUN_NAME="${RUN_NAME:-prelim_reimpl_seed52_cluster}"
SEED="${SEED:-52}"

FEATURE_MODE="${FEATURE_MODE:-full}"
MAX_CONSTITS="${MAX_CONSTITS:-128}"
N_TRAIN_JETS="${N_TRAIN_JETS:-150000}"
N_VAL_JETS="${N_VAL_JETS:-50000}"
N_TEST_JETS="${N_TEST_JETS:-150000}"

BATCH_SIZE="${BATCH_SIZE:-512}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
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

CORRUPTIONS="${CORRUPTIONS:-pt_noise:0.03,pt_noise:0.06,eta_phi_jitter:0.02,eta_phi_jitter:0.05,dropout:0.05,dropout:0.10,merge:0.10,merge:0.20,global_scale:0.03}"

EXPLAIN_SUBSET_SIZE="${EXPLAIN_SUBSET_SIZE:-20000}"
EXPLAIN_BATCH_SIZE="${EXPLAIN_BATCH_SIZE:-128}"
MASK_FRACS="${MASK_FRACS:-0.02,0.05,0.10,0.20}"
IG_STEPS="${IG_STEPS:-16}"
SMOOTHGRAD_SAMPLES="${SMOOTHGRAD_SAMPLES:-16}"
SMOOTHGRAD_SIGMA="${SMOOTHGRAD_SIGMA:-0.10}"
RANDOM_MASK_REPEATS="${RANDOM_MASK_REPEATS:-3}"

DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SHUFFLE_FILES="${SHUFFLE_FILES:-0}"

SCRIPT_CANDIDATES=(
  "${WORKDIR}/restart_studies/reimplement_preliminary_studies.py"
  "${PROJECT_ROOT}/restart_studies/reimplement_preliminary_studies.py"
  "/home/ryreu/atlas/CompPhys_FinalProject/restart_studies/reimplement_preliminary_studies.py"
  "/home/ryan/Documents/School/CompPhys/new_final_project/restart_studies/reimplement_preliminary_studies.py"
)
SCRIPT_PATH=""
for c in "${SCRIPT_CANDIDATES[@]}"; do
  if [[ -f "${c}" ]]; then
    SCRIPT_PATH="${c}"
    break
  fi
done

if [[ -z "${SCRIPT_PATH}" ]]; then
  echo "ERROR: reimplement_preliminary_studies.py not found." >&2
  echo "Checked: ${SCRIPT_CANDIDATES[*]}" >&2
  exit 2
fi

PROJECT_ROOT_RESOLVED="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT_RESOLVED}/restart_studies/results}"

if [[ -n "${DATA_DIR:-}" ]]; then
  DATA_CANDIDATES=(
    "${DATA_DIR}"
    "/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"
    "/home/ryreu/atlas/ATLAS-top-tagging-open-data/data/jetclass_part0"
    "/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/data/jetclass_part0"
  )
else
  DATA_CANDIDATES=(
    "/home/ryreu/atlas/PracticeTagging/data/jetclass_part0"
    "/home/ryreu/atlas/ATLAS-top-tagging-open-data/data/jetclass_part0"
    "/home/ryan/ComputerScience/ATLAS/HLT_Reco/ATLAS-top-tagging-open-data/data/jetclass_part0"
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
  echo "ERROR: jetclass_part0 data directory not found." >&2
  echo "Checked: ${DATA_CANDIDATES[*]}" >&2
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
echo "[preflight] DATA_DIR=${DATA_DIR_RESOLVED}"
echo "[preflight] python: $(command -v "${PYTHON_BIN}" || true)"

"${PYTHON_BIN}" - << 'PY'
import importlib.util as iu
mods = ["torch", "numpy", "sklearn", "scipy", "uproot", "awkward", "tqdm"]
missing = [m for m in mods if iu.find_spec(m) is None]
if missing:
    raise SystemExit(f"[preflight] Missing required Python modules: {missing}")
print("[preflight] module check OK")
PY

CMD=(
  "${PYTHON_BIN}" -u "${SCRIPT_PATH}"
  --data_dir "${DATA_DIR_RESOLVED}"
  --output_root "${OUTPUT_ROOT}"
  --run_name "${RUN_NAME}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --feature_mode "${FEATURE_MODE}"
  --max_constits "${MAX_CONSTITS}"
  --n_train_jets "${N_TRAIN_JETS}"
  --n_val_jets "${N_VAL_JETS}"
  --n_test_jets "${N_TEST_JETS}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --patience "${PATIENCE}"
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
  --corruptions "${CORRUPTIONS}"
  --explain_subset_size "${EXPLAIN_SUBSET_SIZE}"
  --explain_batch_size "${EXPLAIN_BATCH_SIZE}"
  --mask_fracs "${MASK_FRACS}"
  --ig_steps "${IG_STEPS}"
  --smoothgrad_samples "${SMOOTHGRAD_SAMPLES}"
  --smoothgrad_sigma "${SMOOTHGRAD_SIGMA}"
  --random_mask_repeats "${RANDOM_MASK_REPEATS}"
)

if [[ "${SHUFFLE_FILES}" == "1" ]]; then
  CMD+=(--shuffle_files)
fi

echo "============================================================"
echo "Preliminary Study Reimplementation (Single Seed)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Run: ${OUTPUT_ROOT}/${RUN_NAME}"
echo "Data: ${DATA_DIR_RESOLVED}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "[done] Results:"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/summary.json"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/clean_metrics.json"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/corruption_metrics.csv"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/correlations.csv"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/top_shift_metric_ranking.csv"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/method_effectiveness_summary.csv"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/sanity_checks.json"
