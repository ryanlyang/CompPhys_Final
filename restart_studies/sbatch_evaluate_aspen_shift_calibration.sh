#!/usr/bin/env bash
#SBATCH --job-name=aspenShift
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=restart_studies/logs/aspen_shift_%j.out
#SBATCH --error=restart_studies/logs/aspen_shift_%j.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/atlas/CompPhys_FinalProject}"

SCRIPT_CANDIDATES=(
  "${WORKDIR}/restart_studies/evaluate_aspen_shift_calibration.py"
  "${PROJECT_ROOT}/restart_studies/evaluate_aspen_shift_calibration.py"
  "/home/ryreu/atlas/CompPhys_FinalProject/restart_studies/evaluate_aspen_shift_calibration.py"
  "/home/ryan/Documents/School/CompPhys/new_final_project/restart_studies/evaluate_aspen_shift_calibration.py"
)
SCRIPT_PATH=""
for c in "${SCRIPT_CANDIDATES[@]}"; do
  if [[ -f "${c}" ]]; then
    SCRIPT_PATH="${c}"
    break
  fi
done

if [[ -z "${SCRIPT_PATH}" ]]; then
  echo "ERROR: evaluate_aspen_shift_calibration.py not found." >&2
  echo "Checked: ${SCRIPT_CANDIDATES[*]}" >&2
  exit 2
fi

PROJECT_ROOT_RESOLVED="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT_RESOLVED}/restart_studies/results}"
RUN_BASENAME="${RUN_BASENAME:-prelim_reimpl_cluster}"
SEED="${SEED:-52}"
MODEL_RUN_NAME="${MODEL_RUN_NAME:-${RUN_BASENAME}_seed${SEED}}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT_RESOLVED}/restart_studies/results}"
OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-${MODEL_RUN_NAME}_aspen_shift_1M}"

ASPEN_DATA_DIR="${ASPEN_DATA_DIR:-/home/ryreu/atlas/CompPhys_Final/data/AspenOpenJets}"
ASPEN_GLOB="${ASPEN_GLOB:-Run*.h5}"
ASPEN_N_JETS="${ASPEN_N_JETS:-1000000}"
ASPEN_CHUNK_JETS="${ASPEN_CHUNK_JETS:-50000}"

JETCLASS_DATA_DIR="${JETCLASS_DATA_DIR:-}"
BATCH_SIZE="${BATCH_SIZE:--1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
CLIP_DELTA_MIN="${CLIP_DELTA_MIN:-0.0}"
CLIP_DELTA_MAX="${CLIP_DELTA_MAX:-1.0}"

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

echo "[preflight] WORKDIR=${WORKDIR}"
echo "[preflight] PROJECT_ROOT=${PROJECT_ROOT_RESOLVED}"
echo "[preflight] SCRIPT=${SCRIPT_PATH}"
echo "[preflight] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[preflight] MODEL_RUN_NAME=${MODEL_RUN_NAME}"
echo "[preflight] OUTPUT_RUN_NAME=${OUTPUT_RUN_NAME}"
echo "[preflight] ASPEN_DATA_DIR=${ASPEN_DATA_DIR}"
echo "[preflight] python: $(command -v "${PYTHON_BIN}" || true)"

"${PYTHON_BIN}" - << 'PY'
import importlib.util as iu
mods = ["torch", "numpy", "scipy", "h5py", "uproot", "awkward"]
missing = [m for m in mods if iu.find_spec(m) is None]
if missing:
    raise SystemExit(f"[preflight] Missing required Python modules: {missing}")
print("[preflight] module check OK")
PY

CMD=(
  "${PYTHON_BIN}" -u "${SCRIPT_PATH}"
  --results_root "${RESULTS_ROOT}"
  --run_basename "${RUN_BASENAME}"
  --seed "${SEED}"
  --model_run_name "${MODEL_RUN_NAME}"
  --aspen_data_dir "${ASPEN_DATA_DIR}"
  --aspen_glob "${ASPEN_GLOB}"
  --aspen_n_jets "${ASPEN_N_JETS}"
  --aspen_chunk_jets "${ASPEN_CHUNK_JETS}"
  --output_root "${OUTPUT_ROOT}"
  --output_run_name "${OUTPUT_RUN_NAME}"
  --device "${DEVICE}"
  --num_workers "${NUM_WORKERS}"
  --batch_size "${BATCH_SIZE}"
  --clip_delta_min "${CLIP_DELTA_MIN}"
  --clip_delta_max "${CLIP_DELTA_MAX}"
)

if [[ -n "${JETCLASS_DATA_DIR}" ]]; then
  CMD+=(--jetclass_data_dir "${JETCLASS_DATA_DIR}")
fi

echo "============================================================"
echo "Aspen Shift Calibration Evaluation (Single Seed)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Seed: ${SEED}"
echo "Model run: ${MODEL_RUN_NAME}"
echo "Output run: ${OUTPUT_RUN_NAME}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "[done] outputs:"
echo "  ${OUTPUT_ROOT}/${OUTPUT_RUN_NAME}/summary.json"
echo "  ${OUTPUT_ROOT}/${OUTPUT_RUN_NAME}/metric_to_deltaacc_mapping.csv"
echo "  ${OUTPUT_ROOT}/${OUTPUT_RUN_NAME}/aspen_shift_metrics.json"
echo "  ${OUTPUT_ROOT}/${OUTPUT_RUN_NAME}/aspen_predicted_deltaacc_by_metric.csv"
