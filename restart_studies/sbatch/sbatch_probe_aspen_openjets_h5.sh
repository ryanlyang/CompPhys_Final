#!/usr/bin/env bash
#SBATCH --job-name=aspenProbe
#SBATCH --partition=tier3
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=restart_studies/logs/aspen_probe_%j.out
#SBATCH --error=restart_studies/logs/aspen_probe_%j.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/atlas/CompPhys_FinalProject}"

SCRIPT_CANDIDATES=(
  "${WORKDIR}/restart_studies/probe_aspen_openjets_h5.py"
  "${PROJECT_ROOT}/restart_studies/probe_aspen_openjets_h5.py"
  "/home/ryreu/atlas/CompPhys_FinalProject/restart_studies/probe_aspen_openjets_h5.py"
  "/home/ryan/Documents/School/CompPhys/new_final_project/restart_studies/probe_aspen_openjets_h5.py"
)
SCRIPT_PATH=""
for c in "${SCRIPT_CANDIDATES[@]}"; do
  if [[ -f "${c}" ]]; then
    SCRIPT_PATH="${c}"
    break
  fi
done

if [[ -z "${SCRIPT_PATH}" ]]; then
  echo "ERROR: probe_aspen_openjets_h5.py not found." >&2
  echo "Checked: ${SCRIPT_CANDIDATES[*]}" >&2
  exit 2
fi

PROJECT_ROOT_RESOLVED="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT_RESOLVED}/restart_studies/results}"
RUN_NAME="${RUN_NAME:-aspen_probe_cluster}"

if [[ -n "${DATA_DIR:-}" ]]; then
  DATA_CANDIDATES=(
    "${DATA_DIR}"
    "/home/ryreu/atlas/CompPhys_Final/data/AspenOpenJets"
  )
else
  DATA_CANDIDATES=(
    "/home/ryreu/atlas/CompPhys_Final/data/AspenOpenJets"
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
  echo "ERROR: AspenOpenJets data directory not found." >&2
  echo "Checked: ${DATA_CANDIDATES[*]}" >&2
  exit 2
fi

GLOB_PATTERN="${GLOB_PATTERN:-Run*.h5}"
MAX_FILES="${MAX_FILES:-0}"
SAMPLE_JETS_PER_FILE="${SAMPLE_JETS_PER_FILE:-10000}"
SPLIT_FRACTIONS="${SPLIT_FRACTIONS:-0.8,0.1,0.1}"
SEED="${SEED:-52}"

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
echo "[preflight] DATA_DIR=${DATA_DIR_RESOLVED}"
echo "[preflight] python: $(command -v "${PYTHON_BIN}" || true)"

"${PYTHON_BIN}" - << 'PY'
import importlib.util as iu
mods = ["h5py", "numpy"]
missing = [m for m in mods if iu.find_spec(m) is None]
if missing:
    raise SystemExit(f"[preflight] Missing required Python modules: {missing}")
print("[preflight] module check OK")
PY

CMD=(
  "${PYTHON_BIN}" -u "${SCRIPT_PATH}"
  --data_dir "${DATA_DIR_RESOLVED}"
  --glob "${GLOB_PATTERN}"
  --max_files "${MAX_FILES}"
  --sample_jets_per_file "${SAMPLE_JETS_PER_FILE}"
  --split_fractions "${SPLIT_FRACTIONS}"
  --seed "${SEED}"
  --output_dir "${OUTPUT_ROOT}"
  --run_name "${RUN_NAME}"
)

echo "============================================================"
echo "AspenOpenJets H5 Probe"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Data: ${DATA_DIR_RESOLVED}"
echo "Run: ${OUTPUT_ROOT}/${RUN_NAME}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "[done] Wrote:"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/probe_summary.json"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/schema_summary.json"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/loader_recipe.json"
echo "  ${OUTPUT_ROOT}/${RUN_NAME}/file_manifest.csv"
