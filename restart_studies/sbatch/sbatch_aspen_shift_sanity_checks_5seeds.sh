#!/usr/bin/env bash
#SBATCH --job-name=aspenSanity5
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=restart_studies/logs/aspen_sanity5_%j.out
#SBATCH --error=restart_studies/logs/aspen_sanity5_%j.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/atlas/CompPhys_FinalProject}"

SCRIPT_CANDIDATES=(
  "${WORKDIR}/restart_studies/aspen_shift_sanity_checks_5seeds.py"
  "${PROJECT_ROOT}/restart_studies/aspen_shift_sanity_checks_5seeds.py"
  "/home/ryreu/atlas/CompPhys_FinalProject/restart_studies/aspen_shift_sanity_checks_5seeds.py"
  "/home/ryan/Documents/School/CompPhys/new_final_project/restart_studies/aspen_shift_sanity_checks_5seeds.py"
)
SCRIPT_PATH=""
for c in "${SCRIPT_CANDIDATES[@]}"; do
  if [[ -f "${c}" ]]; then
    SCRIPT_PATH="${c}"
    break
  fi
done

if [[ -z "${SCRIPT_PATH}" ]]; then
  echo "ERROR: aspen_shift_sanity_checks_5seeds.py not found." >&2
  echo "Checked: ${SCRIPT_CANDIDATES[*]}" >&2
  exit 2
fi

PROJECT_ROOT_RESOLVED="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT_RESOLVED}/restart_studies/results}"
RUN_BASENAME="${RUN_BASENAME:-prelim_reimpl_cluster}"
ASPEN_SUFFIX="${ASPEN_SUFFIX:-_aspen_shift_1M}"
SEEDS="${SEEDS:-41,52,63,74,85}"

OUTPUT_DIR="${OUTPUT_DIR:-}"

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
echo "[preflight] RUN_BASENAME=${RUN_BASENAME}"
echo "[preflight] ASPEN_SUFFIX=${ASPEN_SUFFIX}"
echo "[preflight] SEEDS=${SEEDS}"
echo "[preflight] python: $(command -v "${PYTHON_BIN}" || true)"

"${PYTHON_BIN}" - << 'PY'
import importlib.util as iu
mods = ["numpy"]
missing = [m for m in mods if iu.find_spec(m) is None]
if missing:
    raise SystemExit(f"[preflight] Missing required Python modules: {missing}")
print("[preflight] module check OK")
PY

CMD=(
  "${PYTHON_BIN}" -u "${SCRIPT_PATH}"
  --results_root "${RESULTS_ROOT}"
  --run_basename "${RUN_BASENAME}"
  --aspen_suffix "${ASPEN_SUFFIX}"
  --seeds "${SEEDS}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output_dir "${OUTPUT_DIR}")
fi

echo "============================================================"
echo "Aspen Shift Sanity Checks (5 Seeds)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

if [[ -n "${OUTPUT_DIR}" ]]; then
  FINAL_OUT="${OUTPUT_DIR}"
else
  FINAL_OUT="${RESULTS_ROOT}/${RUN_BASENAME}_aspen_sanity_5seeds"
fi

echo "[done] outputs in:"
echo "  ${FINAL_OUT}"
echo "[done] key files:"
echo "  ${FINAL_OUT}/per_seed_checks.csv"
echo "  ${FINAL_OUT}/aggregate_summary.json"
echo "  ${FINAL_OUT}/sanity_report.md"
