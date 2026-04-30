#!/usr/bin/env bash
#SBATCH --job-name=rrrAgg48
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=restart_studies/logs/rrr_agg48_%j.out
#SBATCH --error=restart_studies/logs/rrr_agg48_%j.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/atlas/CompPhys_FinalProject}"

SCRIPT_CANDIDATES=(
  "${WORKDIR}/restart_studies/aggregate_rrr_find_another_sweep.py"
  "${PROJECT_ROOT}/restart_studies/aggregate_rrr_find_another_sweep.py"
  "/home/ryreu/atlas/CompPhys_FinalProject/restart_studies/aggregate_rrr_find_another_sweep.py"
  "/home/ryan/Documents/School/CompPhys/new_final_project/restart_studies/aggregate_rrr_find_another_sweep.py"
)
SCRIPT_PATH=""
for c in "${SCRIPT_CANDIDATES[@]}"; do
  if [[ -f "${c}" ]]; then
    SCRIPT_PATH="${c}"
    break
  fi
done

if [[ -z "${SCRIPT_PATH}" ]]; then
  echo "ERROR: aggregate_rrr_find_another_sweep.py not found." >&2
  echo "Checked: ${SCRIPT_CANDIDATES[*]}" >&2
  exit 2
fi

PROJECT_ROOT_RESOLVED="$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT_RESOLVED}/restart_studies/results}"
RUN_BASENAME="${RUN_BASENAME:-rrr_findanother_seed52}"
AGG_RUN_NAME="${AGG_RUN_NAME:-${RUN_BASENAME}_aggregate_48cfg}"
A_SOURCES="${A_SOURCES:-input_grad,integrated_gradients,smoothgrad}"
LAMBDA_VALUES="${LAMBDA_VALUES:-1,10,100,1000}"
MASK_FRACS="${MASK_FRACS:-0.05,0.10,0.20,0.30}"

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
echo "[preflight] AGG_RUN_NAME=${AGG_RUN_NAME}"
echo "[preflight] A_SOURCES=${A_SOURCES}"
echo "[preflight] LAMBDA_VALUES=${LAMBDA_VALUES}"
echo "[preflight] MASK_FRACS=${MASK_FRACS}"
echo "[preflight] python: $(command -v "${PYTHON_BIN}" || true)"

CMD=(
  "${PYTHON_BIN}" -u "${SCRIPT_PATH}"
  --results_root "${RESULTS_ROOT}"
  --run_basename "${RUN_BASENAME}"
  --a_sources "${A_SOURCES}"
  --lambda_values "${LAMBDA_VALUES}"
  --mask_fracs "${MASK_FRACS}"
  --output_run_name "${AGG_RUN_NAME}"
)

echo "============================================================"
echo "RRR Find-Another Sweep Aggregation"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "[done] outputs:"
echo "  ${RESULTS_ROOT}/${AGG_RUN_NAME}/aggregate_summary.json"
echo "  ${RESULTS_ROOT}/${AGG_RUN_NAME}/config_manifest.csv"
echo "  ${RESULTS_ROOT}/${AGG_RUN_NAME}/best_iteration_per_config.csv"

