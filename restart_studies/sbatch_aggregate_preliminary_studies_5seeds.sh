#!/usr/bin/env bash
#SBATCH --job-name=prelimAgg5
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=restart_studies/logs/prelim_aggregate5_%j.out
#SBATCH --error=restart_studies/logs/prelim_aggregate5_%j.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/atlas/CompPhys_FinalProject}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/restart_studies/results}"
RUN_BASENAME="${RUN_BASENAME:-prelim_reimpl_cluster}"
SEEDS="${SEEDS:-41,52,63,74,85}"
AGG_DIR_NAME="${AGG_DIR_NAME:-${RUN_BASENAME}_aggregate_5seeds}"

SCRIPT_CANDIDATES=(
  "${WORKDIR}/restart_studies/aggregate_preliminary_studies_multi_seed.py"
  "${PROJECT_ROOT}/restart_studies/aggregate_preliminary_studies_multi_seed.py"
)
SCRIPT_PATH=""
for c in "${SCRIPT_CANDIDATES[@]}"; do
  if [[ -f "${c}" ]]; then
    SCRIPT_PATH="${c}"
    break
  fi
done

if [[ -z "${SCRIPT_PATH}" ]]; then
  echo "ERROR: aggregate_preliminary_studies_multi_seed.py not found." >&2
  echo "Checked: ${SCRIPT_CANDIDATES[*]}" >&2
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

mkdir -p "${WORKDIR}/restart_studies/logs"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CMD=(
  "${PYTHON_BIN}" -u "${SCRIPT_PATH}"
  --results_root "${RESULTS_ROOT}"
  --run_basename "${RUN_BASENAME}"
  --seeds "${SEEDS}"
  --output_dir "${RESULTS_ROOT}/${AGG_DIR_NAME}"
)

echo "============================================================"
echo "Aggregate Preliminary Reimplementation (5 seeds)"
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Results root: ${RESULTS_ROOT}"
echo "Run basename: ${RUN_BASENAME}"
echo "Seeds: ${SEEDS}"
echo "Output dir: ${RESULTS_ROOT}/${AGG_DIR_NAME}"
echo "============================================================"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
