#!/usr/bin/env bash
#SBATCH --job-name=aspen_schema
#SBATCH --partition=tier3
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=sbatch/slurm_logs/slurm_aspen_schema_%j.out
#SBATCH --error=sbatch/slurm_logs/slurm_aspen_schema_%j.err

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-atlas_kd}"
ASPEN_DIR="${ASPEN_DIR:-/home/ryreu/atlas/CompPhys_Final/data/AspenOpenJets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ryreu/atlas/CompPhys_Final/runs/aspen_schema_probe}"
RUN_TAG="${RUN_TAG:-aspen_schema_probe}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/slurm_logs}"
AUTO_INSTALL_DEPS="${AUTO_INSTALL_DEPS:-1}"
MIRROR_TO_SLURM_STDIO="${MIRROR_TO_SLURM_STDIO:-0}"
MAX_FILES="${MAX_FILES:-4}"
PREVIEW_ELEMENTS="${PREVIEW_ELEMENTS:-2048}"

set +u
source ~/.bashrc
set -u
conda activate "${CONDA_ENV}"
export PATH="${HOME}/.local/bin:${PATH}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_TAG}_job${JOB_ID}"
LOG_DIR="${LOG_ROOT}/${RUN_TAG}"
STDOUT_LOG="${LOG_DIR}/job${JOB_ID}.out"
STDERR_LOG="${LOG_DIR}/job${JOB_ID}.err"
mkdir -p "${RUN_DIR}" "${LOG_DIR}" "sbatch/slurm_logs"

if [[ "${MIRROR_TO_SLURM_STDIO}" == "1" ]]; then
  exec > >(tee -a "${STDOUT_LOG}") 2> >(tee -a "${STDERR_LOG}" >&2)
else
  exec > >(tee -a "${STDOUT_LOG}" >/dev/null) 2> >(tee -a "${STDERR_LOG}" >/dev/null)
fi

if [[ ! -d "${ASPEN_DIR}" ]]; then
  echo "ERROR: ASPEN_DIR does not exist: ${ASPEN_DIR}" >&2
  exit 1
fi

SCRIPT_PATH="inspect_aspen_openjets_schema.py"
if [[ ! -f "${SCRIPT_PATH}" ]]; then
  echo "ERROR: cannot find ${SCRIPT_PATH} in $(pwd)" >&2
  exit 1
fi

missing_modules=()
for mod in h5py numpy; do
  if ! python3 -c "import ${mod}" >/dev/null 2>&1; then
    missing_modules+=("${mod}")
  fi
done

if [[ "${#missing_modules[@]}" -gt 0 ]]; then
  if [[ "${AUTO_INSTALL_DEPS}" == "1" ]]; then
    echo "Installing missing deps: ${missing_modules[*]}"
    python3 -m pip install --upgrade pip
    python3 -m pip install --no-user h5py numpy
  else
    echo "ERROR: Missing dependencies: ${missing_modules[*]}" >&2
    echo "Set AUTO_INSTALL_DEPS=1 to install automatically." >&2
    exit 1
  fi
fi

CMD=(
  python3 "${SCRIPT_PATH}"
  --aspen-dir "${ASPEN_DIR}"
  --output-dir "${RUN_DIR}"
  --max-files "${MAX_FILES}"
  --preview-elements "${PREVIEW_ELEMENTS}"
)

echo "============================================================"
echo "AspenOpenJets schema probe"
echo "Run dir: ${RUN_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Stdout log: ${STDOUT_LOG}"
echo "Stderr log: ${STDERR_LOG}"
echo "Aspen dir: ${ASPEN_DIR}"
echo "Conda env: ${CONDA_ENV}"
echo "============================================================"
echo
printf ' %q' "${CMD[@]}"
echo
echo

"${CMD[@]}"

echo
echo "Done. Outputs:"
echo "  ${RUN_DIR}/aspen_schema_report.json"
echo "  ${RUN_DIR}/aspen_schema_report.txt"
echo "  ${RUN_DIR}/aspen_dataset_keys.csv"
echo "  ${STDOUT_LOG}"
echo "  ${STDERR_LOG}"
