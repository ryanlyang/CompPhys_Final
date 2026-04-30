#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/atlas/CompPhys_FinalProject}"
SEEDS="${SEEDS:-41,52,63,74,85}"
RUN_BASENAME="${RUN_BASENAME:-prelim_reimpl_cluster}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/restart_studies/results}"
AGG_DIR_NAME="${AGG_DIR_NAME:-${RUN_BASENAME}_aggregate_5seeds}"

ARRAY_SCRIPT="${SCRIPT_DIR}/sbatch_reimplement_preliminary_studies_5seeds_array.sh"
AGG_SCRIPT="${SCRIPT_DIR}/sbatch_aggregate_preliminary_studies_5seeds.sh"

if [[ ! -f "${ARRAY_SCRIPT}" ]]; then
  echo "ERROR: missing ${ARRAY_SCRIPT}" >&2
  exit 2
fi
if [[ ! -f "${AGG_SCRIPT}" ]]; then
  echo "ERROR: missing ${AGG_SCRIPT}" >&2
  exit 2
fi

echo "[submit] project root hint (local): ${PROJECT_ROOT_DEFAULT}"
echo "[submit] cluster PROJECT_ROOT: ${PROJECT_ROOT}"
echo "[submit] run basename: ${RUN_BASENAME}"
echo "[submit] seeds: ${SEEDS}"
echo "[submit] results root: ${RESULTS_ROOT}"

SUBMIT_OUT="$(SEEDS="${SEEDS}" sbatch \
  --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",RUN_BASENAME="${RUN_BASENAME}",OUTPUT_ROOT="${RESULTS_ROOT}" \
  "${ARRAY_SCRIPT}")"
ARRAY_JOB_ID="$(echo "${SUBMIT_OUT}" | grep -Eo '[0-9]+' | tail -1)"

if [[ -z "${ARRAY_JOB_ID}" ]]; then
  echo "ERROR: failed to parse array job id from: ${SUBMIT_OUT}" >&2
  exit 2
fi

echo "[submit] array submitted: ${SUBMIT_OUT}"

AGG_OUT="$(SEEDS="${SEEDS}" sbatch \
  --dependency=afterok:${ARRAY_JOB_ID} \
  --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",RESULTS_ROOT="${RESULTS_ROOT}",RUN_BASENAME="${RUN_BASENAME}",AGG_DIR_NAME="${AGG_DIR_NAME}" \
  "${AGG_SCRIPT}")"
AGG_JOB_ID="$(echo "${AGG_OUT}" | grep -Eo '[0-9]+' | tail -1)"

if [[ -z "${AGG_JOB_ID}" ]]; then
  echo "ERROR: failed to parse aggregate job id from: ${AGG_OUT}" >&2
  exit 2
fi

echo "[submit] aggregate submitted: ${AGG_OUT}"
echo "[submit] dependency: aggregate starts after array job ${ARRAY_JOB_ID} succeeds"
echo "[submit] expected aggregate dir: ${RESULTS_ROOT}/${AGG_DIR_NAME}"
