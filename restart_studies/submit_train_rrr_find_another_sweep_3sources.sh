#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ROOT="${PROJECT_ROOT:-/home/ryreu/atlas/CompPhys_FinalProject}"
SEED="${SEED:-52}"
RUN_BASENAME="${RUN_BASENAME:-rrr_findanother_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/restart_studies/results}"
AGG_RUN_NAME="${AGG_RUN_NAME:-${RUN_BASENAME}_aggregate_48cfg}"
SUBMIT_AGG="${SUBMIT_AGG:-1}"

LAMBDA_VALUES="${LAMBDA_VALUES:-1,10,100,1000}"
MASK_FRACS="${MASK_FRACS:-0.05,0.10,0.20,0.30}"
A_SOURCES="${A_SOURCES:-input_grad,integrated_gradients,smoothgrad}"

ARRAY_SCRIPT="${SCRIPT_DIR}/sbatch_train_rrr_find_another_sweep_array.sh"
AGG_SCRIPT="${SCRIPT_DIR}/sbatch_aggregate_rrr_find_another_sweep.sh"
if [[ ! -f "${ARRAY_SCRIPT}" ]]; then
  echo "ERROR: missing ${ARRAY_SCRIPT}" >&2
  exit 2
fi
if [[ "${SUBMIT_AGG}" == "1" && ! -f "${AGG_SCRIPT}" ]]; then
  echo "ERROR: missing ${AGG_SCRIPT}" >&2
  exit 2
fi

echo "[submit] project root hint (local): ${PROJECT_ROOT_DEFAULT}"
echo "[submit] cluster PROJECT_ROOT: ${PROJECT_ROOT}"
echo "[submit] run basename: ${RUN_BASENAME}"
echo "[submit] seed: ${SEED}"
echo "[submit] output root: ${OUTPUT_ROOT}"
echo "[submit] lambda grid: ${LAMBDA_VALUES}"
echo "[submit] mask grid: ${MASK_FRACS}"
echo "[submit] a_sources: ${A_SOURCES}"
echo "[submit] submit aggregate job: ${SUBMIT_AGG}"

A_SOURCES_CSV="$(echo "${A_SOURCES}" | tr ' ' ',' | tr -s ',' | sed 's/^,//; s/,$//')"
IFS=',' read -r -a SRC_LIST <<< "${A_SOURCES_CSV}"

if [[ "${#SRC_LIST[@]}" -ne 3 ]]; then
  echo "ERROR: expected exactly 3 A sources, got ${#SRC_LIST[@]} from '${A_SOURCES}'" >&2
  exit 2
fi

declare -a JOB_IDS=()
for src in "${SRC_LIST[@]}"; do
  SUBMIT_OUT="$(
    A_SOURCE="${src}" \
    LAMBDA_VALUES="${LAMBDA_VALUES}" \
    MASK_FRACS="${MASK_FRACS}" \
    sbatch \
      --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",RUN_BASENAME="${RUN_BASENAME}",SEED="${SEED}" \
      "${ARRAY_SCRIPT}"
  )"
  JOB_ID="$(echo "${SUBMIT_OUT}" | grep -Eo '[0-9]+' | tail -1)"
  if [[ -z "${JOB_ID}" ]]; then
    echo "ERROR: failed to parse job id from: ${SUBMIT_OUT}" >&2
    exit 2
  fi
  JOB_IDS+=("${JOB_ID}")
  echo "[submit] ${src}: ${SUBMIT_OUT}"
done

echo "[submit] queued 3 array jobs: ${JOB_IDS[*]}"
echo "[submit] each array has 16 configs; expected total configs: 48"

if [[ "${SUBMIT_AGG}" == "1" ]]; then
  DEP_STR="$(IFS=:; echo "${JOB_IDS[*]}")"
  AGG_OUT="$(
    A_SOURCES="${A_SOURCES}" \
    LAMBDA_VALUES="${LAMBDA_VALUES}" \
    MASK_FRACS="${MASK_FRACS}" \
    sbatch \
      --dependency=afterok:${DEP_STR} \
      --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",RESULTS_ROOT="${OUTPUT_ROOT}",RUN_BASENAME="${RUN_BASENAME}",AGG_RUN_NAME="${AGG_RUN_NAME}" \
      "${AGG_SCRIPT}"
  )"
  AGG_JOB_ID="$(echo "${AGG_OUT}" | grep -Eo '[0-9]+' | tail -1)"
  if [[ -z "${AGG_JOB_ID}" ]]; then
    echo "ERROR: failed to parse aggregate job id from: ${AGG_OUT}" >&2
    exit 2
  fi
  echo "[submit] aggregate: ${AGG_OUT}"
  echo "[submit] aggregate dependency: afterok:${DEP_STR}"
  echo "[submit] aggregate output dir: ${OUTPUT_ROOT}/${AGG_RUN_NAME}"
fi
