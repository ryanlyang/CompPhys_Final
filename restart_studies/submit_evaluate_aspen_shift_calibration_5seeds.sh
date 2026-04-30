#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${WORKDIR}"

SCRIPT="restart_studies/sbatch_evaluate_aspen_shift_calibration_5seeds_array.sh"
if [[ ! -f "${SCRIPT}" ]]; then
  echo "ERROR: Missing ${SCRIPT}" >&2
  exit 2
fi

mkdir -p restart_studies/logs

echo "Submitting 5-seed Aspen shift calibration array job..."
JOBID="$(
  sbatch "${SCRIPT}" | awk '/Submitted batch job/ {print $4}'
)"

if [[ -z "${JOBID}" ]]; then
  echo "ERROR: Failed to parse submitted job id." >&2
  exit 2
fi

echo "Submitted job array: ${JOBID}"
echo "Monitor with:"
echo "  squeue -j ${JOBID}"
