#!/usr/bin/env bash
# Submit the AIx portion of the synchronized-timing re-test.
#   models:   watercnn8k (7680x4320), giantmlp288 (480x288)
#   configs:  24c1g, 96c4g (single node), 24p24c1g, 96p96c4g (het surrogate)
#   cells:    full 3x3 (default/affine/numa x coll/p2p1/p2p24), sequential
#   repeats:  3
# Output: results/retest_aix/<model>/<cfg>/, logs/retest_aix_*.txt
set -euo pipefail
cd "$(dirname "$0")"

N="${1:-3}"
JOBIDS_FILE="results/retest_aix/jobids.txt"
mkdir -p results/retest_aix
: > "${JOBIDS_FILE}"

submit_one() {
  local cfg="$1" model="$2" repeat="$3"
  local script="retest_aix_${cfg}.sbatch"
  local export_extra=""
  case "${cfg}" in
    24c1g)        export_extra="" ;;
    96c4g)        export_extra="" ;;
    24p24c1g)     export_extra=",HET_G_SIDE_TASKS_ENV=24,GPUS_PER_NODE_ENV=1" ;;
    96p96c4g)     export_extra=",HET_G_SIDE_TASKS_ENV=96,GPUS_PER_NODE_ENV=4" ;;
  esac
  local jid
  jid=$(sbatch --parsable --export="ALL,MODEL_ENV=${model},REPEAT_ENV=${repeat}${export_extra}" "${script}")
  echo "${model} ${cfg} r${repeat}: ${jid}"
  echo "${model},${cfg},${repeat},${jid}" >> "${JOBIDS_FILE}"
}

for model in watercnn8k giantmlp288; do
  for cfg in 24c1g 96c4g 24p24c1g 96p96c4g; do
    for k in $(seq 1 "${N}"); do
      submit_one "${cfg}" "${model}" "${k}"
    done
  done
done

echo "All job IDs in ${JOBIDS_FILE}"
