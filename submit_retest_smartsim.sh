#!/usr/bin/env bash
# Submit the SmartSim-provider portion of the re-test.
#   models:  watercnn8k (7680x4320), giantmlp288 (480x288)
#   configs: 24c1g, 96c4g, 24p24c1g, 96p96c4g
#   cases:   shared_c0 (all), pergpu_c0 (4-GPU configs only)
#   repeats: 3
# Reuses the AIx retest sbatch allocations, with RUNNER_ENV selecting the
# SmartSim runner.
set -euo pipefail
cd "$(dirname "$0")"

N="${1:-3}"
RUNNER="/hpcwork/ro092286/smartsim/mini_app/retest_smartsim_cells.zsh"
JOBIDS_FILE="results/retest_smartsim/jobids.txt"
mkdir -p results/retest_smartsim
: > "${JOBIDS_FILE}"

submit_one() {
  local cfg="$1" model="$2" repeat="$3"
  local script="retest_aix_${cfg}.sbatch"
  local extra=""
  case "${cfg}" in
    24c1g)    extra="" ;;
    96c4g)    extra="" ;;
    24p24c1g) extra=",HET_G_SIDE_TASKS_ENV=24,GPUS_PER_NODE_ENV=1" ;;
    96p96c4g) extra=",HET_G_SIDE_TASKS_ENV=96,GPUS_PER_NODE_ENV=4" ;;
  esac
  local jid
  jid=$(sbatch --parsable \
      --export="ALL,RUNNER_ENV=${RUNNER},MODEL_ENV=${model},REPEAT_ENV=${repeat}${extra}" \
      "${script}")
  echo "smartsim ${model} ${cfg} r${repeat}: ${jid}"
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
