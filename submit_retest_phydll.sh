#!/usr/bin/env bash
# Submit PhyDLL native C++ DL client retest.
#   models:  watercnn8k (7680x4320), giantmlp288 (480x288)
#   configs: 24c1g, 96c4g, 24p24c1g, 96p96c4g
#   repeats: 3
#
# Single-node configs reuse the AIx retest sbatch allocations.
# Het configs use dedicated phydll sbatch files (DL clients on GPU-side node).
set -euo pipefail
cd "$(dirname "$0")"

N="${1:-3}"
RUNNER="/hpcwork/ro092286/smartsim/mini_app/retest_phydll_cells.zsh"
JOBIDS_FILE="results/retest_phydll/jobids.txt"
mkdir -p results/retest_phydll
: > "${JOBIDS_FILE}"

submit_one() {
  local cfg="$1" model="$2" repeat="$3"
  local script extra
  case "${cfg}" in
    24c1g)
      script="retest_aix_24c1g.sbatch"
      extra="RANKS_ENV=24,GRID_X_ENV=6,GRID_Z_ENV=4,GPUS_ENV=1"
      ;;
    96c4g)
      script="retest_aix_96c4g.sbatch"
      extra="RANKS_ENV=96,GRID_X_ENV=12,GRID_Z_ENV=8,GPUS_ENV=4"
      ;;
    24p24c1g)
      script="retest_phydll_24p24c1g.sbatch"
      extra="HET_G_SIDE_TASKS_ENV=24,GPUS_PER_NODE_ENV=1"
      ;;
    96p96c4g)
      script="retest_phydll_96p96c4g.sbatch"
      extra="HET_G_SIDE_TASKS_ENV=96,GPUS_PER_NODE_ENV=4"
      ;;
  esac
  local jid
  jid=$(sbatch --parsable \
      --export="ALL,RUNNER_ENV=${RUNNER},MODEL_ENV=${model},REPEAT_ENV=${repeat},CFG_ENV=${cfg},${extra}" \
      "${script}")
  echo "phydll ${model} ${cfg} r${repeat}: ${jid}"
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
