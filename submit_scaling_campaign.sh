#!/usr/bin/env bash
# Submit the AIx core/GPU strong-scaling campaign (5 repeats x 3x3 cells).
#
#   ./submit_scaling_campaign.sh          # all configs, 5 repeats
#   ./submit_scaling_campaign.sh 1        # one repeat each (quick check)
#
# Single-node configs -> rwth0792 (c23g), het configs -> thes2181 (c23mm+c23g).
# No chain dependencies (independent wrappers).
set -euo pipefail
cd "$(dirname "$0")"

N="${1:-5}"
JOBIDS_FILE="results/aix_layout_matrix/5r_scaling_jobids.txt"
mkdir -p "$(dirname "${JOBIDS_FILE}")"
: > "${JOBIDS_FILE}"

chmod +x matrix_5r_scaling_template.sbatch matrix_5r_scaling_het_template.sbatch 5r_driver_single.zsh 5r_driver_het.zsh

COMMON_CONFIG="TARGET_WIDTH_ENV=3840,TARGET_HEIGHT_ENV=2160,ML_BATCH_SIZE_ENV=20000000,MODEL_NAME_ENV=watercnn"

submit_single() {
  # cfg account time ntasks gpus gridx gridz
  local cfg="$1" account="$2" time="$3" ntasks="$4" gpus="$5" gx="$6" gz="$7"
  local out="matrix_5r_${cfg}.sbatch"
  sed -e "s/__CFG__/${cfg}/g" -e "s/__ACCOUNT__/${account}/g" -e "s/__TIME__/${time}/g" \
      -e "s/__NTASKS__/${ntasks}/g" -e "s/__GPUS__/${gpus}/g" \
      matrix_5r_scaling_template.sbatch > "${out}"
  chmod +x "${out}"
  local k jid
  for k in $(seq 1 "${N}"); do
    jid=$(sbatch --parsable --export="ALL,CFG_ENV=${cfg},REPEAT_ENV=${k},MPI_RANKS_ENV=${ntasks},RANK_GRID_X_ENV=${gx},RANK_GRID_Z_ENV=${gz},GPUS_PER_NODE_ENV=${gpus},${COMMON_CONFIG}" "${out}")
    echo "single ${cfg} r${k}: ${jid}"
    echo "single ${cfg} r${k} ${jid}" >> "${JOBIDS_FILE}"
  done
}

submit_het() {
  # cfg account time mm_tasks g_tasks gpus
  local cfg="$1" account="$2" time="$3" mm="$4" g="$5" gpus="$6"
  local out="matrix_5r_${cfg}.sbatch"
  sed -e "s/__CFG__/${cfg}/g" -e "s/__ACCOUNT__/${account}/g" -e "s/__TIME__/${time}/g" \
      -e "s/__HET_MM__/${mm}/g" -e "s/__HET_G__/${g}/g" -e "s/__GPUS__/${gpus}/g" \
      matrix_5r_scaling_het_template.sbatch > "${out}"
  chmod +x "${out}"
  local k jid
  for k in $(seq 1 "${N}"); do
    jid=$(sbatch --parsable --export="ALL,CFG_ENV=${cfg},REPEAT_ENV=${k},HET_G_SIDE_TASKS_ENV=${g},GPUS_PER_NODE_ENV=${gpus},${COMMON_CONFIG}" "${out}")
    echo "het    ${cfg} r${k}: ${jid}"
    echo "het    ${cfg} r${k} ${jid}" >> "${JOBIDS_FILE}"
  done
}

echo "=== Single-node scaling (block: no; cyclic site default) ==="
submit_single 24c1g  rwth0792 00:45:00 24 1 6  4
submit_single 48c2g  rwth0792 00:45:00 48 2 12 4
submit_single 72c3g  rwth0792 00:45:00 72 3 12 6

echo "=== Het scaling ==="
submit_het 24p24c1g thes2181 01:00:00 24 24 1
submit_het 48p48c2g thes2181 01:00:00 48 48 2
submit_het 72p72c3g thes2181 01:00:00 72 72 3

echo "job ids recorded in ${JOBIDS_FILE}"
