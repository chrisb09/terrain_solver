#!/usr/bin/env bash
# Submit the super-high-res AIx P2P-vs-collective comparison campaign.
#   96c4g_superres: 96 ranks / 12x8 grid / 4 GPUs  (per-rank payload == 24c1g normal)
#   24c1g_superres: 24 ranks /  6x4 grid / 1 GPU   (4x per-rank payload)
# Target 7680x4320, WaterCNN, batch 20M, `numa` layout only, comms
# coll/p2p1/p2p24, 5 repeats, timelines off.
#
#   ./submit_superres_campaign.sh [N]     # N repeats (default 5)
set -euo pipefail
cd "$(dirname "$0")"

N="${1:-5}"
JOBIDS_FILE="results/aix_layout_matrix/5r_superres_jobids.txt"
mkdir -p "$(dirname "${JOBIDS_FILE}")"
: > "${JOBIDS_FILE}"

chmod +x matrix_superres_template.sbatch 5r_driver_single.zsh

COMMON_CONFIG="TARGET_WIDTH_ENV=7680,TARGET_HEIGHT_ENV=4320,ML_BATCH_SIZE_ENV=20000000,MODEL_NAME_ENV=watercnn"

submit_superres() {
  # cfg account time ntasks gpus gridx gridz
  local cfg="$1" account="$2" time="$3" ntasks="$4" gpus="$5" gx="$6" gz="$7"
  local out="matrix_5r_${cfg}.sbatch"
  sed -e "s/__CFG__/${cfg}/g" -e "s/__ACCOUNT__/${account}/g" -e "s/__TIME__/${time}/g" \
      -e "s/__NTASKS__/${ntasks}/g" -e "s/__GPUS__/${gpus}/g" \
      matrix_superres_template.sbatch > "${out}"
  chmod +x "${out}"
  local k jid
  for k in $(seq 1 "${N}"); do
    jid=$(sbatch --parsable --export="ALL,CFG_ENV=${cfg},REPEAT_ENV=${k},MPI_RANKS_ENV=${ntasks},RANK_GRID_X_ENV=${gx},RANK_GRID_Z_ENV=${gz},GPUS_PER_NODE_ENV=${gpus},${COMMON_CONFIG}" "${out}")
    echo "superres ${cfg} r${k}: ${jid}"
    echo "single ${cfg} r${k} ${jid}" >> "${JOBIDS_FILE}"
  done
}

echo "=== Super-high-res comparison (7680x4320, numa, coll/p2p1/p2p24) ==="
submit_superres 96c4g_superres rwth0792 01:00:00 96 4 12 8
submit_superres 24c1g_superres rwth0792 01:00:00 24 1 6  4

echo "job ids recorded in ${JOBIDS_FILE}"
