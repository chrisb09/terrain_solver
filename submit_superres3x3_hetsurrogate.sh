#!/usr/bin/env bash
# Submit the super-high-res full-3x3 HET SURROGATE campaign under rwth0792.
# Both het groups are c23g nodes; the CPU-side group requests gpu:1 (required by
# Slurm) and the surrogate driver hides it from AIx via CUDA_VISIBLE_DEVICES=
# NoDevFiles. Topology is identical to c23mm, so this stands in while thes2181
# (the only c23mm account) is out of CPU quota.
#
#   ./submit_superres3x3_hetsurrogate.sh [N]   # N repeats (default 5)
set -euo pipefail
cd "$(dirname "$0")"

N="${1:-5}"
JOBIDS_FILE="results/aix_layout_matrix/5r_superres3x3_het_surrogate_jobids.txt"
mkdir -p "$(dirname "${JOBIDS_FILE}")"
: > "${JOBIDS_FILE}"

chmod +x matrix_superres3x3_hetsurrogate_template.sbatch 5r_driver_hetsurrogate.zsh 5r_solver_wrapper.sh

COMMON_CONFIG="TARGET_WIDTH_ENV=7680,TARGET_HEIGHT_ENV=4320,ML_BATCH_SIZE_ENV=20000000,MODEL_NAME_ENV=watercnn"

submit_het() {
  # cfg mm g gpus
  local cfg="$1" mm="$2" g="$3" gpus="$4"
  local out="matrix_5r_${cfg}.sbatch"
  sed -e "s/__CFG__/${cfg}/g" -e "s/__ACCOUNT__/rwth0792/g" -e "s/__TIME__/01:00:00/g" \
      -e "s/__HET_MM__/${mm}/g" -e "s/__HET_G__/${g}/g" -e "s/__GPUS__/${gpus}/g" \
      matrix_superres3x3_hetsurrogate_template.sbatch > "${out}"
  chmod +x "${out}"
  local k jid
  for k in $(seq 1 "${N}"); do
    jid=$(sbatch --parsable --export="ALL,CFG_ENV=${cfg},REPEAT_ENV=${k},HET_G_SIDE_TASKS_ENV=${g},GPUS_PER_NODE_ENV=${gpus},${COMMON_CONFIG}" "${out}")
    echo "hetsurrogate ${cfg} r${k}: ${jid}"
    echo "het ${cfg} r${k} ${jid}" >> "${JOBIDS_FILE}"
  done
}

echo "=== Super-high-res full 3x3 HET SURROGATE (rwth0792, c23g+c23g) ==="
submit_het 24p24c1g_superres 24 24 1
submit_het 48p48c2g_superres 48 48 2
submit_het 72p72c3g_superres 72 72 3
submit_het 96p96c4g_superres 96 96 4

echo "job ids recorded in ${JOBIDS_FILE}"
