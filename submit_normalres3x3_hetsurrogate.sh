#!/usr/bin/env bash
# Submit the normal-resolution full-3x3 HET SURROGATE campaign under rwth0792.
# Both het groups are c23g nodes; Node 0 has GPUs hidden via CUDA_VISIBLE_DEVICES=NoDevFiles
# in 5r_solver_wrapper.sh to act as a CPU-only solver node.
# Compares directly against the original c23mm + c23g runs.
#
#   ./submit_normalres3x3_hetsurrogate.sh [N]   # N repeats (default 5)
set -euo pipefail
cd "$(dirname "$0")"

N="${1:-5}"
JOBIDS_FILE="results/aix_layout_matrix/5r_normalres_hetsurrogate_jobids.txt"
mkdir -p "$(dirname "${JOBIDS_FILE}")"
: > "${JOBIDS_FILE}"

chmod +x matrix_normalres3x3_hetsurrogate_template.sbatch 5r_driver_hetsurrogate.zsh 5r_solver_wrapper.sh

COMMON_CONFIG="TARGET_WIDTH_ENV=3840,TARGET_HEIGHT_ENV=2160,ML_BATCH_SIZE_ENV=20000000,MODEL_NAME_ENV=watercnn"

submit_het() {
  # cfg mm g gpus time
  local cfg="$1" mm="$2" g="$3" gpus="$4" walltime="$5"
  local out="matrix_5r_${cfg}.sbatch"
  sed -e "s/__CFG__/${cfg}/g" -e "s/__ACCOUNT__/rwth0792/g" -e "s/__TIME__/${walltime}/g" \
      -e "s/__HET_MM__/${mm}/g" -e "s/__HET_G__/${g}/g" -e "s/__GPUS__/${gpus}/g" \
      matrix_normalres3x3_hetsurrogate_template.sbatch > "${out}"
  chmod +x "${out}"
  local k jid
  for k in $(seq 1 "${N}"); do
    jid=$(sbatch --parsable --export="ALL,CFG_ENV=${cfg},REPEAT_ENV=${k},HET_G_SIDE_TASKS_ENV=${g},GPUS_PER_NODE_ENV=${gpus},${COMMON_CONFIG}" "${out}")
    echo "hetsurrogate ${cfg} r${k}: ${jid}"
    echo "het ${cfg} r${k} ${jid}" >> "${JOBIDS_FILE}"
  done
}

echo "=== Normal-res full 3x3 HET SURROGATE (rwth0792, c23g+c23g) ==="
submit_het 24p24c1g_surrogate 24 24 1 00:35:00
submit_het 48p48c2g_surrogate 48 48 2 00:30:00
submit_het 72p72c3g_surrogate 72 72 3 00:25:00
submit_het 96p96c4g_surrogate 96 96 4 00:25:00

echo "job ids recorded in ${JOBIDS_FILE}"
