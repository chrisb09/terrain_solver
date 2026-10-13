#!/usr/bin/env bash
# Submit the 3-repeat Giant-MLP 1080p (1920x1080) matrix campaign under rwth0792.
# 4 configurations: 24c1g, 96c4g (single node) and 24p24c1g, 96p96c4g (het surrogate).
# 12 steps (1 warmup + 5 steady ML steps), batch cap 500k.
set -euo pipefail
cd "$(dirname "$0")"

JOBIDS_FILE="results/aix_layout_matrix/3r_giantmlp_1080p_jobids.txt"
mkdir -p "$(dirname "${JOBIDS_FILE}")"
: > "${JOBIDS_FILE}"

# 1. Single node configurations
submit_single() {
  local cfg="$1" ranks="$2" gpus="$3" rx="$4" rz="$5" mem="$6" walltime="$7"
  local sbatch_file="matrix_3r_${cfg}.sbatch"
  sed -e "s/__CFG__/${cfg}/g" \
      -e "s/__RANKS__/${ranks}/g" \
      -e "s/__GPUS__/${gpus}/g" \
      -e "s/__RX__/${rx}/g" \
      -e "s/__RZ__/${rz}/g" \
      -e "s/__MEM__/${mem}/g" \
      -e "s/__TIME__/${walltime}/g" \
      matrix_3r_giantmlp_1080p_single_template.sbatch > "${sbatch_file}"
  chmod +x "${sbatch_file}"

  for k in $(seq 1 3); do
    jid=$(sbatch --parsable --export="ALL,CFG_ENV=${cfg},REPEAT_ENV=${k}" "${sbatch_file}")
    echo "single ${cfg} r${k}: ${jid}"
    echo "single ${cfg} r${k} ${jid}" >> "${JOBIDS_FILE}"
  done
}

# 2. Het surrogate configurations
submit_het() {
  local cfg="$1" mm="$2" g="$3" gpus="$4" walltime="$5"
  local sbatch_file="matrix_3r_${cfg}.sbatch"
  sed -e "s/__CFG__/${cfg}/g" \
      -e "s/__HET_MM__/${mm}/g" \
      -e "s/__HET_G__/${g}/g" \
      -e "s/__GPUS__/${gpus}/g" \
      -e "s/__TIME__/${walltime}/g" \
      matrix_3r_giantmlp_1080p_hetsurrogate_template.sbatch > "${sbatch_file}"
  chmod +x "${sbatch_file}"

  for k in $(seq 1 3); do
    jid=$(sbatch --parsable --export="ALL,CFG_ENV=${cfg},REPEAT_ENV=${k},HET_G_SIDE_TASKS_ENV=${g},GPUS_PER_NODE_ENV=${gpus}" "${sbatch_file}")
    echo "het ${cfg} r${k}: ${jid}"
    echo "het ${cfg} r${k} ${jid}" >> "${JOBIDS_FILE}"
  done
}

echo "=== Submitting Giant-MLP 1080p (1920x1080) 3-repeat suite (c23g, rwth0792) ==="
submit_single 24c1g_giantmlp_1080p 24 1 6 4 120G 01:00:00
submit_single 96c4g_giantmlp_1080p 96 4 12 8 480G 00:30:00
submit_het    24p24c1g_giantmlp_1080p 24 24 1 01:00:00
submit_het    96p96c4g_giantmlp_1080p 96 96 4 00:30:00

echo "All job IDs recorded in ${JOBIDS_FILE}"
