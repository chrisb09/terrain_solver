#!/usr/bin/env bash
# Submit the streamlined 5-repeat Giant-MLP campaign (24c1g and 96c4g) under rwth0792.
# 12 steps (1 warmup + 5 steady ML steps), batch cap 500k, resolution 7680x4320.
set -euo pipefail
cd "$(dirname "$0")"

JOBIDS_FILE="results/giant_mlp_scaling/5r_giantmlp_jobids.txt"
mkdir -p "$(dirname "${JOBIDS_FILE}")"
: > "${JOBIDS_FILE}"

submit_cfg() {
  local cfg="$1" ranks="$2" gpus="$3" rx="$4" rz="$5" mem="$6" walltime="$7"
  local sbatch_file="matrix_5r_giantmlp_${cfg}.sbatch"
  sed -e "s/__CFG__/${cfg}/g" \
      -e "s/__RANKS__/${ranks}/g" \
      -e "s/__GPUS__/${gpus}/g" \
      -e "s/__RX__/${rx}/g" \
      -e "s/__RZ__/${rz}/g" \
      -e "s/__MEM__/${mem}/g" \
      -e "s/__TIME__/${walltime}/g" \
      matrix_5r_giantmlp_superres_template.sbatch > "${sbatch_file}"
  chmod +x "${sbatch_file}"

  for k in $(seq 1 5); do
    jid=$(sbatch --parsable --export="ALL,CFG_ENV=${cfg},REPEAT_ENV=${k}" "${sbatch_file}")
    echo "giantmlp ${cfg} r${k}: ${jid}"
    echo "${cfg} r${k} ${jid}" >> "${JOBIDS_FILE}"
  done
}

echo "=== Submitting Giant-MLP 5-repeat scaling suite ==="
submit_cfg 24c1g 24 1 6 4 120G 01:20:00
submit_cfg 96c4g 96 4 12 8 480G 00:30:00

echo "Job IDs recorded in ${JOBIDS_FILE}"
