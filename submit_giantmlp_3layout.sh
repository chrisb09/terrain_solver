#!/usr/bin/env bash
# Submit the 5-repeat 3-layout Giant-MLP campaign (default, affine, numa in collective mode).
# Target 7680x4320, batch cap 500k, steps 12 (1 warmup + 5 steady ML steps).
set -euo pipefail
cd "$(dirname "$0")"

JOBIDS_FILE="results/aix_layout_matrix/5r_giantmlp_3layout_jobids.txt"
mkdir -p "$(dirname "${JOBIDS_FILE}")"
: > "${JOBIDS_FILE}"

submit_cfg() {
  local cfg="$1" ranks="$2" gpus="$3" rx="$4" rz="$5" mem="$6" walltime="$7"
  local sbatch_file="matrix_5r_${cfg}.sbatch"
  sed -e "s/__CFG__/${cfg}/g" \
      -e "s/__RANKS__/${ranks}/g" \
      -e "s/__GPUS__/${gpus}/g" \
      -e "s/__RX__/${rx}/g" \
      -e "s/__RZ__/${rz}/g" \
      -e "s/__MEM__/${mem}/g" \
      -e "s/__TIME__/${walltime}/g" \
      matrix_5r_giantmlp_3layout_template.sbatch > "${sbatch_file}"
  chmod +x "${sbatch_file}"

  for k in $(seq 1 5); do
    jid=$(sbatch --parsable --export="ALL,CFG_ENV=${cfg},REPEAT_ENV=${k}" "${sbatch_file}")
    echo "giantmlp ${cfg} r${k}: ${jid}"
    echo "single ${cfg} r${k} ${jid}" >> "${JOBIDS_FILE}"
  done
}

echo "=== Submitting Giant-MLP 3-layout 5-repeat suite (c23g, rwth0792) ==="
submit_cfg 96c4g_giantmlp 96 4 12 8 480G 01:15:00
submit_cfg 24c1g_giantmlp 24 1 6 4 120G 03:45:00

echo "Job IDs recorded in ${JOBIDS_FILE}"
