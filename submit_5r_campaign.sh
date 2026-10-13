#!/usr/bin/env bash
# Submit the 5-repeat AIx layout-matrix campaign.
#
#   ./submit_5r_campaign.sh            # 5 single + 5 het (default)
#   ./submit_5r_campaign.sh all 1      # 1 single + 1 het (quick validation)
#   ./submit_5r_campaign.sh single 5   # 5 single only
#   ./submit_5r_campaign.sh het 5      # 5 het only
#
# Jobs are independent (no chain dependencies) so they queue/backfill freely.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-all}"
N="${2:-5}"

chmod +x matrix_5r_single.sbatch matrix_5r_het.sbatch 5r_driver_single.zsh 5r_driver_het.zsh

JOBIDS_FILE="results/aix_layout_matrix/5r_jobids.txt"
mkdir -p "$(dirname "${JOBIDS_FILE}")"
: > "${JOBIDS_FILE}"

submit_single() {
  local k jid
  for k in $(seq 1 "${N}"); do
    jid=$(sbatch --parsable --export=ALL,REPEAT_ENV="${k}" matrix_5r_single.sbatch)
    echo "matrix5r_single r${k}: ${jid}"
    echo "single r${k} ${jid}" >> "${JOBIDS_FILE}"
  done
}

submit_het() {
  local k jid
  for k in $(seq 1 "${N}"); do
    jid=$(sbatch --parsable --export=ALL,REPEAT_ENV="${k}" matrix_5r_het.sbatch)
    echo "matrix5r_het r${k}: ${jid}"
    echo "het r${k} ${jid}" >> "${JOBIDS_FILE}"
  done
}

case "${MODE}" in
  all)    submit_single; submit_het ;;
  single) submit_single ;;
  het)    submit_het ;;
  *) echo "usage: $0 [all|single|het] [N]" >&2; exit 2 ;;
esac

echo "job ids recorded in ${JOBIDS_FILE}"
