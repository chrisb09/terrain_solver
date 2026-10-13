#!/usr/bin/env bash
# Post-process one 5-repeat scaling config: split wrapper logs, aggregate the
# 500-sample summary, comparison figure, per-run node-variance table + figure.
#
#   ./process_5r_config.sh <cfg> <config-label> <single|het> [jobids-file]
#
# Assumes the jobids file exists (default results/aix_layout_matrix/5r_scaling_jobids.txt).
set -euo pipefail
cd "$(dirname "$0")"

CFG="${1:?cfg}"
LABEL="${2:?label}"
SERIES="${3:?series (single|het)}"
RES="results/aix_layout_matrix/${CFG}"
JOBIDS="${4:-results/aix_layout_matrix/5r_scaling_jobids.txt}"
mkdir -p "${RES}/logs"

LOGS=()
while read -r ser cfg _r jid; do
  [[ "${cfg}" == "${CFG}" ]] || continue
  if [[ "${ser}" == "single" ]]; then
    f="logs/mini_app_output_5r_${jid}.txt"
  else
    f="logs/mini_app_output_5r_het_${jid}.txt"
  fi
  [[ -f "${f}" ]] && LOGS+=("${f}")
done < "${JOBIDS}"

if [[ "${#LOGS[@]}" -eq 0 ]]; then
  echo "no wrapper logs found for ${CFG}" >&2
  exit 1
fi

echo "splitting ${#LOGS[@]} wrapper logs for ${CFG}"
python3 split_wrapper_logs.py --out-dir logs "${LOGS[@]}" >/dev/null
cp -f "${LOGS[@]}" "${RES}/logs/"

echo "aggregating 500 samples"
python3 analyze_matrix_results.py --config-label "${LABEL}" --logs-dir logs \
  --runs-csv ${RES}/runs_5r_${CFG}_r*.csv \
  --out-csv "${RES}/aix_layout_matrix_${CFG}_summary.csv"

python3 plot_layout_matrix.py --config-label "${LABEL}" --logs-dir logs \
  --runs-csv ${RES}/runs_5r_${CFG}_r*.csv \
  --out "${RES}/aix_layout_matrix_${CFG}_comparison.png"

python3 node_variance_report.py --config "${CFG}" --logs-dir logs \
  --manifest ${RES}/runs_5r_${CFG}_r*.csv \
  --out-csv "${RES}/aix_5r_per_run_medians.csv" >/dev/null

python3 plot_per_run_variance.py --config-label "${LABEL}" \
  --per-run-csv "${RES}/aix_5r_per_run_medians.csv" \
  --out "${RES}/aix_5r_per_run_variance.png"

echo "processed ${CFG} -> ${RES}"
