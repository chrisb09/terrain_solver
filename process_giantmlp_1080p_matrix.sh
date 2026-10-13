#!/usr/bin/env bash
# Post-process Giant-MLP 1080p 3-repeat campaigns for single-node and het surrogate.
set -euo pipefail
cd "$(dirname "$0")"

JOBIDS="results/aix_layout_matrix/3r_giantmlp_1080p_jobids.txt"

for cfg in 24c1g_giantmlp_1080p 96c4g_giantmlp_1080p; do
  echo "=== Processing single-node ${cfg} ==="
  ./process_5r_config.sh "${cfg}" "${cfg}" single "${JOBIDS}"
done

for cfg in 24p24c1g_giantmlp_1080p 96p96c4g_giantmlp_1080p; do
  echo "=== Processing het surrogate ${cfg} ==="
  ./process_5r_config.sh "${cfg}" "${cfg}" het "${JOBIDS}"
done

echo "All 4 Giant-MLP 1080p configurations processed."
