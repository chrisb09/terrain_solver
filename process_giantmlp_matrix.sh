#!/usr/bin/env bash
# Post-process Giant-MLP 3-layout campaigns for 96c4g and 24c1g.
# Generates summary CSVs, bar+box plots (PGF/PDF/PNG), and per-run variance plots
# in results/aix_layout_matrix/<cfg>/ and links figures to thesis graphics.
set -euo pipefail
cd "$(dirname "$0")"

JOBIDS="results/aix_layout_matrix/5r_giantmlp_3layout_jobids.txt"

for cfg in 96c4g_giantmlp 24c1g_giantmlp; do
  echo "=== Processing ${cfg} ==="
  ./process_5r_config.sh "${cfg}" "${cfg}" single "${JOBIDS}"
  
  # Copy figures to thesis graphics if present
  THESIS_GRAPHICS="/home/ro092286/Master-Thesis/gitlab/thesis/graphics/giant_mlp_scaling"
  mkdir -p "${THESIS_GRAPHICS}"
  cp -f "results/aix_layout_matrix/${cfg}/aix_layout_matrix_${cfg}_comparison".* "${THESIS_GRAPHICS}/" 2>/dev/null || true
  cp -f "results/aix_layout_matrix/${cfg}/aix_5r_per_run_variance".* "${THESIS_GRAPHICS}/${cfg}_per_run_variance".* 2>/dev/null || true
done

echo "Giant-MLP 3-layout processing complete."
