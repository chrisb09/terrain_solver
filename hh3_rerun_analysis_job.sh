#!/bin/zsh
# Re-analyze the 6 re-run H-class affine configs (s2/s4/s5 x p2p/p2p_fullcredits),
# regenerate their p2p_inference_metrics.csv, then rebuild the overview tables
# and scaling plots. Runs in a devel job to avoid login-node CPU limits.
#SBATCH --job-name=hh3_rerun_analysis
#SBATCH --partition=devel
#SBATCH --account=default
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/hh3_rerun_analysis_%j.txt

set -uo pipefail
cd /hpcwork/ro092286/smartsim/mini_app
source /hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/activate
P2P_PHASES=/hpcwork/ro092286/smartsim/CPP-ML-Interface/scripts/analyze_p2p_inference_phases.py
CPP_ML=/hpcwork/ro092286/smartsim/CPP-ML-Interface

echo "=== HH3 rerun analysis start $(date) ==="

# Force a clean recompute of the overview metric cache.
rm -f results/hethires3/overview_metric_cache.json

CONFIGS=(s2_aix_p2p_affine s4_aix_p2p_affine s5_aix_p2p_affine \
         s2_aix_p2p_fullcredits_affine s4_aix_p2p_fullcredits_affine s5_aix_p2p_fullcredits_affine)

for cfg in "${CONFIGS[@]}"; do
  echo "--- per-config analysis: ${cfg} ---"
  SKIP_CROSS_ENV=1 P2P_GANTT_ENV=0 FORCE_ENV=1 ./analyze_scorep_suites.sh hethires3 "${cfg}"
  echo "rc=$?"
done

# Regenerate p2p_inference_metrics.csv for the re-run scenarios s2/s4/s5 by
# driving analyze_p2p_inference_phases.py directly over each scenario's
# (default/affine/numa) timelines and its collective trace.
OC=/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode/scorep-8.4-papi72-install/bin/otf2-print
for scen in 2 4 5; do
  echo "--- p2p phases for scenario s${scen} ---"
  tl_args=()
  for lib in aix_p2p aix_p2p_fullcredits; do
    for layout in default affine numa; do
      name="s${scen}_${lib}_${layout}"
      tl="logs/aix_p2p_timeline_hethires3_${name}"
      [[ -d "${tl}" ]] && tl_args+=(--timeline "${name}=${tl}")
    done
  done
  coll="s${scen}_aix_coll_default"
  coll_tr="scorep_runs/hethires3_${coll}_hethires3_${coll}/traces.otf2"
  OTF2_PRINT="${OC}" python3 "${P2P_PHASES}" \
      --output-dir results/hethires3 \
      "${tl_args[@]}" \
      --coll-trace "${coll}=${coll_tr}" \
      --file-prefix "s${scen}" \
    || echo "[WARN] phases failed for s${scen}"
done

echo "--- rebuild overview tables ---"
python3 build_hh3_overview.py
echo "overview rc=$?"
python3 plot_hh3_scaling.py
echo "scaling rc=$?"
echo "=== HH3 rerun analysis done $(date) ==="
