#!/bin/zsh
# Regenerate p2p_inference_metrics.csv for the re-run affine configs (s2/s4/s5
# x aix_p2p / aix_p2p_fullcredits) and rebuild the overview tables + scaling
# plots. The phases script takes ALL timeline pairs after a single --timeline
# (argparse nargs="*"), and its --coll-trace value is a directory.
#SBATCH --job-name=hh3_fix_metrics
#SBATCH --partition=devel
#SBATCH --account=default
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/hh3_fix_metrics_%j.txt

set -uo pipefail
cd /hpcwork/ro092286/smartsim/mini_app
source /hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/activate
P2P_PHASES=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/scripts/analyze_p2p_inference_phases.py
OC=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode/scorep-8.4-papi72-install/bin/otf2-print

echo "=== HH3 fix metrics start $(date) ==="

for scen in 2 4 5; do
  echo "--- p2p phases for scenario s${scen} ---"
  # single --timeline carrying all (default/affine/numa) pairs
  timeline_words=""
  for lib in aix_p2p aix_p2p_fullcredits; do
    for layout in default affine numa; do
      name="s${scen}_${lib}_${layout}"
      tl="logs/aix_p2p_timeline_hethires3_${name}"
      [[ -d "${tl}" ]] && timeline_words="${timeline_words} ${name}=${tl}"
    done
  done
  coll_dir="scorep_runs/hethires3_s${scen}_aix_coll_default_hethires3_s${scen}_aix_coll_default"
  OTF2_PRINT="${OC}" python3 "${P2P_PHASES}" \
      --output-dir results/hethires3 \
      --timeline ${=timeline_words} \
      --coll-trace "s${scen}_aix_coll_default=${coll_dir}" \
      --file-prefix "s${scen}" \
    || echo "[WARN] phases failed for s${scen}"
done

echo "--- verify affine metrics present ---"
for scen in 2 4 5; do
  for lib in aix_p2p aix_p2p_fullcredits; do
    f="results/hethires3/s${scen}_${lib}_affine/p2p_inference_metrics.csv"
    [[ -f "$f" ]] && echo "OK $f ($(($(wc -l < "$f")-1)) steps)" || echo "MISSING $f"
  done
done

echo "--- rebuild overview tables ---"
rm -f results/hethires3/overview_metric_cache.json
python3 build_hh3_overview.py
echo "overview rc=$?"
python3 plot_hh3_scaling.py
echo "scaling rc=$?"
echo "=== HH3 fix metrics done $(date) ==="
