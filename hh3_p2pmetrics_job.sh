#!/bin/zsh
# Fill missing scale (s8-s11) p2p_inference_metrics.csv by running
# analyze_p2p_inference_phases.py with timelines only (no OTF2 traces), then
# rebuild the overview tables + scaling plots.
#SBATCH --job-name=hh3_p2pmetrics
#SBATCH --partition=devel
#SBATCH --account=default
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/hh3_p2pmetrics_%j.txt

set -uo pipefail
cd /hpcwork/ro092286/smartsim/mini_app
source /hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/activate
P2P_PHASES=/hpcwork/ro092286/smartsim/CPP-ML-Interface/scripts/analyze_p2p_inference_phases.py

echo "=== HH3 p2p metrics fill start $(date) ==="

for layout in default affine numa; do
  for lib in aix_p2p aix_p2p_fullcredits; do
    # one invocation per scenario so each config gets its own metrics CSV
    for scen in 8 9 10 11; do
      name="s${scen}_${lib}_${layout}"
      tl="logs/aix_p2p_timeline_hethires3_${name}"
      out="results/hethires3/${name}"
      [[ -d "$tl" ]] || { echo "[SKIP] no timeline dir for $name"; continue; }
      [[ -f "$out/p2p_inference_metrics.csv" ]] && { echo "[SKIP-DONE] $name"; continue; }
      echo "--- $name ---"
      python3 "$P2P_PHASES" --output-dir results/hethires3 \
          --timeline "${name}=${tl}" --file-prefix "${scen}_fill" >/dev/null 2>&1 \
        || python3 "$P2P_PHASES" --output-dir results/hethires3 \
          --timeline "${name}=${tl}" >/dev/null 2>&1 \
        || echo "[WARN] phases failed for $name"
    done
  done
done

echo "--- fill done, rebuilding tables ---"
python3 build_hh3_overview.py
echo "overview rc=$?"
python3 plot_hh3_scaling.py
echo "scaling rc=$?"
echo "=== HH3 p2p metrics fill done $(date) ==="
