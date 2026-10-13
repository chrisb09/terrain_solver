#!/bin/zsh
# Post-processing for the new suites (hh3_g1080, hh3_w8k) once their runs are
# complete: split summaries, per-config Score-P analysis, cross-config tail,
# overview tables, scaling-free comparison plots, and the AIx 5-stage
# prediction. All figure renderers honor --pgf.
#SBATCH --job-name=hh3_newsuites_analysis
#SBATCH --partition=devel
#SBATCH --account=default
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/hh3_newsuites_analysis_%j.txt

set -uo pipefail
cd /hpcwork/ro092286/smartsim/mini_app
source /hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/activate
CPP_ML=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface
export HH3_PGF_ENV=1   # suite-level figures also as .pgf/.pdf

echo "=== HH3 new-suites analysis start $(date) ==="
SUITES="${1:-hh3_g1080 hh3_w8k}"

echo "--- 1. split wrapper summaries into per-scenario solver logs ---"
python3 split_hh3_summaries.py ${=SUITES}

for suite in ${=SUITES}; do
  echo "--- 2. per-config Score-P analysis for ${suite} ---"
  # gantts: never for this analysis pass (fast); metrics CSVs are enough
  while IFS='|' read -r s c j; do
    [[ "${s}" != "${suite}" ]] && continue
    SKIP_CROSS_ENV=1 P2P_GANTT_ENV=0 FORCE_ENV=1 ./analyze_scorep_suites.sh "${s}" "${c}"
  done < "hh3_config_table_${suite}.txt"

  echo "--- 3. cross-config tail for ${suite} (makespan, p2p phases, comparisons) ---"
  CROSS_ONLY_ENV=1 ./analyze_scorep_suites.sh "${suite}"

  echo "--- 4. overview tables + comparison plots (with PGF) for ${suite} ---"
  python3 build_hh3_overview.py --suite "${suite}"
  python3 /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/scripts/plot_config_comparison.py "results/${suite}" \
      --output "results/${suite}/config_comparison.png" --pgf || echo "[WARN] config comparison failed for ${suite}"
  python3 /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/scripts/plot_avg_step_time.py "results/${suite}" --pgf \
      || echo "[WARN] avg step time plot failed for ${suite}"

  echo "--- 5. AIx 5-stage p2p predictions for ${suite} ---"
  bash run_hh3_predictions.sh "${suite}"
done

echo "=== HH3 new-suites analysis done $(date) ==="
