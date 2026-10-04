#!/bin/bash
#SBATCH --job-name=terrain_final_gate
#SBATCH --account=rwth2150
#SBATCH --partition=c23ms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/terrain_cpu_final_workflow_%j.log
set -eo pipefail
cd /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
unset PYTHONHOME PYTHONPATH PYTHONOPTIMIZE LD_PRELOAD USE_SCOREP
source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh
PYTHON=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python
STAGE=${WORKFLOW_STAGE:-gate}
if [[ $STAGE == gate ]]; then
    DATASET=${SMOKE_DATASET:-smoke_pin_v2}
    REPS=1; STEADY=2
else
    [[ $STAGE == analysis && -n ${FINAL_DATASET:-} ]] || exit 1
    DATASET=$FINAL_DATASET
    REPS=5; STEADY=10
fi
[[ $DATASET =~ ^[A-Za-z0-9_-]+$ ]] || exit 1
ROOT=$PWD/results/terrain_cpu_final/$DATASET
FIGURES=$PWD/campaign_figures/terrain_cpu_final_$DATASET
"$PYTHON" terrain_cpu_final_analysis.py --campaign-root "$ROOT/performance" \
    --reps "$REPS" --steady-steps "$STEADY" --output-dir "$FIGURES/performance"
memory_args=()
if [[ $STAGE == analysis ]]; then
    memory_args+=(--reference-memory "$PWD/campaign_figures/nobatch_4584230/memory_latency/memory_latency_replicates.csv")
fi
"$PYTHON" terrain_cpu_final_analysis.py --mode memory --campaign-root "$ROOT/memory" \
    --reps "$REPS" --steady-steps 2 --output-dir "$FIGURES/memory" "${memory_args[@]}"
export FINAL_WORKFLOW_FIGURES=$FIGURES
"$PYTHON" - <<'PY'
import csv
import os
from pathlib import Path
path = Path(os.environ['FINAL_WORKFLOW_FIGURES']) / 'performance/phases/raw_callpaths.csv'
with path.open(newline='') as stream:
    rows = list(csv.DictReader(stream))
for r in rows:
    if (r['scope'] == 'dl_steady_count_validated' and r['selected_steady'] == 'True'
            and r['region'] == 'MPI_Barrier' and 'dl_send_output' in r['callpath']
            and int(r['raw_calls']) != 0):
        raise ValueError(f'Unexpected output-send synchronization: {r["cubex"]}')
print('Validated performance without memory sampling, matched memory follow-ups, and disabled PhyDLL output-send logging barriers.')
PY
if [[ $STAGE == analysis ]]; then
    printf 'Final figures and provenance: %s\n' "$FIGURES"
    exit 0
fi
# Export canonical values before sbatch: --export=ALL takes inherited values first.
for name in $(compgen -e); do
    case $name in SBATCH_*|SMARTSIM_*|SCOREP_*|MLCOUPLING_*|PHYDLL_*|OMP_*|MKL_*|OPENBLAS_*|KMP_*|GOMP_*|OMPI_*|PMIX_*) unset "$name" ;; esac
done
unset CONFIGS MODEL REP SOLVER_BIN RUN_ROOT LOG_ROOT TERRAIN_MEMORY_TOKEN SSDB
export APPROVED_PHYDLL_SHA256=dd551169e436d07e367fc5304389eafba3999f43587a20af0ec0fc8b8c87f553
export DATASET_ID=production_${SLURM_JOB_ID} RUN_TIMEOUT=7200 DB_START_TIMEOUT=180 MEMORY_INTERVAL=1.0
export MODE=performance STEPS=22
performance=$(sbatch --parsable --array=0-9%2 --job-name=terrain_final_performance --export=ALL terrain_cpu_final.sbatch)
export MODE=memory STEPS=6
memory=$(sbatch --parsable --array=0-9%2 --job-name=terrain_final_memory \
    --dependency="afterok:${performance%%;*}" --export=ALL terrain_cpu_final.sbatch)
export WORKFLOW_STAGE=analysis FINAL_DATASET=$DATASET_ID
analysis=$(sbatch --parsable --job-name=terrain_final_analysis \
    --dependency="afterok:${performance%%;*}:${memory%%;*}" --export=ALL terrain_cpu_final_workflow.sbatch)
printf '%s\n' "dataset=$DATASET_ID performance=$performance memory=$memory analysis=$analysis" | tee "$FIGURES/production_submission.txt"
