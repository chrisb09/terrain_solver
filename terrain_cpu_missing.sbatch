#!/bin/bash
#SBATCH --job-name=terrain-missing
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --array=0-9%2
#SBATCH --output=logs/terrain_cpu_missing_%A_%a.log
set -eo pipefail
mini=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
campaign=${TERRAIN_MISSING_CAMPAIGN:-terrain_missing_20261012}
[[ $campaign =~ ^[A-Za-z0-9_-]+$ ]] || exit 1
task=${SLURM_ARRAY_TASK_ID:-0}
[[ $task =~ ^[0-9]+$ ]] || exit 1
unset TERRAIN_RETEST_BUILD TERRAIN_PLAIN_BUILD TERRAIN_READINESS TERRAIN_PHYDLL_CONFIG SOLVER_BIN PHYDLL_DL_CLIENT SMARTSIM_PYTHON
export MODE=${MODE:-timing} RUN_TIMEOUT=7200 DB_START_TIMEOUT=180 MEMORY_INTERVAL=1.0
export SLURM_EXPORT_ENV=ALL
case ${TERRAIN_GROUP:-other} in
    other)
        (( task < 10 )) || exit 1
        case $MODE in
            timing) export STEPS=${STEPS:-22} CONFIGS='AIx SS_DEFAULT SS_TPQ1_I96_B96 SS_TPQ12_I8_B96 SS_TPQ96_I1_B96' ;;
            memory) export STEPS=6 CONFIGS='AIx SS_DEFAULT SS_TPQ96_I1_B96'
                export APPROVED_PHYDLL_SHA256=dd551169e436d07e367fc5304389eafba3999f43587a20af0ec0fc8b8c87f553 ;;
            *) exit 1 ;;
        esac
        export DATASET_ID=${campaign}_other ;;
    phydll)
        [[ $MODE == timing ]] && (( task < 20 )) || exit 1
        export TERRAIN_READINESS=false
        (( task < 10 )) || export TERRAIN_READINESS=true
        export SLURM_ARRAY_TASK_ID=$((task % 10)) STEPS=${STEPS:-22} CONFIGS=PHYDLL_CPP
        export DATASET_ID=${campaign}_readiness_${TERRAIN_READINESS} ;;
    *) exit 1 ;;
esac
# Read the whole runner before execution so later edits cannot shift Bash's file offset.
exec bash -c "$(<"$mini/terrain_cpu_final.sbatch")" "$mini/terrain_cpu_final.sbatch"
