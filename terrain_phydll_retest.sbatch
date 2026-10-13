#!/usr/bin/env bash
#SBATCH --job-name=terrain-phydll-retest
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --array=0-19%2
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/terrain_phydll_retest_%A_%a.log
set -eo pipefail
mini=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
task=${SLURM_ARRAY_TASK_ID:-0}
[[ $task =~ ^[0-9]+$ ]] && (( task < 20 )) || exit 1
export TERRAIN_RETEST_BUILD=$mini/solver_cpp/build_terrain_phydll_retest
export TERRAIN_READINESS=false
(( task < 10 )) || export TERRAIN_READINESS=true
export TERRAIN_PHYDLL_CONFIG=$mini/config_terrain_phydll_${TERRAIN_READINESS}.toml
export SLURM_ARRAY_TASK_ID=$((task % 10))
export MODE=${MODE:-performance} CONFIGS=PHYDLL_CPP
case $MODE in performance) export STEPS=22 ;; memory) export STEPS=6 ;; *) exit 1 ;; esac
campaign=${RETEST_CAMPAIGN:-phydll_master}
[[ $campaign =~ ^[A-Za-z0-9_-]+$ ]] || exit 1
export DATASET_ID=${campaign}_readiness_${TERRAIN_READINESS}
export RUN_TIMEOUT=7200 MEMORY_INTERVAL=1.0
if [[ ${PLAN_ONLY:-0} == 1 ]]; then
    printf 'retest_task=%s readiness=%s dataset=%s ' "$task" "$TERRAIN_READINESS" "$DATASET_ID"
fi
exec bash "$mini/terrain_cpu_final.sbatch"
