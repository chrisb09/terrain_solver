#!/bin/zsh
############################
# Giant-MLP super-high-res (7680x4320) campaign repeat.
# Runs 12 steps (1 warmup + 5 steady ML steps) at batch cap 500k.
############################
#SBATCH --job-name=giantmlp_96c4g
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=logs/mini_app_output_giantmlp_5r_%j.txt

set -uo pipefail

MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
cd "${MINI_APP_DIR}" || exit 1
AIX_ROOT="${MINI_APP_DIR}/../CPP-ML-Interface/extern/AIxeleratorService"
DRIVER="${MINI_APP_DIR}/5r_driver_single.zsh"
mkdir -p "${MINI_APP_DIR}/logs" "${MINI_APP_DIR}/results/giant_mlp_scaling"

REPEAT="${REPEAT_ENV:?REPEAT_ENV must be set (1..5)}"
CFG="${CFG_ENV:?CFG_ENV must be set}"
MANIFEST_DIR="${MINI_APP_DIR}/results/giant_mlp_scaling/${CFG}"
MANIFEST="${MANIFEST_DIR}/runs_${CFG}_r${REPEAT}.csv"
mkdir -p "${MANIFEST_DIR}"

export USE_SCOREP_ENV=0
export SKIP_PAPI_METRICS=1
export SCOREP_METRIC_PAPI=
export MLCOUPLING_MEM_LOG_VERBOSE=0
export TOTAL_STEPS_ENV=12
export SAVE_EVERY_ENV=12
export MPI_RANKS_ENV=96
export RANK_GRID_X_ENV=12
export RANK_GRID_Z_ENV=8
export GPUS_PER_NODE_ENV=4
export DB_NODES_ENV=1
export ML_INFERENCE_CPU_CORES_ENV=96
export TARGET_WIDTH_ENV=7680
export TARGET_HEIGHT_ENV=4320
export MODEL_NAME_ENV=benchmark_giant_mlp
export ML_BATCH_SIZE_ENV=500000
export FORCE_FRESH_RUN_ENV=1
export OVERWRITE_OUTPUT_ENV=1
export SKIP_RENDERING_ENV=1
export USE_LOCAL_RUNTIME_STAGE_ENV=0
export USE_LOCAL_MODEL_CACHE_ENV=1
export USE_SMARTSIM=0
export USE_CPP_ML_INTERFACE=1
export ML_INTERFACE_ENV=cpp
export CPP_ML_INTERFACE_PROVIDER_ENV=AIX
export SOLVER_CPU_BIND_ENV=cores
export AIX_P2P_TIMELINE_ENV=0
export NIC_SAMPLING_ENV=0
export OVERWRITE_JOB_NAME_ENV=1
export JOB_NAME_ENV="giantmlp_${CFG}_r${REPEAT}"
export AIX_INSTALL_PREFIX_ENV="${AIX_ROOT}/INSTALL"
export COMPILE_OUTPUT_PATH_ENV="build_matrix_default"
export SKIP_COMPILE_ENV=1
export CPP_ML_CONFIG_ENV="config_aix.toml"
export RUN_ID_ENV="giantmlp_${CFG}_r${REPEAT}"

echo "run_id,jobid,repeat,layout,comm,rc" > "${MANIFEST}"
echo "Giant-MLP campaign: CFG=${CFG} repeat=${REPEAT} ranks=${MPI_RANKS_ENV} gpus=${GPUS_PER_NODE_ENV}"

zsh "${DRIVER}"
rc=$?
echo "${RUN_ID_ENV},${SLURM_JOB_ID:-NA},${REPEAT},default,coll,${rc}" >> "${MANIFEST}"
echo "GIANTMLP_RC=${rc} run_id=${RUN_ID_ENV}"
exit ${rc}
