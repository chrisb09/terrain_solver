#!/bin/zsh
############################
# Giant-MLP smoke test -- 24c1g, super-high-res (7680x4320), single cell (default/collective).
# Purpose: measure per-ML-step cost of benchmark_giant_mlp at 7680x4320 (33.18M samples)
# on 24 ranks / 1 GPU at batch cap 500k. 12 steps -> 1 warmup (step 2) + 5 steady ML steps.
############################
#SBATCH --job-name=smoke_giantmlp_24c1g_superres
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=01:25:00
#SBATCH --output=logs/mini_app_output_smoke_giantmlp_superres_%j.txt

set -uo pipefail

MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
cd "${MINI_APP_DIR}" || exit 1
AIX_ROOT="${MINI_APP_DIR}/../CPP-ML-Interface/extern/AIxeleratorService"
DRIVER="${MINI_APP_DIR}/5r_driver_single.zsh"
mkdir -p "${MINI_APP_DIR}/logs"

########## run configuration ##########
export USE_SCOREP_ENV=0
export SKIP_PAPI_METRICS=1
export SCOREP_METRIC_PAPI=
export MLCOUPLING_MEM_LOG_VERBOSE=0
export TOTAL_STEPS_ENV=12
export SAVE_EVERY_ENV=12
export MPI_RANKS_ENV=24
export RANK_GRID_X_ENV=6
export RANK_GRID_Z_ENV=4
export GPUS_PER_NODE_ENV=1
export DB_NODES_ENV=1
export ML_INFERENCE_CPU_CORES_ENV=24
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
export JOB_NAME_ENV="smoke_giantmlp_24c1g_superres"
export AIX_INSTALL_PREFIX_ENV="${AIX_ROOT}/INSTALL"
export COMPILE_OUTPUT_PATH_ENV="build_matrix_default"
export SKIP_COMPILE_ENV=1
export CPP_ML_CONFIG_ENV="config_aix.toml"
export RUN_ID_ENV="smoke_giantmlp_24c1g_superres"

echo "Giant-MLP super-res smoke: 24c1g, batch cap ${ML_BATCH_SIZE_ENV}, ${MPI_RANKS_ENV} ranks, ${GPUS_PER_NODE_ENV} GPU, steps ${TOTAL_STEPS_ENV}"
zsh "${DRIVER}"
rc=$?
echo "SMOKE_GIANTMLP_RC=${rc} run_id=${RUN_ID_ENV}"
exit ${rc}
