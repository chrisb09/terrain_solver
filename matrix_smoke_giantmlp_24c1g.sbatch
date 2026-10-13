#!/bin/zsh
############################
# Giant-MLP smoke test -- 24c1g, NON-exclusive, single cell (default/collective).
# Purpose: measure the per-ML-step cost of benchmark_giant_mlp at batch cap 1.5M
# on 24 ranks / 1 GPU, to extrapolate the walltime of future giant-MLP 5x100
# layout benchmarks. 22 steps -> 11 ML steps; the first ML step is the warmup,
# leaving 10 steady ML steps.
############################
#SBATCH --job-name=smoke_giantmlp_24c1g
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mini_app_output_smoke_giantmlp_%j.txt

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
export TOTAL_STEPS_ENV=22
export SAVE_EVERY_ENV=22
export MPI_RANKS_ENV=24
export RANK_GRID_X_ENV=6
export RANK_GRID_Z_ENV=4
export GPUS_PER_NODE_ENV=1
export DB_NODES_ENV=1
export ML_INFERENCE_CPU_CORES_ENV=24
export TARGET_WIDTH_ENV=3840
export TARGET_HEIGHT_ENV=2160
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
export JOB_NAME_ENV="smoke_giantmlp_24c1g"
export AIX_INSTALL_PREFIX_ENV="${AIX_ROOT}/INSTALL"
export COMPILE_OUTPUT_PATH_ENV="build_matrix_default"
export SKIP_COMPILE_ENV=1
export CPP_ML_CONFIG_ENV="config_aix.toml"
export RUN_ID_ENV="smoke_giantmlp_24c1g"

echo "Giant-MLP smoke: giant model, batch cap ${ML_BATCH_SIZE_ENV}, ${MPI_RANKS_ENV} ranks, ${GPUS_PER_NODE_ENV} GPU, steps ${TOTAL_STEPS_ENV}"
zsh "${DRIVER}"
rc=$?
echo "SMOKE_GIANTMLP_RC=${rc} run_id=${RUN_ID_ENV}"
exit ${rc}
