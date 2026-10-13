#!/bin/zsh
#SBATCH --job-name=phydll_4k_s11_readiness
#SBATCH --account=thes2181
#SBATCH --partition=c23ms
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:12:00
#SBATCH --output=logs/phydll_4k_s11_readiness_%j.txt

#SBATCH hetjob

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:12:00

set -uo pipefail

CONFIG_TAG="phydll_readiness"
CONFIG_TOML="config_phydll_4k_readiness.toml"
SUITE="hethires3"

MINI_APP_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app"
cd "${MINI_APP_DIR}"
mkdir -p logs

SUMMARY="logs/phydll_summary_${SLURM_JOB_ID}.txt"
echo "PhyDLL 4K s11 readiness start=$(date)" > "${SUMMARY}"

export BASE_EXPORTS="ALL"
export USE_SCOREP="1"
export USE_SCOREP_ENV="1"
export SCOREP_ENABLE_PROFILING="true"
export SCOREP_ENABLE_PROFILING_ENV="true"
export SCOREP_ENABLE_TRACING="true"
export SCOREP_ENABLE_TRACING_ENV="true"
export SKIP_PAPI_METRICS="1"
export SCOREP_METRIC_PAPI=""
export MLCOUPLING_MEM_LOG_VERBOSE="1"
export MODEL_NAME="watercnn"
export MODEL_NAME_ENV="watercnn"
export TOTAL_STEPS="202"
export TOTAL_STEPS_ENV="202"
export SAVE_EVERY="202"
export SAVE_EVERY_ENV="202"
export TARGET_WIDTH="3840"
export TARGET_WIDTH_ENV="3840"
export TARGET_HEIGHT="2160"
export TARGET_HEIGHT_ENV="2160"
export CHUNK_SIZE="60"
export CHUNK_SIZE_ENV="60"
export ML_BATCH_SIZE="20000000"
export ML_BATCH_SIZE_ENV="20000000"
export SKIP_COMPILE="1"
export SKIP_COMPILE_ENV="1"
export FORCE_FRESH_RUN="1"
export FORCE_FRESH_RUN_ENV="1"
export OVERWRITE_OUTPUT="1"
export OVERWRITE_OUTPUT_ENV="1"
export SKIP_RENDERING="1"
export SKIP_RENDERING_ENV="1"
export USE_LOCAL_RUNTIME_STAGE="0"
export USE_LOCAL_RUNTIME_STAGE_ENV="0"
export USE_LOCAL_MODEL_CACHE="1"
export USE_LOCAL_MODEL_CACHE_ENV="1"
export USE_SMARTSIM="0"
export USE_CPP_ML_INTERFACE="1"
export ML_INTERFACE_ENV="cpp"
export CPP_ML_INTERFACE_PROVIDER_ENV="PHYDLL"
export CPP_ML_CONFIG_ENV="${CONFIG_TOML}"
export USE_PYTHON_DL_CLIENT="0"
export PHYDLL_DL_CLIENT="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/dl_clients/build-scorep-pilots/phydll_dl_client"
export COMPILE_OUTPUT_PATH_ENV="build_scorep_cmi_default"
export PHYDLL_LIB_DIR_ENV="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/phydll/build/lib"

sid=11
mm=1536
mm_nodes=16
gside=96
ngpus=4

export HET_MM_NODES_ENV="${mm_nodes}"
export MPI_RANKS_ENV="${mm}"
export HET_G_SIDE_TASKS_ENV="${gside}"
export GPUS_PER_NODE_ENV="${ngpus}"
export RUN_ID_ENV="${SUITE}_s${sid}_${CONFIG_TAG}"
export JOB_NAME_ENV="${SUITE}_s${sid}_${CONFIG_TAG}"
export SCOREP_DIR_TAG_ENV="${SUITE}_s${sid}_${CONFIG_TAG}"
export SMARTSIM_PORT_OFFSET_ENV="${sid}"
export PHYDLL_NP_DL_ENV="${ngpus}"
export PHYDLL_DL_OVERSUBSCRIBE_ENV="1"

echo "=== HH3 s${sid}_${CONFIG_TAG} (mm=${mm}, mm_nodes=${mm_nodes}, gside=${gside}, gpus=${ngpus}) start $(date) ===" >> "${SUMMARY}"
zsh "${MINI_APP_DIR}/het_hires_job.sh" >> "${SUMMARY}" 2>&1
rc=$?
echo "=== HH3 s${sid}_${CONFIG_TAG} finished rc=${rc} $(date) ===" >> "${SUMMARY}"
exit ${rc}
