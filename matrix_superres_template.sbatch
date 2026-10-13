#!/bin/zsh
############################
# Super-high-res AIx comparison wrapper -- single c23g node, parameterized.
#
# 7680x4320 target at the config's rank grid gives each rank 4x the samples of
# the normal 3840x2160 run, so 96c4g_superres has the same per-rank payload as
# 24c1g at normal resolution (and 24c1g_superres 4x more again). Only the `numa`
# layout is run (layout was shown not to be the cause), with comm modes
# collective / p2p credit=1 / p2p credit=24.
#
# Generated from matrix_superres_template.sbatch by submit_superres_campaign.sh.
# P2P timelines are DISABLED (they perturb the timings under comparison).
############################
#SBATCH --job-name=matrix5r___CFG__
#SBATCH --account=__ACCOUNT__
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=__NTASKS__
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:__GPUS__
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=__TIME__
#SBATCH --output=logs/mini_app_output_5r_%j.txt

set -uo pipefail

MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
cd "${MINI_APP_DIR}" || exit 1
AIX_ROOT="${MINI_APP_DIR}/../CPP-ML-Interface/extern/AIxeleratorService"
DRIVER="${MINI_APP_DIR}/5r_driver_single.zsh"
REPEAT="${REPEAT_ENV:?REPEAT_ENV must be set (1..5)}"
CFG="${CFG_ENV:?CFG_ENV must be set}"
MANIFEST_DIR="${MINI_APP_DIR}/results/aix_layout_matrix/${CFG}"
MANIFEST="${MANIFEST_DIR}/runs_5r_${CFG}_r${REPEAT}.csv"
mkdir -p "${MANIFEST_DIR}" "${MINI_APP_DIR}/logs"

########## campaign constants ##########
export USE_SCOREP_ENV=0
export SKIP_PAPI_METRICS=1
export SCOREP_METRIC_PAPI=
export MLCOUPLING_MEM_LOG_VERBOSE=0
export TOTAL_STEPS_ENV=202
export SAVE_EVERY_ENV=202
export DB_NODES_ENV=1
export ML_INFERENCE_CPU_CORES_ENV=24
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
export JOB_NAME_ENV="5r_${CFG}_r${REPEAT}"

echo "config: CFG=${CFG} MPI_RANKS=${MPI_RANKS_ENV:-auto} RANK_GRID=${RANK_GRID_X_ENV:-auto}x${RANK_GRID_Z_ENV:-auto} GPUS_PER_NODE=${GPUS_PER_NODE_ENV:-auto} TARGET=${TARGET_WIDTH_ENV}x${TARGET_HEIGHT_ENV} MODEL=${MODEL_NAME_ENV} BATCH=${ML_BATCH_SIZE_ENV}"
echo "run_id_cell,jobid,repeat,layout,comm,rc" > "${MANIFEST}"

run_cell() {
  local layout="$1" comm="$2"
  local config credits
  case "${comm}" in
    coll)  config="config_aix.toml";           credits="" ;;
    p2p1)  config="config_aix_pipelined.toml"; credits="1" ;;
    p2p24) config="config_aix_pipelined.toml"; credits="24" ;;
    *) echo "unknown comm ${comm}"; return 0 ;;
  esac
  export AIX_INSTALL_PREFIX_ENV="${AIX_ROOT}/INSTALL-NUMA"
  export COMPILE_OUTPUT_PATH_ENV="build_matrix_numa_v3"
  export SKIP_COMPILE_ENV=1
  export CPP_ML_CONFIG_ENV="${config}"
  if [[ -n "${credits}" ]]; then
    export AIX_P2P_INITIAL_CREDITS_ENV="${credits}"
  else
    unset AIX_P2P_INITIAL_CREDITS_ENV
  fi
  export RUN_ID_ENV="5r_${CFG}_${layout}_${comm}_r${REPEAT}"
  echo "=================================================================="
  echo "[$(date '+%F %T')] cell layout=${layout} comm=${comm} run_id=${RUN_ID_ENV}"
  echo "=================================================================="
  zsh "${DRIVER}"
  local rc=$?
  echo "CELL_RESULT layout=${layout} comm=${comm} repeat=${REPEAT} rc=${rc} run_id=${RUN_ID_ENV}"
  echo "${RUN_ID_ENV},${SLURM_JOB_ID:-NA},${REPEAT},${layout},${comm},${rc}" >> "${MANIFEST}"
  return 0
}

echo "Campaign ${CFG} repeat ${REPEAT} starting on $(hostname) at $(date '+%F %T')"
for comm in coll p2p1 p2p24; do
  run_cell numa "${comm}"
done
echo "Campaign ${CFG} repeat ${REPEAT} done at $(date '+%F %T')"
