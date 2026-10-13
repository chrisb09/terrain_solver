#!/bin/zsh
############################
# 5-repeat AIx layout-matrix campaign -- single c23g node (96c4g).
# Runs all 9 (layout x comm) cells sequentially in one allocation, so every
# cell of a repeat shares the same node (no node-to-node confounding within a
# repeat). Five copies of this script are submitted (REPEAT_ENV=1..5) to obtain
# 5 x 100 = 500 steady-state samples per cell.
#
# Uses the frozen driver snapshot 5r_driver_single.zsh so that concurrent edits
# to single_node_96c_4gpu.sh by other work cannot change the campaign.
############################
#SBATCH --job-name=matrix5r_single
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=logs/mini_app_output_5r_single_%j.txt

set -uo pipefail

MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
cd "${MINI_APP_DIR}" || exit 1
AIX_ROOT="${MINI_APP_DIR}/../CPP-ML-Interface/extern/AIxeleratorService"
DRIVER="${MINI_APP_DIR}/5r_driver_single.zsh"
REPEAT="${REPEAT_ENV:?REPEAT_ENV must be set (1..5)}"
CFG="96c4g"
MANIFEST_DIR="${MINI_APP_DIR}/results/aix_layout_matrix/${CFG}"
MANIFEST="${MANIFEST_DIR}/runs_5r_${CFG}_r${REPEAT}.csv"
mkdir -p "${MANIFEST_DIR}" "${MINI_APP_DIR}/logs"

########## common matrix exports ##########
export USE_SCOREP_ENV=0
export SKIP_PAPI_METRICS=1
export SCOREP_METRIC_PAPI=
export MLCOUPLING_MEM_LOG_VERBOSE=0
export MODEL_NAME_ENV=watercnn
export TOTAL_STEPS_ENV=202
export SAVE_EVERY_ENV=202
export MPI_RANKS_ENV=96
export RANK_GRID_X_ENV=12
export RANK_GRID_Z_ENV=8
export GPUS_PER_NODE_ENV=4
export DB_NODES_ENV=1
export ML_INFERENCE_CPU_CORES_ENV=24
export TARGET_WIDTH_ENV=3840
export TARGET_HEIGHT_ENV=2160
export ML_BATCH_SIZE_ENV=20000000
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
# no NIC sampling: a background srun step is only reaped at job end, so it would
# accumulate across the 9 sequential cells
export NIC_SAMPLING_ENV=0
export OVERWRITE_JOB_NAME_ENV=1
# one shared output directory per repeat (cells run sequentially; force-fresh)
export JOB_NAME_ENV="5r_${CFG}_r${REPEAT}"

echo "run_id_cell,jobid,repeat,layout,comm,rc" > "${MANIFEST}"

run_cell() {
  local layout="$1" comm="$2"
  local install build config credits
  case "${layout}" in
    default) install="${AIX_ROOT}/INSTALL";        build="build_matrix_default" ;;
    affine)  install="${AIX_ROOT}/INSTALL-AFFINE"; build="build_matrix_affine" ;;
    numa)    install="${AIX_ROOT}/INSTALL-NUMA";   build="build_matrix_numa_v3" ;;
    *) echo "unknown layout ${layout}"; return 0 ;;
  esac
  case "${comm}" in
    coll)  config="config_aix.toml";           credits="" ;;
    p2p1)  config="config_aix_pipelined.toml"; credits="1" ;;
    p2p23) config="config_aix_pipelined.toml"; credits="23" ;;
    *) echo "unknown comm ${comm}"; return 0 ;;
  esac
  export AIX_INSTALL_PREFIX_ENV="${install}"
  export COMPILE_OUTPUT_PATH_ENV="${build}"
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
for layout in default affine numa; do
  for comm in coll p2p1 p2p23; do
    run_cell "${layout}" "${comm}"
  done
done
echo "Campaign ${CFG} repeat ${REPEAT} done at $(date '+%F %T')"
