#!/bin/zsh
# Calibration smoke for the synchronized-timing / no-op memory-log re-test.
# Measures, per single c23g node:
#   - WaterCNN 8k (7680x4320) AIx default/coll and SmartSim shared_c0
#     (+ SmartSim per-GPU DB c0 when the node has >1 GPU)
#   - Giant-MLP 270p (480x270) AIx default/coll and SmartSim shared_c0
#     (+ SmartSim per-GPU DB c0 when the node has >1 GPU)
# Cells run sequentially in one allocation so every cell shares the node.
# Uses the frozen driver 5r_driver_single.zsh and the fixed build_matrix_default
# (supports both AIX and SmartSim providers at runtime).
set -uo pipefail

MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
cd "${MINI_APP_DIR}" || exit 1
DRIVER="${MINI_APP_DIR}/5r_driver_single.zsh"
AIX_ROOT="${MINI_APP_DIR}/../CPP-ML-Interface/extern/AIxeleratorService"

CFG="${CFG_ENV:?CFG_ENV must be set}"
MPI_RANKS="${MPI_RANKS_ENV:?MPI_RANKS_ENV must be set}"
GRID_X="${RANK_GRID_X_ENV:?RANK_GRID_X_ENV must be set}"
GRID_Z="${RANK_GRID_Z_ENV:?RANK_GRID_Z_ENV must be set}"
GPUS="${GPUS_PER_NODE_ENV:?GPUS_PER_NODE_ENV must be set}"
CAL_SCOPE="${CAL_SCOPE:-ALL}"

MANIFEST_DIR="${MINI_APP_DIR}/results/retest_calibration/${CFG}"
mkdir -p "${MANIFEST_DIR}" "${MINI_APP_DIR}/logs"
MANIFEST="${MANIFEST_DIR}/calibration_runs_${CAL_SCOPE}.csv"
echo "run_id,provider,dbmode,model,res,jobid,rc" > "${MANIFEST}"

########## common ##########
export USE_SCOREP_ENV=0
export SKIP_PAPI_METRICS=1
export SCOREP_METRIC_PAPI=
export MLCOUPLING_MEM_LOG_VERBOSE=0
export MLCOUPLING_MEM_LOG_COLLECTIVE=0
export MLCOUPLING_STEP_TIMING_SYNC=1
export DB_NODES_ENV=1
export MPI_RANKS_ENV="${MPI_RANKS}"
export RANK_GRID_X_ENV="${GRID_X}"
export RANK_GRID_Z_ENV="${GRID_Z}"
export GPUS_PER_NODE_ENV="${GPUS}"
export ML_INFERENCE_CPU_CORES_ENV=24
export FORCE_FRESH_RUN_ENV=1
export OVERWRITE_OUTPUT_ENV=1
export SKIP_RENDERING_ENV=1
export USE_LOCAL_RUNTIME_STAGE_ENV=0
export USE_LOCAL_MODEL_CACHE_ENV=1
export USE_SMARTSIM=0
export USE_CPP_ML_INTERFACE=1
export ML_INTERFACE_ENV=cpp
export SOLVER_CPU_BIND_ENV=cores
export NIC_SAMPLING_ENV=0
export OVERWRITE_JOB_NAME_ENV=1
export COMPILE_OUTPUT_PATH_ENV=build_matrix_default
export SKIP_COMPILE_ENV=1

run_cell() {
  local provider="$1" dbmode="$2" model="$3"
  local run_id="cal_${CFG}_${model}_${provider}_${dbmode}"

  if [[ "${model}" == "watercnn8k" ]]; then
    export MODEL_NAME_ENV=watercnn
    export TARGET_WIDTH_ENV=7680
    export TARGET_HEIGHT_ENV=4320
    export ML_BATCH_SIZE_ENV=20000000
    export TOTAL_STEPS_ENV=202
    export SAVE_EVERY_ENV=202
    export CHUNK_SIZE_ENV=60
    export RANK_GRID_X_ENV="${GRID_X}"
    export RANK_GRID_Z_ENV="${GRID_Z}"
  else
    export MODEL_NAME_ENV=benchmark_giant_mlp
    export TARGET_WIDTH_ENV=480
    export TARGET_HEIGHT_ENV=270
    export ML_BATCH_SIZE_ENV=500000
    export TOTAL_STEPS_ENV=12
    export SAVE_EVERY_ENV=12
    export CHUNK_SIZE_ENV=30
    # 480x270 must be divisible by both the chunk size and the rank grid.
    if [[ "${MPI_RANKS}" -eq 24 ]]; then
      export RANK_GRID_X_ENV=8
      export RANK_GRID_Z_ENV=3
    elif [[ "${MPI_RANKS}" -eq 96 ]]; then
      export RANK_GRID_X_ENV=16
      export RANK_GRID_Z_ENV=6
    else
      export RANK_GRID_X_ENV="${GRID_X}"
      export RANK_GRID_Z_ENV="${GRID_Z}"
    fi
  fi

  if [[ "${provider}" == "aix" ]]; then
    export CPP_ML_INTERFACE_PROVIDER_ENV=AIX
    export CPP_ML_CONFIG_ENV=config_aix.toml
    export AIX_INSTALL_PREFIX_ENV="${AIX_ROOT}/INSTALL"
    unset DB_LAYOUT_ENV SMARTSIM_MPI_SEQUENTIAL_PUT SMARTSIM_DB_GPU_COUNT SMARTSIM_DB_GPU_COUNT_ENV
  else
    export CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM
    export CPP_ML_CONFIG_ENV=config.toml
    export SMARTSIM_MPI_SEQUENTIAL_PUT=0
    if [[ "${dbmode}" == "pergpu" ]]; then
      export DB_LAYOUT_ENV=per-ml-node
      export DB_NODES_ENV="${GPUS}"
      export SMARTSIM_DB_GPU_COUNT="${GPUS}"
    else
      export DB_LAYOUT_ENV=shared
      export DB_NODES_ENV=1
      unset SMARTSIM_DB_GPU_COUNT
    fi
  fi

  export RUN_ID_ENV="${run_id}"
  export JOB_NAME_ENV="${run_id}"

  echo "=================================================================="
  echo "[$(date '+%F %T')] cell run_id=${run_id} provider=${provider} dbmode=${dbmode} model=${MODEL_NAME_ENV} res=${TARGET_WIDTH_ENV}x${TARGET_HEIGHT_ENV} gpus=${GPUS}"
  echo "=================================================================="
  zsh "${DRIVER}"
  local rc=$?
  echo "${run_id},${provider},${dbmode},${MODEL_NAME_ENV},${TARGET_WIDTH_ENV}x${TARGET_HEIGHT_ENV},${SLURM_JOB_ID:-NA},${rc}" >> "${MANIFEST}"
  echo "CELL_RESULT run_id=${run_id} rc=${rc}"
  return 0
}

echo "Calibration ${CFG} starting on $(hostname) at $(date '+%F %T') (ranks=${MPI_RANKS} gpus=${GPUS} scope=${CAL_SCOPE})"

run_watercnn() {
  run_cell aix      shared  watercnn8k
  run_cell smartsim shared  watercnn8k
  if [[ "${GPUS}" -gt 1 ]]; then
    run_cell smartsim pergpu watercnn8k
  fi
}

run_giantmlp() {
  run_cell aix      shared  giantmlp270
  run_cell smartsim shared  giantmlp270
  if [[ "${GPUS}" -gt 1 ]]; then
    run_cell smartsim pergpu giantmlp270
  fi
}

case "${CAL_SCOPE}" in
  ALL|all)   run_watercnn; run_giantmlp ;;
  WCNN|wcnn) run_watercnn ;;
  GMLP|gmlp) run_giantmlp ;;
  *) echo "Unknown CAL_SCOPE=${CAL_SCOPE}" >&2 ;;
esac

echo "Calibration ${CFG} done at $(date '+%F %T')"
