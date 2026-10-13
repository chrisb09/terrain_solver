#!/bin/zsh
# SmartSim-provider retest runner. Cases per config:
#   shared_c0   : DB_LAYOUT=shared,      DB_NODES=1,   SMARTSIM_MPI_SEQUENTIAL_PUT=0
#   pergpu_c0   : DB_LAYOUT=per-ml-node, DB_NODES=#GPUs, one DB pinned per GPU
#                 (4-GPU configs only)
# Models/res identical to the AIx retest: watercnn8k (7680x4320, chunk 60) or
# giantmlp288 (480x288, chunk 16). Writes results/retest_smartsim/<model>/<cfg>/.
#
# DB execution settings (applied to all runs in this retest):
#   SMARTSIM_THREADS_PER_QUEUE=1  -- one GPU execution worker; controller default
#                                    was previously cpu_cores/8=3 for GPU runs.
#   ML_DB_BATCH_SIZE_ENV          -- opportunistic RedisAI request coalescing up to
#                                    BATCH samples (model-specific); no minimum size
#                                    or timeout set, so coalescing is best-effort only.
set -uo pipefail

MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
cd "${MINI_APP_DIR}" || exit 1
DRIVER="${DRIVER_ENV:-${MINI_APP_DIR}/5r_driver_single.zsh}"
AIX_ROOT="${MINI_APP_DIR}/../CPP-ML-Interface/extern/AIxeleratorService"

CFG="${CFG_ENV:?CFG_ENV must be set}"
REPEAT="${REPEAT_ENV:?REPEAT_ENV must be set}"
MODEL="${MODEL_ENV:?MODEL_ENV must be set}"
BUILD_SET="${BUILD_SET_ENV:-matrix}"

case "${MODEL}" in
  watercnn8k)   MN=watercnn;             TW=7680; TH=4320; BATCH=20000000; STEPS=202; SAVE=202; CHUNK=60 ;;
  giantmlp288)  MN=benchmark_giant_mlp;  TW=480;  TH=288;  BATCH=500000;   STEPS=12;  SAVE=12;  CHUNK=16 ;;
  *) echo "Unknown MODEL_ENV=${MODEL}" >&2; exit 1 ;;
esac

OUT_DIR="${MINI_APP_DIR}/results/retest_smartsim/${MODEL}/${CFG}"
mkdir -p "${OUT_DIR}" "${MINI_APP_DIR}/logs"
MANIFEST="${OUT_DIR}/runs_${CFG}_${MODEL}_r${REPEAT}.csv"
echo "run_id_case,dbmode,rc" > "${MANIFEST}"

########## common ##########
export USE_SCOREP_ENV=0
export SKIP_PAPI_METRICS=1
export SCOREP_METRIC_PAPI=
export MLCOUPLING_MEM_LOG_VERBOSE=0
export MLCOUPLING_MEM_LOG_COLLECTIVE=0
export MLCOUPLING_STEP_TIMING_SYNC=1
export USE_SMARTSIM=0
export USE_CPP_ML_INTERFACE=1
export ML_INTERFACE_ENV=cpp
export CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM
export CPP_ML_CONFIG_ENV=config.toml
export SMARTSIM_MPI_SEQUENTIAL_PUT=0
export MODEL_NAME_ENV="${MN}"
export TARGET_WIDTH_ENV="${TW}"
export TARGET_HEIGHT_ENV="${TH}"
export ML_BATCH_SIZE_ENV="${BATCH}"
export TOTAL_STEPS_ENV="${STEPS}"
export SAVE_EVERY_ENV="${SAVE}"
export CHUNK_SIZE_ENV="${CHUNK}"
export ML_INFERENCE_CPU_CORES_ENV=24
# RedisAI DB execution settings -- must be set before the driver sources the
# controller so the controller reads these on startup.
export SMARTSIM_THREADS_PER_QUEUE_ENV=1
export ML_DB_BATCH_SIZE_ENV="${BATCH}"   # opportunistic coalescing cap, model-specific
export FORCE_FRESH_RUN_ENV=1
export OVERWRITE_OUTPUT_ENV=1
export SKIP_RENDERING_ENV=1
export USE_LOCAL_RUNTIME_STAGE_ENV=0
export USE_LOCAL_MODEL_CACHE_ENV=1
export SOLVER_CPU_BIND_ENV=cores
export NIC_SAMPLING_ENV=0
export OVERWRITE_JOB_NAME_ENV=1
export SKIP_COMPILE_ENV=1
export COMPILE_OUTPUT_PATH_ENV="build_${BUILD_SET}_default"
export AIX_INSTALL_PREFIX_ENV="${AIX_ROOT}/INSTALL"
if [[ -n "${GRID_X_ENV:-}" && -n "${GRID_Z_ENV:-}" ]]; then
  export MPI_RANKS_ENV="${RANKS_ENV:?RANKS_ENV must be set with GRID_X_ENV}"
  export RANK_GRID_X_ENV="${GRID_X_ENV}"
  export RANK_GRID_Z_ENV="${GRID_Z_ENV}"
  export GPUS_PER_NODE_ENV="${GPUS_ENV:?GPUS_ENV must be set with GRID_X_ENV}"
fi

run_case() {
  local dbmode="$1"   # shared_c0 | pergpu_c0
  if [[ "${dbmode}" == pergpu* ]]; then
    export DB_LAYOUT_ENV=per-ml-node
    export DB_NODES_ENV="${GPUS_PER_NODE_ENV:?pergpu needs GPUS_PER_NODE_ENV}"
    export SMARTSIM_DB_GPU_COUNT="${DB_NODES_ENV}"
  else
    export DB_LAYOUT_ENV=shared
    export DB_NODES_ENV=1
    unset SMARTSIM_DB_GPU_COUNT
  fi
  export RUN_ID_ENV="retest_smartsim_${CFG}_${MODEL}_${dbmode}_r${REPEAT}"
  export JOB_NAME_ENV="${RUN_ID_ENV}"
  echo "=================================================================="
  echo "[$(date '+%F %T')] case run_id=${RUN_ID_ENV} dbmode=${dbmode}"
  echo "=================================================================="
  zsh "${DRIVER}"
  local rc=$?
  echo "${RUN_ID_ENV},${dbmode},${rc}" >> "${MANIFEST}"
  echo "CASE_RESULT run_id=${RUN_ID_ENV} rc=${rc}"
  return 0
}

echo "RetestSmartSim ${CFG} ${MODEL} repeat ${REPEAT} on $(hostname) at $(date '+%F %T')"

run_case shared_c0
if [[ "${GPUS_PER_NODE_ENV:-1}" -gt 1 ]]; then
  run_case pergpu_c0
fi

echo "RetestSmartSim ${CFG} ${MODEL} repeat ${REPEAT} done at $(date '+%F %T')"
