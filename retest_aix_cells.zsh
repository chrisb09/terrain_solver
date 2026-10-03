#!/bin/zsh
# AIx retest runner: full 3x3 layout matrix (default/affine/numa x coll/p2p1/p2p24)
# for one (config, model, repeat). Cells run sequentially in one allocation.
# Writes to results/retest_aix/<model>/<cfg>/ so the old campaigns are untouched.
# Model/res: watercnn8k (7680x4320, 202 steps, chunk 60) or
#            giantmlp288 (480x288, 12 steps, chunk 16).
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

OUT_DIR="${MINI_APP_DIR}/results/retest_aix/${MODEL}/${CFG}"
mkdir -p "${OUT_DIR}" "${MINI_APP_DIR}/logs"
MANIFEST="${OUT_DIR}/runs_${CFG}_${MODEL}_r${REPEAT}.csv"
echo "run_id_cell,layout,comm,rc" > "${MANIFEST}"

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
export CPP_ML_INTERFACE_PROVIDER_ENV=AIX
export MODEL_NAME_ENV="${MN}"
export TARGET_WIDTH_ENV="${TW}"
export TARGET_HEIGHT_ENV="${TH}"
export ML_BATCH_SIZE_ENV="${BATCH}"
export TOTAL_STEPS_ENV="${STEPS}"
export SAVE_EVERY_ENV="${SAVE}"
export CHUNK_SIZE_ENV="${CHUNK}"
export DB_NODES_ENV=1
export ML_INFERENCE_CPU_CORES_ENV=24
export FORCE_FRESH_RUN_ENV=1
export OVERWRITE_OUTPUT_ENV=1
export SKIP_RENDERING_ENV=1
export USE_LOCAL_RUNTIME_STAGE_ENV=0
export USE_LOCAL_MODEL_CACHE_ENV=1
export SOLVER_CPU_BIND_ENV=cores
export AIX_P2P_TIMELINE_ENV=0
export NIC_SAMPLING_ENV=0
export OVERWRITE_JOB_NAME_ENV=1
export SKIP_COMPILE_ENV=1
# Single-node runs pass an explicit grid; het surrogate runs use the auto grid.
if [[ -n "${GRID_X_ENV:-}" && -n "${GRID_Z_ENV:-}" ]]; then
  export MPI_RANKS_ENV="${RANKS_ENV:?RANKS_ENV must be set with GRID_X_ENV}"
  export RANK_GRID_X_ENV="${GRID_X_ENV}"
  export RANK_GRID_Z_ENV="${GRID_Z_ENV}"
  export GPUS_PER_NODE_ENV="${GPUS_ENV:?GPUS_ENV must be set with GRID_X_ENV}"
fi

run_cell() {
  local layout="$1" comm="$2"
  local install build config credits
  case "${layout}" in
    default) install="${AIX_ROOT}/INSTALL";        build="build_${BUILD_SET}_default" ;;
    affine)  install="${AIX_ROOT}/INSTALL-AFFINE"; build="build_${BUILD_SET}_affine" ;;
    numa)    install="${AIX_ROOT}/INSTALL-NUMA";   build="build_${BUILD_SET}_numa_v3" ;;
    *) echo "unknown layout ${layout}"; return 0 ;;
  esac
  case "${comm}" in
    coll)  config="config_aix.toml";           credits="" ;;
    p2p1)  config="config_aix_pipelined.toml"; credits="1" ;;
    p2p24) config="config_aix_pipelined.toml"; credits="24" ;;
    *) echo "unknown comm ${comm}"; return 0 ;;
  esac
  export AIX_INSTALL_PREFIX_ENV="${install}"
  export COMPILE_OUTPUT_PATH_ENV="${build}"
  export CPP_ML_CONFIG_ENV="${config}"
  if [[ -n "${credits}" ]]; then
    export AIX_P2P_INITIAL_CREDITS_ENV="${credits}"
  else
    unset AIX_P2P_INITIAL_CREDITS_ENV
  fi
  export RUN_ID_ENV="retest_${CFG}_${MODEL}_${layout}_${comm}_r${REPEAT}"
  export JOB_NAME_ENV="${RUN_ID_ENV}"
  echo "=================================================================="
  echo "[$(date '+%F %T')] cell run_id=${RUN_ID_ENV}"
  echo "=================================================================="
  zsh "${DRIVER}"
  local rc=$?
  echo "${RUN_ID_ENV},${layout},${comm},${rc}" >> "${MANIFEST}"
  echo "CELL_RESULT run_id=${RUN_ID_ENV} rc=${rc}"
  return 0
}

echo "RetestAIx ${CFG} ${MODEL} repeat ${REPEAT} on $(hostname) at $(date '+%F %T')"

for layout in default affine numa; do
  for comm in coll p2p1 p2p24; do
    run_cell "${layout}" "${comm}"
  done
done

echo "RetestAIx ${CFG} ${MODEL} repeat ${REPEAT} done at $(date '+%F %T')"
