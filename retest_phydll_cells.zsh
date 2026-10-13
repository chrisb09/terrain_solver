#!/bin/zsh
# PhyDLL native C++ DL client retest runner.
# Matches AIx/SmartSim retest in model, resolution, steps, and configs:
#   24c1g    : 1 node / 1 GPU  -- 24 solver ranks + 1 DL client  (mpirun MPMD)
#   96c4g    : 1 node / 4 GPUs -- 96 solver ranks + 4 DL clients (mpirun MPMD)
#   24p24c1g : 2 nodes het     -- 24 CPU-side ranks + 24 GPU-side ranks + 1 DL client
#   96p96c4g : 2 nodes het     -- 96 CPU-side ranks + 96 GPU-side ranks + 4 DL clients
#
# Inference caps (post-aggregation sub-batch limit per DL client, not transport):
#   watercnn8k  : 20,000,000 samples  (larger than any single frame)
#   giantmlp288 :    500,000 samples  (larger than any single frame)
#
# Builds reused from existing production builds (WITH_PHYDLL=ON required):
#   single:  build_matrix_default  + CPP-ML-Interface/dl_clients/build-miniapp
#   het:     build_het_matrix_default + CPP-ML-Interface/dl_clients/build-miniapp
#
# Results written to results/retest_phydll/<model>/<cfg>/.
set -uo pipefail

MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
cd "${MINI_APP_DIR}" || exit 1
DRIVER="${DRIVER_ENV:-${MINI_APP_DIR}/5r_driver_single.zsh}"

CFG="${CFG_ENV:?CFG_ENV must be set}"
REPEAT="${REPEAT_ENV:?REPEAT_ENV must be set}"
MODEL="${MODEL_ENV:?MODEL_ENV must be set}"
BUILD_SET="${BUILD_SET_ENV:-matrix}"

case "${MODEL}" in
  watercnn8k)   MN=watercnn;            TW=7680; TH=4320; BATCH=20000000; STEPS=202; SAVE=202; CHUNK=60 ;;
  giantmlp288)  MN=benchmark_giant_mlp; TW=480;  TH=288;  BATCH=500000;   STEPS=12;  SAVE=12;  CHUNK=16 ;;
  *) echo "Unknown MODEL_ENV=${MODEL}" >&2; exit 1 ;;
esac

# DL client count equals the GPU count for the config.
case "${CFG}" in
  24c1g)    NP_DL=1 ;;
  96c4g)    NP_DL=4 ;;
  24p24c1g) NP_DL=1 ;;
  96p96c4g) NP_DL=4 ;;
  *) echo "Unknown CFG_ENV=${CFG}" >&2; exit 1 ;;
esac

OUT_DIR="${MINI_APP_DIR}/results/retest_phydll/${MODEL}/${CFG}"
mkdir -p "${OUT_DIR}" "${MINI_APP_DIR}/logs"
MANIFEST="${OUT_DIR}/runs_${CFG}_${MODEL}_r${REPEAT}.csv"
echo "run_id_case,rc" > "${MANIFEST}"

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
export CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL
export CPP_ML_CONFIG_ENV=config_phydll.toml
export MODEL_NAME_ENV="${MN}"
export TARGET_WIDTH_ENV="${TW}"
export TARGET_HEIGHT_ENV="${TH}"
export ML_BATCH_SIZE_ENV="${BATCH}"      # inference cap transmitted in metadata to DL client
export TOTAL_STEPS_ENV="${STEPS}"
export SAVE_EVERY_ENV="${SAVE}"
export CHUNK_SIZE_ENV="${CHUNK}"
export ML_INFERENCE_CPU_CORES_ENV=24
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
# PhyDLL: one C++ DL client per GPU; DL_FIELD_COUNT=1 (packed transport).
export PHYDLL_NP_DL="${NP_DL}"
export PHYDLL_DL_FIELD_COUNT=1
export PHYDLL_DL_COUNT=1
# Reuse existing DL client binary; driver checks for WITH_PHYDLL=ON in cache.
export PHYDLL_DL_BUILD_DIR="${MINI_APP_DIR}/../CPP-ML-Interface/dl_clients/build-miniapp"

if [[ -n "${GRID_X_ENV:-}" && -n "${GRID_Z_ENV:-}" ]]; then
  export MPI_RANKS_ENV="${RANKS_ENV:?RANKS_ENV must be set with GRID_X_ENV}"
  export RANK_GRID_X_ENV="${GRID_X_ENV}"
  export RANK_GRID_Z_ENV="${GRID_Z_ENV}"
  export GPUS_PER_NODE_ENV="${GPUS_ENV:?GPUS_ENV must be set with GRID_X_ENV}"
fi

export RUN_ID_ENV="retest_phydll_${CFG}_${MODEL}_r${REPEAT}"
export JOB_NAME_ENV="${RUN_ID_ENV}"

echo "=========================================================================="
echo "[$(date '+%F %T')] retest_phydll run_id=${RUN_ID_ENV} NP_DL=${NP_DL}"
echo "=========================================================================="

zsh "${DRIVER}"
rc=$?
echo "${RUN_ID_ENV},${rc}" >> "${MANIFEST}"
echo "CASE_RESULT run_id=${RUN_ID_ENV} rc=${rc}"

echo "RetestPhyDLL ${CFG} ${MODEL} repeat ${REPEAT} done at $(date '+%F %T')"
