#!/usr/bin/env bash
# Runs the production 96c4g hires configuration for AIx collective and pipelined
# P2P with NO instrumentation: no Score-P, no tracing/profiling, no P2P
# timeline CSVs, and no per-step memory-log CSVs. The collective job builds a
# fresh plain solver (WITH_SCOREP=OFF); the P2P job reuses that build via an
# afterok dependency. Compare against the instrumented 96c4g_hires runs with
# the STEP_TIMING lines in logs/mini_app_output_*.txt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build_cmi_plain"
JOB_SCRIPT="single_node_96c_4gpu.sh"

COMMON_EXPORTS="ALL,USE_SCOREP_ENV=0,SKIP_PAPI_METRICS=1,SCOREP_METRIC_PAPI=,MLCOUPLING_MEM_LOG_VERBOSE=0,MODEL_NAME_ENV=watercnn,TOTAL_STEPS_ENV=202,SAVE_EVERY_ENV=202,MPI_RANKS_ENV=96,RANK_GRID_X_ENV=12,RANK_GRID_Z_ENV=8,GPUS_PER_NODE_ENV=4,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=3840,TARGET_HEIGHT_ENV=2160,ML_BATCH_SIZE_ENV=20000000,COMPILE_OUTPUT_PATH_ENV=${BUILD_DIR},FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX"

echo "=== Submitting no-instrumentation 96c4g AIx suite (build dir: ${BUILD_DIR}) ==="

COLL_JID=$(sbatch --parsable --job-name=NI_96c4g_aix_coll --account=rwth0792 \
    --export="${COMMON_EXPORTS},CPP_ML_CONFIG_ENV=config_aix.toml,SKIP_COMPILE_ENV=0" \
    "${JOB_SCRIPT}")
echo "AIx collective (builds plain solver): ${COLL_JID}"

P2P_JID=$(sbatch --parsable --job-name=NI_96c4g_aix_p2p --account=rwth0792 \
    --dependency=afterok:${COLL_JID} \
    --export="${COMMON_EXPORTS},CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,SKIP_COMPILE_ENV=1" \
    "${JOB_SCRIPT}")
echo "AIx pipelined P2P (reuses build):     ${P2P_JID}"

echo ""
echo "After completion, compare STEP_TIMING lines against the instrumented"
echo "hires runs (3738533 collective, 3738549/3738560 P2P)."
