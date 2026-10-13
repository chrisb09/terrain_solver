#!/usr/bin/env bash
# Submits short hires 96c4g AIx P2P runs against the instrumented AIx build
# (AIxeleratorService-trace/INSTALL-TRACE) with P2P timeline CSVs enabled but
# no Score-P. The first job compiles the solver against the trace install; the
# second reuses that build. Both request short walltimes for faster backfill.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

JOB_TEMPLATE="single_node_96c_4gpu.sh"
TEMP_JOB_SCRIPT="$(mktemp .cmi_trace_XXXXXX.sh)"
sed -e 's/^#SBATCH --time=.*/#SBATCH --time=00:15:00/' \
    -e 's|AIxeleratorService/INSTALL"|AIxeleratorService-trace/INSTALL-TRACE"|g' \
    "${JOB_TEMPLATE}" > "${TEMP_JOB_SCRIPT}"
chmod +x "${TEMP_JOB_SCRIPT}"
trap 'rm -f "${TEMP_JOB_SCRIPT}"' EXIT

PARTITION="${PARTITION_ENV:-c23g}"
BUILD_DIR="build_trace_plain"
if [[ "${PARTITION}" == "c25g" ]]; then
    BUILD_DIR="build_trace_plain_c25g"
fi

COMMON="ALL,USE_SCOREP_ENV=0,SKIP_PAPI_METRICS=1,SCOREP_METRIC_PAPI=,MLCOUPLING_MEM_LOG_VERBOSE=0,MODEL_NAME_ENV=watercnn,TOTAL_STEPS_ENV=22,SAVE_EVERY_ENV=22,MPI_RANKS_ENV=96,RANK_GRID_X_ENV=12,RANK_GRID_Z_ENV=8,GPUS_PER_NODE_ENV=4,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=3840,TARGET_HEIGHT_ENV=2160,ML_BATCH_SIZE_ENV=20000000,COMPILE_OUTPUT_PATH_ENV=${BUILD_DIR},FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,AIX_P2P_TIMELINE_ENV=1"

echo "=== Submitting instrumented AIx P2P runs (trace install, partition ${PARTITION}) ==="
FC=$(sbatch --parsable --account=rwth0792 --partition="${PARTITION}" --job-name=TR_96c4g_fc \
    --export="${COMMON},CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,SKIP_COMPILE_ENV=0,AIX_P2P_INITIAL_CREDITS_ENV=23" \
    "${TEMP_JOB_SCRIPT}")
echo "fullcredits (builds trace solver): ${FC}"

C1=$(sbatch --parsable --account=rwth0792 --partition="${PARTITION}" --job-name=TR_96c4g_c1 --dependency=afterok:${FC} \
    --export="${COMMON},CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,SKIP_COMPILE_ENV=1" \
    "${TEMP_JOB_SCRIPT}")
echo "credit=1 (reuses build):          ${C1}"
