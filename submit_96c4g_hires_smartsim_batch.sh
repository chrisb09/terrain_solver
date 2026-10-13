#!/usr/bin/env bash
# Submit only the 3 new SmartSim DB-batching/TPQ variants of the 96c4g hires
# suite (the other 9 cases already ran with jobs 3738506-3738560).
# Usage: SBATCH_ACCOUNT=rwth0792 SKIP_COMPILE_ENV=1 ./submit_96c4g_hires_smartsim_batch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build_scorep_cmi"
JOB_SCRIPT_TEMPLATE="single_node_96c_4gpu.sh"

if [ "${SKIP_COMPILE_ENV:-0}" -ne 1 ]; then
    echo "=== Compiling solver into ${BUILD_DIR} ==="
    ./slurm_build.sh "${BUILD_DIR}"
fi

TEMP_JOB_SCRIPT="$(mktemp .cmi_scorep_96c4g_batch_XXXXXX.sh)"
sed -e 's/^#SBATCH --time=.*/#SBATCH --time=00:40:00/' "${JOB_SCRIPT_TEMPLATE}" > "${TEMP_JOB_SCRIPT}"
chmod +x "${TEMP_JOB_SCRIPT}"
trap 'rm -f "${TEMP_JOB_SCRIPT}"' EXIT

COMMON_EXPORTS="ALL,USE_SCOREP_ENV=1,SCOREP_ENABLE_PROFILING_ENV=true,SKIP_PAPI_METRICS=1,SCOREP_METRIC_PAPI=,MLCOUPLING_MEM_LOG_VERBOSE=1,MODEL_NAME_ENV=watercnn,TOTAL_STEPS_ENV=202,SAVE_EVERY_ENV=202,MPI_RANKS_ENV=96,RANK_GRID_X_ENV=12,RANK_GRID_Z_ENV=8,GPUS_PER_NODE_ENV=4,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=3840,TARGET_HEIGHT_ENV=2160,ML_BATCH_SIZE_ENV=20000000,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${BUILD_DIR},FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1"

PREV_JID="${PREV_JID_ENV:-}"

submit_job() {
    local job_name="$1"
    local specific_exports="$2"
    local full_exports="${COMMON_EXPORTS},${specific_exports}"

    local sbatch_cmd="sbatch --parsable --job-name=${job_name} --export=${full_exports}"
    if [ -n "${PREV_JID}" ]; then
        sbatch_cmd="${sbatch_cmd} --dependency=afterany:${PREV_JID}"
    fi

    echo "Submitting ${job_name}..."
    local jid
    jid=$(eval ${sbatch_cmd} "${TEMP_JOB_SCRIPT}")
    echo "  -> Submitted Job ID: ${jid}"
    if [ -n "${PREV_JID}" ]; then
        echo "     (Depends on ${PREV_JID})"
    fi
    if [ -x "${HOME}/scripts/swatch" ]; then
        "${HOME}/scripts/swatch" --jobid "${jid}" 2>/dev/null || true
    fi
    PREV_JID="${jid}"
}

echo ""
echo "=== Submitting SmartSim DB-batching/TPQ variants (96c4g hires, profiling+tracing) ==="

submit_job "SP_96c4g_hires_smartsim_c0_batch20m" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,DB_NODES_ENV=1,SMARTSIM_MPI_SEQUENTIAL_PUT=0,ML_DB_BATCH_SIZE_ENV=20000000,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=96c4g_hires_smartsim_c0_batch20m"

submit_job "SP_96c4g_hires_smartsim_c0_tpq4" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,DB_NODES_ENV=1,SMARTSIM_MPI_SEQUENTIAL_PUT=0,SMARTSIM_THREADS_PER_QUEUE_ENV=4,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=96c4g_hires_smartsim_c0_tpq4"

submit_job "SP_96c4g_hires_smartsim_c0_batch20m_tpq4" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,DB_NODES_ENV=1,SMARTSIM_MPI_SEQUENTIAL_PUT=0,ML_DB_BATCH_SIZE_ENV=20000000,SMARTSIM_THREADS_PER_QUEUE_ENV=4,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=96c4g_hires_smartsim_c0_batch20m_tpq4"

echo ""
echo "=== All 3 SmartSim batch/TPQ cases submitted successfully ==="
echo "LAST_JOB_ID=${PREV_JID}"
