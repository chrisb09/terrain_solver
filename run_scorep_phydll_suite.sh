#!/usr/bin/env bash

# Exit on error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_SCOREP="build_scorep_cmi"

MODEL="${MODEL_ENV:-watercnn}"
TARGET_WIDTH="${TARGET_WIDTH_ENV:-480}"
TARGET_HEIGHT="${TARGET_HEIGHT_ENV:-264}"
TOTAL_STEPS="${TOTAL_STEPS_ENV:-10}"
SAVE_EVERY="${TOTAL_STEPS}"
CHUNK_SIZE="${CHUNK_SIZE_ENV:-12}"

echo "=== Score-P PhyDLL Measurement Suite ==="
echo "Model: ${MODEL} | Resolution: ${TARGET_WIDTH}x${TARGET_HEIGHT} | Steps: ${TOTAL_STEPS} | Chunk: ${CHUNK_SIZE}"

# Prepare temporary runner script with 15-minute time limit
TEMP_JOB_SCRIPT="$(mktemp .scorep_phydll_suite_XXXXXX.sh)"
sed -e 's/^#SBATCH --time=.*/#SBATCH --time=00:15:00/' proper_slurm_job.sh > "${TEMP_JOB_SCRIPT}"
chmod +x "${TEMP_JOB_SCRIPT}"
trap 'rm -f "${TEMP_JOB_SCRIPT}"' EXIT

COMMON_EXPORTS="ALL,USE_SCOREP_ENV=1,SCOREP_ENABLE_PROFILING_ENV=true,SCOREP_ENABLE_TRACING_ENV=false,MLCOUPLING_MEM_LOG_VERBOSE=1,APPLY_CUDA_STUBS_ENV=1,PHYDLL_REBUILD_DL_CLIENT_ENV=0,PHYDLL_SAFE_MPI_ENV=1,MODEL_NAME_ENV=${MODEL},TOTAL_STEPS_ENV=${TOTAL_STEPS},SAVE_EVERY_ENV=${SAVE_EVERY},MPI_RANKS_ENV=24,RANK_GRID_X_ENV=6,RANK_GRID_Z_ENV=4,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=${TARGET_WIDTH},TARGET_HEIGHT_ENV=${TARGET_HEIGHT},CHUNK_SIZE_ENV=${CHUNK_SIZE},ML_BATCH_SIZE_ENV=50000,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${BUILD_SCOREP},FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1"

PREV_JID=""

submit_suite_job() {
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

    if [ $? -eq 0 ]; then
        echo "  -> Submitted Job ID: ${jid}"
        if [ -n "${PREV_JID}" ]; then
            echo "     (Depends on ${PREV_JID})"
        fi
        PREV_JID="${jid}"
    else
        echo "  -> ERROR: Failed to submit ${job_name}" >&2
        exit 1
    fi
}

echo ""
echo "=== Submitting Score-P PhyDLL Benchmark Suite (${TARGET_WIDTH}x${TARGET_HEIGHT}, C++ client) ==="

# 1. PhyDLL C++ DL client, packed layout (baseline)
submit_suite_job "ScoreP_${MODEL}_PhyDLL_cpp_packed" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll.toml,USE_PYTHON_DL_CLIENT=0,PHYDLL_DL_FIELD_COUNT=1,SCOREP_DIR_TAG_ENV=phydll_cpp_packed,JOB_NAME_ENV=scorep_${MODEL}_phydll_cpp_packed"

# 2. PhyDLL C++ DL client, uniform_chunks layout
submit_suite_job "ScoreP_${MODEL}_PhyDLL_cpp_uniform" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll_uniform.toml,USE_PYTHON_DL_CLIENT=0,PHYDLL_DL_FIELD_COUNT=1,SCOREP_DIR_TAG_ENV=phydll_cpp_uniform,JOB_NAME_ENV=scorep_${MODEL}_phydll_cpp_uniform"

echo ""
echo "=== All Score-P PhyDLL measurement cases submitted successfully ==="
