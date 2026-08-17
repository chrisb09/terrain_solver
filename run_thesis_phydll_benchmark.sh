#!/usr/bin/env bash

# Exit on error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_SCOREP="build_scorep_cmi"
BUILD_NATIVE="build_cmi_22step"

MODEL="${MODEL_ENV:-watercnn}"
TARGET_WIDTH="${TARGET_WIDTH_ENV:-480}"
TARGET_HEIGHT="${TARGET_HEIGHT_ENV:-264}"
TOTAL_STEPS="${TOTAL_STEPS_ENV:-10}"
SAVE_EVERY="${TOTAL_STEPS}"
CHUNK_SIZE="${CHUNK_SIZE_ENV:-12}"

echo "========================================================================="
echo " PhyDLL Wire Optimization & Transport Benchmark Suite (EXCLUSIVE NODES)"
echo " Model: ${MODEL} | Resolution: ${TARGET_WIDTH}x${TARGET_HEIGHT} | Steps: ${TOTAL_STEPS} | Chunk: ${CHUNK_SIZE}"
echo "========================================================================="

# Prepare temporary runner script with 15-minute time limit
TEMP_JOB_SCRIPT="$(mktemp .thesis_phydll_suite_XXXXXX.sh)"
sed -e 's/^#SBATCH --time=.*/#SBATCH --time=00:15:00/' proper_slurm_job.sh > "${TEMP_JOB_SCRIPT}"
chmod +x "${TEMP_JOB_SCRIPT}"
trap 'rm -f "${TEMP_JOB_SCRIPT}"' EXIT

BASE_EXPORTS="ALL,MLCOUPLING_MEM_LOG_VERBOSE=1,APPLY_CUDA_STUBS_ENV=1,PHYDLL_REBUILD_DL_CLIENT_ENV=0,PHYDLL_SAFE_MPI_ENV=1,MODEL_NAME_ENV=${MODEL},TOTAL_STEPS_ENV=${TOTAL_STEPS},SAVE_EVERY_ENV=${SAVE_EVERY},MPI_RANKS_ENV=24,RANK_GRID_X_ENV=6,RANK_GRID_Z_ENV=4,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=${TARGET_WIDTH},TARGET_HEIGHT_ENV=${TARGET_HEIGHT},CHUNK_SIZE_ENV=${CHUNK_SIZE},ML_BATCH_SIZE_ENV=50000,SKIP_COMPILE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL"

PREV_JID=""

submit_suite_job() {
    local job_name="$1"
    local specific_exports="$2"
    local full_exports="${BASE_EXPORTS},${specific_exports}"

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
echo "=== Phase 1: Score-P Profiled Runs (C++ Client, MPI & IB/Net Metrics) ==="

# 1. Score-P C++ Packed Baseline
submit_suite_job "Thesis_ScoreP_PhyDLL_cpp_packed" \
    "USE_SCOREP_ENV=1,SCOREP_ENABLE_PROFILING_ENV=true,SCOREP_ENABLE_TRACING_ENV=false,COMPILE_OUTPUT_PATH_ENV=${BUILD_SCOREP},CPP_ML_CONFIG_ENV=config_phydll_packed.toml,USE_PYTHON_DL_CLIENT=0,PHYDLL_DL_FIELD_COUNT=1,SCOREP_DIR_TAG_ENV=thesis_scorep_cpp_packed,JOB_NAME_ENV=thesis_scorep_cpp_packed"

# 2. Score-P C++ Auto Mode (selects uniform_chunks)
submit_suite_job "Thesis_ScoreP_PhyDLL_cpp_auto" \
    "USE_SCOREP_ENV=1,SCOREP_ENABLE_PROFILING_ENV=true,SCOREP_ENABLE_TRACING_ENV=false,COMPILE_OUTPUT_PATH_ENV=${BUILD_SCOREP},CPP_ML_CONFIG_ENV=config_phydll.toml,USE_PYTHON_DL_CLIENT=0,PHYDLL_DL_FIELD_COUNT=1,SCOREP_DIR_TAG_ENV=thesis_scorep_cpp_auto,JOB_NAME_ENV=thesis_scorep_cpp_auto"

# 3. Score-P C++ Explicit Uniform Chunks
submit_suite_job "Thesis_ScoreP_PhyDLL_cpp_uniform" \
    "USE_SCOREP_ENV=1,SCOREP_ENABLE_PROFILING_ENV=true,SCOREP_ENABLE_TRACING_ENV=false,COMPILE_OUTPUT_PATH_ENV=${BUILD_SCOREP},CPP_ML_CONFIG_ENV=config_phydll_uniform.toml,USE_PYTHON_DL_CLIENT=0,PHYDLL_DL_FIELD_COUNT=1,SCOREP_DIR_TAG_ENV=thesis_scorep_cpp_uniform,JOB_NAME_ENV=thesis_scorep_cpp_uniform"

echo ""
echo "=== Phase 2: Native Production Runs (Uninstrumented USE_SCOREP=0, Pure Wall-Time) ==="

# 4. Native C++ Packed Baseline
submit_suite_job "Thesis_Native_PhyDLL_cpp_packed" \
    "USE_SCOREP_ENV=0,COMPILE_OUTPUT_PATH_ENV=${BUILD_NATIVE},CPP_ML_CONFIG_ENV=config_phydll_packed.toml,USE_PYTHON_DL_CLIENT=0,PHYDLL_DL_FIELD_COUNT=1,JOB_NAME_ENV=thesis_native_cpp_packed"

# 5. Native C++ Auto Mode
submit_suite_job "Thesis_Native_PhyDLL_cpp_auto" \
    "USE_SCOREP_ENV=0,COMPILE_OUTPUT_PATH_ENV=${BUILD_NATIVE},CPP_ML_CONFIG_ENV=config_phydll.toml,USE_PYTHON_DL_CLIENT=0,PHYDLL_DL_FIELD_COUNT=1,JOB_NAME_ENV=thesis_native_cpp_auto"

# 6. Native Python Packed Baseline
submit_suite_job "Thesis_Native_PhyDLL_py_packed" \
    "USE_SCOREP_ENV=0,COMPILE_OUTPUT_PATH_ENV=${BUILD_NATIVE},CPP_ML_CONFIG_ENV=config_phydll_packed.toml,USE_PYTHON_DL_CLIENT=1,PHYDLL_DL_FIELD_COUNT=1,JOB_NAME_ENV=thesis_native_py_packed"

# 7. Native Python Auto Mode
submit_suite_job "Thesis_Native_PhyDLL_py_auto" \
    "USE_SCOREP_ENV=0,COMPILE_OUTPUT_PATH_ENV=${BUILD_NATIVE},CPP_ML_CONFIG_ENV=config_phydll.toml,USE_PYTHON_DL_CLIENT=1,PHYDLL_DL_FIELD_COUNT=1,JOB_NAME_ENV=thesis_native_py_auto"

echo ""
echo "=== All 7 thesis benchmark jobs submitted successfully ==="
