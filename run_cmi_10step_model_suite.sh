#!/usr/bin/env bash

# Exit on error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Build directories
BUILD_STD="build_cmi_22step"
BUILD_AIX_P2P="build_aix_p2p"

# Model and geometry for this suite
MODEL="${MODEL_ENV:-watercnn}"
TARGET_WIDTH="${TARGET_WIDTH_ENV:-480}"
TARGET_HEIGHT="${TARGET_HEIGHT_ENV:-264}"
TOTAL_STEPS="${TOTAL_STEPS_ENV:-10}"
SAVE_EVERY="${TOTAL_STEPS}"
CHUNK_SIZE="${CHUNK_SIZE_ENV:-12}"

echo "=== 10-Step CMI Model Benchmark Suite ==="
echo "Model: ${MODEL} | Resolution: ${TARGET_WIDTH}x${TARGET_HEIGHT} | Steps: ${TOTAL_STEPS} | Chunk: ${CHUNK_SIZE}"

# Pre-compilation phase
if [ "${SKIP_COMPILE_ENV:-0}" -eq 1 ]; then
    echo "=== Skipping compilation (SKIP_COMPILE_ENV=1) ==="
else
    echo "=== Step 1: Pre-compiling Standard CMI Solver into ${BUILD_STD} ==="
    ./build.sh "${BUILD_STD}"

    echo "=== Step 2: Pre-compiling AIx P2P CMI Solver into ${BUILD_AIX_P2P} ==="
    AIX_SERVICE_NAME_ENV=AIxeleratorService-pipelined ./build.sh "${BUILD_AIX_P2P}"
    echo "=== Pre-compilation completed successfully ==="
fi

# Prepare temporary runner script with 20-minute time limit
TEMP_JOB_SCRIPT="$(mktemp .cmi_10step_suite_XXXXXX.sh)"
sed -e 's/^#SBATCH --time=.*/#SBATCH --time=00:20:00/' proper_slurm_job.sh > "${TEMP_JOB_SCRIPT}"
chmod +x "${TEMP_JOB_SCRIPT}"
trap 'rm -f "${TEMP_JOB_SCRIPT}"' EXIT

# Base parameters for all runs
COMMON_EXPORTS="ALL,USE_SCOREP_ENV=0,APPLY_CUDA_STUBS_ENV=1,PHYDLL_REBUILD_DL_CLIENT_ENV=0,MLCOUPLING_MEM_LOG_VERBOSE=1,MODEL_NAME_ENV=${MODEL},TOTAL_STEPS_ENV=${TOTAL_STEPS},SAVE_EVERY_ENV=${SAVE_EVERY},MPI_RANKS_ENV=24,RANK_GRID_X_ENV=6,RANK_GRID_Z_ENV=4,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=${TARGET_WIDTH},TARGET_HEIGHT_ENV=${TARGET_HEIGHT},CHUNK_SIZE_ENV=${CHUNK_SIZE},ML_BATCH_SIZE_ENV=50000,SKIP_COMPILE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1"

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
        if [ -x "${HOME}/scripts/swatch" ]; then
            "${HOME}/scripts/swatch" --jobid "${jid}" 2>/dev/null || true
        fi
        PREV_JID="${jid}"
    else
        echo "  -> ERROR: Failed to submit ${job_name}" >&2
        exit 1
    fi
}

echo ""
echo "=== Submitting 10-Step ${MODEL} Benchmark Suite (480x264, Uninstrumented USE_SCOREP=0) ==="

# 1. SmartSim Parallel Puts (SMARTSIM_MPI_SEQUENTIAL_PUT=0)
submit_suite_job "CMI_${MODEL}_SmartSim_c0_10step" \
    "COMPILE_OUTPUT_PATH_ENV=${BUILD_STD},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,SMARTSIM_MPI_SEQUENTIAL_PUT=0"

# 2. SmartSim Sequential Chain 1 (SMARTSIM_MPI_SEQUENTIAL_PUT=1)
submit_suite_job "CMI_${MODEL}_SmartSim_c1_10step" \
    "COMPILE_OUTPUT_PATH_ENV=${BUILD_STD},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,SMARTSIM_MPI_SEQUENTIAL_PUT=1"

# 3. SmartSim Sequential Chain 3 (SMARTSIM_MPI_SEQUENTIAL_PUT=3)
submit_suite_job "CMI_${MODEL}_SmartSim_c3_10step" \
    "COMPILE_OUTPUT_PATH_ENV=${BUILD_STD},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,SMARTSIM_MPI_SEQUENTIAL_PUT=3"

# 4. AIxelerator Collective (auto rank grid, 24 solver ranks + 1 GPU controller = 25)
submit_suite_job "CMI_${MODEL}_AIx_collective_10step" \
    "COMPILE_OUTPUT_PATH_ENV=${BUILD_STD},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix.toml,RANK_GRID_X_ENV=,RANK_GRID_Z_ENV="

# 5. AIxelerator P2P / Pipelined (auto rank grid, 24 solver ranks + 1 GPU controller = 25)
submit_suite_job "CMI_${MODEL}_AIx_p2p_10step" \
    "COMPILE_OUTPUT_PATH_ENV=${BUILD_AIX_P2P},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix.toml,RANK_GRID_X_ENV=,RANK_GRID_Z_ENV="

echo ""
echo "=== All 5 ${MODEL} benchmark cases submitted successfully ==="
