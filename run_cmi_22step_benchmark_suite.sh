#!/usr/bin/env bash

# Exit on error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Build directories
BUILD_STD="build_cmi_22step"

# Pre-compilation phase
if [ "${SKIP_COMPILE_ENV:-0}" -eq 1 ]; then
    echo "=== Skipping compilation (SKIP_COMPILE_ENV=1) ==="
else
    echo "=== Step 1: Pre-compiling CMI Solver into ${BUILD_STD} ==="
    ./build.sh "${BUILD_STD}"
    echo "=== Pre-compilation completed successfully ==="
fi

# Prepare temporary runner script with 20-minute time limit
TEMP_JOB_SCRIPT="$(mktemp .cmi_22step_suite_XXXXXX.sh)"
sed -e 's/^#SBATCH --time=.*/#SBATCH --time=00:20:00/' proper_slurm_job.sh > "${TEMP_JOB_SCRIPT}"
chmod +x "${TEMP_JOB_SCRIPT}"
trap 'rm -f "${TEMP_JOB_SCRIPT}"' EXIT

# Base parameters for all runs
COMMON_EXPORTS="ALL,USE_SCOREP_ENV=0,APPLY_CUDA_STUBS_ENV=1,PHYDLL_REBUILD_DL_CLIENT_ENV=0,MLCOUPLING_MEM_LOG_VERBOSE=1,MODEL_NAME_ENV=watercnn,TOTAL_STEPS_ENV=22,SAVE_EVERY_ENV=22,MPI_RANKS_ENV=24,RANK_GRID_X_ENV=6,RANK_GRID_Z_ENV=4,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=1920,TARGET_HEIGHT_ENV=1080,ML_BATCH_SIZE_ENV=50000,SKIP_COMPILE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1"

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
echo "=== Step 3: Submitting 22-Step Benchmark Suite (Uninstrumented USE_SCOREP=0, 10 Warm ML Samples) ==="

# 1. SmartSim Parallel Puts (SMARTSIM_MPI_SEQUENTIAL_PUT=0)
submit_suite_job "CMI_SmartSim_c0_22step" \
    "COMPILE_OUTPUT_PATH_ENV=${BUILD_STD},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,SMARTSIM_MPI_SEQUENTIAL_PUT=0"

# 2. SmartSim Sequential Chain 1 (SMARTSIM_MPI_SEQUENTIAL_PUT=1)
submit_suite_job "CMI_SmartSim_c1_22step" \
    "COMPILE_OUTPUT_PATH_ENV=${BUILD_STD},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,SMARTSIM_MPI_SEQUENTIAL_PUT=1"

# 3. SmartSim Sequential Chain 3 (SMARTSIM_MPI_SEQUENTIAL_PUT=3)
submit_suite_job "CMI_SmartSim_c3_22step" \
    "COMPILE_OUTPUT_PATH_ENV=${BUILD_STD},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,SMARTSIM_MPI_SEQUENTIAL_PUT=3"

# 4. AIxelerator Collective
submit_suite_job "CMI_AIx_collective_22step" \
    "COMPILE_OUTPUT_PATH_ENV=${BUILD_STD},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix.toml,MPI_RANKS_ENV=24,RANK_GRID_X_ENV=,RANK_GRID_Z_ENV="

# 5. AIxelerator P2P / Pipelined
submit_suite_job "CMI_AIx_p2p_22step" \
    "COMPILE_OUTPUT_PATH_ENV=${BUILD_STD},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,MPI_RANKS_ENV=24,RANK_GRID_X_ENV=,RANK_GRID_Z_ENV="

# 6. PhyDLL C++ DL Client
submit_suite_job "CMI_PhyDLL_cpp_22step" \
    "COMPILE_OUTPUT_PATH_ENV=${BUILD_STD},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll.toml,USE_PYTHON_DL_CLIENT=0"

# 7. PhyDLL Python DL Client
submit_suite_job "CMI_PhyDLL_py_22step" \
    "COMPILE_OUTPUT_PATH_ENV=${BUILD_STD},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll.toml,USE_PYTHON_DL_CLIENT=1"

echo ""
echo "=== All 7 benchmark cases submitted successfully ==="
