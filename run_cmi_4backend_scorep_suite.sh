#!/usr/bin/env bash

# Exit on error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build_scorep_cmi"
JOB_SCRIPT_TEMPLATE="proper_slurm_job.sh"

# Pre-compilation phase
if [ "${SKIP_COMPILE_ENV:-0}" -eq 1 ]; then
    echo "=== Skipping compilation (SKIP_COMPILE_ENV=1) ==="
else
    echo "=== Step 1: Pre-compiling solver with Score-P manual instrumentation into ${BUILD_DIR} ==="
    export USE_SCOREP=1
    ./slurm_build.sh "${BUILD_DIR}"
    echo "=== Pre-compilation completed successfully ==="
fi

# Prepare temporary runner script with 20-minute time limit
TEMP_JOB_SCRIPT="$(mktemp .cmi_4backend_20min_XXXXXX.sh)"
sed -e 's/^#SBATCH --time=.*/#SBATCH --time=00:20:00/' "${JOB_SCRIPT_TEMPLATE}" > "${TEMP_JOB_SCRIPT}"
chmod +x "${TEMP_JOB_SCRIPT}"
trap 'rm -f "${TEMP_JOB_SCRIPT}"' EXIT

# Base parameters for all runs
COMMON_EXPORTS="ALL,USE_SCOREP_ENV=1,SCOREP_ENABLE_PROFILING_ENV=true,SCOREP_ENABLE_TRACING_ENV=false,SKIP_PAPI_METRICS=1,SCOREP_METRIC_PAPI=,MLCOUPLING_MEM_LOG_VERBOSE=1,MODEL_NAME_ENV=watercnn,TOTAL_STEPS_ENV=10,SAVE_EVERY_ENV=10,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=1920,TARGET_HEIGHT_ENV=1080,ML_BATCH_SIZE_ENV=50000,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${BUILD_DIR},FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1"

PREV_JID=""

submit_backend_job() {
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
echo "=== Submitting CMI Backend Score-P Jobs (10-step watercnn on 1920x1080) ==="

# 1. CMI SmartSim Shared (c0)
submit_backend_job "CMI_smartsim_watercnn_10step" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,DB_NODES_ENV=1,MPI_RANKS_ENV=24,RANK_GRID_X_ENV=6,RANK_GRID_Z_ENV=4,SCOREP_DIR_TAG_ENV=cmi_watercnn_smartsim"

# 2. CMI SmartSim Per-Node Standalone DB
submit_backend_job "CMI_smartsim_per_ml_node_watercnn_10step" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=per-ml-node,DB_NODES_ENV=1,MPI_RANKS_ENV=24,RANK_GRID_X_ENV=6,RANK_GRID_Z_ENV=4,SCOREP_DIR_TAG_ENV=cmi_watercnn_smartsim_per_ml_node"

# 3. CMI AIx Collective (MPMD requires 25 total ranks = 5x5 grid)
submit_backend_job "CMI_aix_watercnn_10step" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix.toml,MPI_RANKS_ENV=25,RANK_GRID_X_ENV=5,RANK_GRID_Z_ENV=5,SCOREP_DIR_TAG_ENV=cmi_watercnn_aix"

# 4. CMI PhyDLL C++ Client
submit_backend_job "CMI_phydll_cpp_watercnn_10step" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll.toml,MPI_RANKS_ENV=24,RANK_GRID_X_ENV=6,RANK_GRID_Z_ENV=4,USE_PYTHON_DL_CLIENT=0,SCOREP_DIR_TAG_ENV=cmi_watercnn_phydll_cpp"

# 5. CMI PhyDLL Python Client
submit_backend_job "CMI_phydll_py_watercnn_10step" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll.toml,MPI_RANKS_ENV=24,RANK_GRID_X_ENV=6,RANK_GRID_Z_ENV=4,USE_PYTHON_DL_CLIENT=1,PHYDLL_PY_SCOREP_WRAPPER=1,SCOREP_DIR_TAG_ENV=cmi_watercnn_phydll_py"

echo ""
echo "=== All CMI backend Score-P jobs submitted successfully ==="
