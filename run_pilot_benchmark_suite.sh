#!/usr/bin/env bash

# Exit on error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_CMI="build_cmi"
BUILD_DIRECT_AIX="build_direct_aix"
BUILD_DIRECT_SMARTSIM="build_direct_smartsim"

PROFILE="${1:-all}" # cpu, gpu, or all
DRY_RUN="${DRY_RUN:-0}" # 1 = print commands only

echo "================================================================="
echo "=== Pilot Benchmark Suite: 11 Modes x 2 Profiles (WaterCNN)   ==="
echo "================================================================="
echo "Selected profile: ${PROFILE} (DRY_RUN=${DRY_RUN})"

# Pre-compilation phase
if [ "${SKIP_COMPILE_ENV:-0}" -eq 1 ]; then
    echo "=== Skipping compilation (SKIP_COMPILE_ENV=1) ==="
else
    echo "=== Step 1: Pre-compiling all 3 solver binaries ==="
    MODE=cmi SKIP_TIMESTAMP_CHECK=1 ./build.sh "${BUILD_CMI}"
    MODE=direct_aix SKIP_TIMESTAMP_CHECK=1 ./build.sh "${BUILD_DIRECT_AIX}"
    MODE=direct_smartsim SKIP_TIMESTAMP_CHECK=1 ./build.sh "${BUILD_DIRECT_SMARTSIM}"
    echo "=== Pre-compilation completed successfully ==="
fi

# Prepare runner script by commenting out embedded #SBATCH lines so sbatch CLI controls hetjob topology
TEMP_JOB_SCRIPT="$(mktemp .pilot_suite_runner_XXXXXX.sh)"
sed 's/^#SBATCH/##SBATCH/g' proper_slurm_job.sh > "${TEMP_JOB_SCRIPT}"
chmod +x "${TEMP_JOB_SCRIPT}"
trap 'rm -f "${TEMP_JOB_SCRIPT}"' EXIT

PREV_JID="${PREV_JID_ENV:-}"
SUBMITTED_COUNT=0
CURRENT_CASE_INDEX=0
START_CASE_INDEX="${START_CASE_INDEX_ENV:-1}"

submit_job() {
    local job_name="$1"
    local prof="$2"
    local specific_exports="$3"

    CURRENT_CASE_INDEX=$((CURRENT_CASE_INDEX + 1))
    if [[ "${CURRENT_CASE_INDEX}" -lt "${START_CASE_INDEX}" ]]; then
        return 0
    fi

    local c0_part="c23mm"
    local c0_nodes=1
    local c0_tasks=24
    local c0_cpus=1
    local c0_mem="5G"

    local c1_part="c23g"
    local c1_nodes=1
    local c1_tasks=1
    local c1_cpus=8
    local c1_mem="5G"
    local c1_gres="gpu:1"

    local target_width=1920
    local target_height=1080
    local solver_mpi_ranks=24
    local rank_grid_x=6
    local rank_grid_z=4

    if [[ "${prof}" == "cpu" ]]; then
        c0_tasks=8
        target_width=480
        target_height=288
        solver_mpi_ranks=8
        rank_grid_x=4
        rank_grid_z=2

        c1_part="c23mm"
        c1_gres="none"
    fi

    local common_exports="ALL,USE_SCOREP_ENV=0,APPLY_CUDA_STUBS_ENV=1,PHYDLL_REBUILD_DL_CLIENT_ENV=0,MLCOUPLING_MEM_LOG_VERBOSE=1,MODEL_NAME_ENV=watercnn,TOTAL_STEPS_ENV=22,SAVE_EVERY_ENV=22,CHUNK_SIZE_ENV=12,ML_BATCH_SIZE_ENV=50000,SKIP_COMPILE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=${c1_cpus},TARGET_WIDTH_ENV=${target_width},TARGET_HEIGHT_ENV=${target_height},MPI_RANKS_ENV=${solver_mpi_ranks},RANK_GRID_X_ENV=${rank_grid_x},RANK_GRID_Z_ENV=${rank_grid_z},JOB_NAME_ENV=${job_name}"

    local full_exports="${common_exports},${specific_exports}"

    local sbatch_cmd="sbatch --parsable --job-name=${job_name} \
--time=00:20:00 \
--export=${full_exports} \
--partition=${c0_part} --nodes=${c0_nodes} --ntasks-per-node=${c0_tasks} --cpus-per-task=${c0_cpus} --mem-per-cpu=${c0_mem} \
: \
--partition=${c1_part} --nodes=${c1_nodes} --ntasks-per-node=${c1_tasks} --cpus-per-task=${c1_cpus} --mem-per-cpu=${c1_mem}"

    if [[ "${c1_gres}" != "none" ]]; then
        sbatch_cmd="${sbatch_cmd} --gres=${c1_gres}"
    fi

    if [[ -n "${PREV_JID}" ]]; then
        sbatch_cmd="${sbatch_cmd} --dependency=afterany:${PREV_JID}"
    fi

    echo "Submitting ${job_name} (${prof} profile)..."
    if [[ "${DRY_RUN}" -eq 1 ]]; then
        echo "  [DRY RUN] ${sbatch_cmd} ${TEMP_JOB_SCRIPT}"
        SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
        return 0
    fi

    local jid
    jid=$(eval ${sbatch_cmd} "${TEMP_JOB_SCRIPT}")

    if [ $? -eq 0 ]; then
        echo "  -> Submitted Job ID: ${jid}"
        if [ -n "${PREV_JID}" ]; then
            echo "     (Depends on ${PREV_JID})"
        fi
        if [[ "${ENABLE_SWATCH:-0}" == "1" ]] && [ -x "${HOME}/scripts/swatch" ]; then
            "${HOME}/scripts/swatch" --jobid "${jid}" 2>/dev/null || true
        fi
        PREV_JID="${jid}"
        SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
    else
        echo "  -> ERROR: Failed to submit ${job_name}" >&2
        exit 1
    fi
}

run_profile() {
    local p="$1"
    local p_upper
    p_upper=$(echo "${p}" | tr '[:lower:]' '[:upper:]')
    echo ""
    echo "================================================================="
    echo "=== Submitting 11 Cases for Profile: ${p_upper} ==="
    echo "================================================================="

    # --- Group A: Direct vs CMI Comparison (7 cases) ---
    # 1. Direct SmartSim (split I/O, one-time terrain preload)
    submit_job "pilot_${p}_direct_smartsim_preload" "${p}" \
        "COMPILE_OUTPUT_PATH_ENV=${BUILD_DIRECT_SMARTSIM},USE_SMARTSIM=1,USE_CPP_ML_INTERFACE=0,USE_DIRECT_AIX=0,ML_INTERFACE_ENV=smartsim,FORCE_TERRAIN_UPLOAD_EACH_STEP_ENV=0,MODEL_ARTIFACT_MANIFEST_ENV=train_models/model_a/artifact_manifest_watercnn_split.json"

    # 2. Direct SmartSim (split I/O, forced terrain upload each ML step)
    submit_job "pilot_${p}_direct_smartsim_forceterrain" "${p}" \
        "COMPILE_OUTPUT_PATH_ENV=${BUILD_DIRECT_SMARTSIM},USE_SMARTSIM=1,USE_CPP_ML_INTERFACE=0,USE_DIRECT_AIX=0,ML_INTERFACE_ENV=smartsim,FORCE_TERRAIN_UPLOAD_EACH_STEP_ENV=1,MODEL_ARTIFACT_MANIFEST_ENV=train_models/model_a/artifact_manifest_watercnn_split.json"

    # 3. Direct AIx (Collective)
    submit_job "pilot_${p}_direct_aix_collective" "${p}" \
        "COMPILE_OUTPUT_PATH_ENV=${BUILD_DIRECT_AIX},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=0,USE_DIRECT_AIX=1,ML_INTERFACE_ENV=aix,AIX_COMMUNICATION_MODE_ENV=collective,MODEL_ARTIFACT_MANIFEST_ENV=train_models/model_a/artifact_manifest_watercnn_flat.json,RANK_GRID_X_ENV=,RANK_GRID_Z_ENV="

    # 4. Direct AIx (Pipelined)
    submit_job "pilot_${p}_direct_aix_pipelined" "${p}" \
        "COMPILE_OUTPUT_PATH_ENV=${BUILD_DIRECT_AIX},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=0,USE_DIRECT_AIX=1,ML_INTERFACE_ENV=aix,AIX_COMMUNICATION_MODE_ENV=pipelined,MODEL_ARTIFACT_MANIFEST_ENV=train_models/model_a/artifact_manifest_watercnn_flat.json,RANK_GRID_X_ENV=,RANK_GRID_Z_ENV="

    # 5. CMI AIx (Collective)
    submit_job "pilot_${p}_cmi_aix_collective" "${p}" \
        "COMPILE_OUTPUT_PATH_ENV=${BUILD_CMI},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,USE_DIRECT_AIX=0,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix.toml,MODEL_ARTIFACT_MANIFEST_ENV=train_models/model_a/artifact_manifest_watercnn_flat.json,RANK_GRID_X_ENV=,RANK_GRID_Z_ENV="

    # 6. CMI AIx (Pipelined)
    submit_job "pilot_${p}_cmi_aix_pipelined" "${p}" \
        "COMPILE_OUTPUT_PATH_ENV=${BUILD_CMI},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,USE_DIRECT_AIX=0,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,MODEL_ARTIFACT_MANIFEST_ENV=train_models/model_a/artifact_manifest_watercnn_flat.json,RANK_GRID_X_ENV=,RANK_GRID_Z_ENV="

    # 7. CMI SmartSim (Parallel puts c0)
    submit_job "pilot_${p}_cmi_smartsim_c0" "${p}" \
        "COMPILE_OUTPUT_PATH_ENV=${BUILD_CMI},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,USE_DIRECT_AIX=0,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,SMARTSIM_MPI_SEQUENTIAL_PUT=0,MODEL_ARTIFACT_MANIFEST_ENV=train_models/model_a/artifact_manifest_watercnn_flat.json"

    # --- Group B: PhyDLL Basic Performance (4 cases) ---
    # 8. CMI PhyDLL C++ (Packed layout)
    submit_job "pilot_${p}_phydll_cpp_packed" "${p}" \
        "COMPILE_OUTPUT_PATH_ENV=${BUILD_CMI},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,USE_DIRECT_AIX=0,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll_packed.toml,USE_PYTHON_DL_CLIENT=0,PHYDLL_DL_FIELD_COUNT=1,PHYDLL_SAFE_MPI_ENV=1,MODEL_ARTIFACT_MANIFEST_ENV=train_models/model_a/artifact_manifest_watercnn_flat.json"

    # 9. CMI PhyDLL C++ (Uniform Chunks layout)
    submit_job "pilot_${p}_phydll_cpp_uniform" "${p}" \
        "COMPILE_OUTPUT_PATH_ENV=${BUILD_CMI},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,USE_DIRECT_AIX=0,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll_uniform.toml,USE_PYTHON_DL_CLIENT=0,PHYDLL_DL_FIELD_COUNT=1,PHYDLL_SAFE_MPI_ENV=1,MODEL_ARTIFACT_MANIFEST_ENV=train_models/model_a/artifact_manifest_watercnn_flat.json"

    # 10. CMI PhyDLL Python (Packed layout)
    submit_job "pilot_${p}_phydll_py_packed" "${p}" \
        "COMPILE_OUTPUT_PATH_ENV=${BUILD_CMI},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,USE_DIRECT_AIX=0,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll_packed.toml,USE_PYTHON_DL_CLIENT=1,PHYDLL_DL_FIELD_COUNT=1,PHYDLL_SAFE_MPI_ENV=1,MODEL_ARTIFACT_MANIFEST_ENV=train_models/model_a/artifact_manifest_watercnn_flat.json"

    # 11. CMI PhyDLL Python (Uniform Chunks layout)
    submit_job "pilot_${p}_phydll_py_uniform" "${p}" \
        "COMPILE_OUTPUT_PATH_ENV=${BUILD_CMI},USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,USE_DIRECT_AIX=0,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll_uniform.toml,USE_PYTHON_DL_CLIENT=1,PHYDLL_DL_FIELD_COUNT=1,PHYDLL_SAFE_MPI_ENV=1,MODEL_ARTIFACT_MANIFEST_ENV=train_models/model_a/artifact_manifest_watercnn_flat.json"
}

if [[ "${PROFILE}" == "cpu" || "${PROFILE}" == "all" ]]; then
    run_profile "cpu"
fi

if [[ "${PROFILE}" == "gpu" || "${PROFILE}" == "all" ]]; then
    run_profile "gpu"
fi

echo ""
echo "================================================================="
echo "=== Total jobs submitted: ${SUBMITTED_COUNT} ==="
echo "================================================================="
