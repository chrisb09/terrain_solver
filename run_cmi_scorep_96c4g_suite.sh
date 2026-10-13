#!/usr/bin/env bash

# Exit on error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build_scorep_cmi"
JOB_SCRIPT_TEMPLATE="single_node_96c_4gpu.sh"

# Pre-compilation phase (Score-P instrumented build)
if [ "${SKIP_COMPILE_ENV:-0}" -eq 1 ]; then
    echo "=== Skipping compilation (SKIP_COMPILE_ENV=1) ==="
else
    echo "=== Step 1: Pre-compiling solver with Score-P manual instrumentation into ${BUILD_DIR} ==="
    ./slurm_build.sh "${BUILD_DIR}"
    echo "=== Pre-compilation completed successfully ==="
fi

# Prepare temporary runner script with 40-minute time limit (OTF2 tracing writes)
TEMP_JOB_SCRIPT="$(mktemp .cmi_scorep_96c4g_XXXXXX.sh)"
sed -e 's/^#SBATCH --time=.*/#SBATCH --time=00:40:00/' "${JOB_SCRIPT_TEMPLATE}" > "${TEMP_JOB_SCRIPT}"
chmod +x "${TEMP_JOB_SCRIPT}"
trap 'rm -f "${TEMP_JOB_SCRIPT}"' EXIT

# Base parameters for all single-node 96-core + 4-GPU runs
# Hires configuration (Sep 6 2026): 3840x2160 world, batch 20M (never binds ->
# 1 inference call per rank/range), 202 steps = 101 Regular + 101 ML.
COMMON_EXPORTS="ALL,USE_SCOREP_ENV=1,SCOREP_ENABLE_PROFILING_ENV=true,SKIP_PAPI_METRICS=1,SCOREP_METRIC_PAPI=,MLCOUPLING_MEM_LOG_VERBOSE=1,MODEL_NAME_ENV=watercnn,TOTAL_STEPS_ENV=202,SAVE_EVERY_ENV=202,MPI_RANKS_ENV=96,RANK_GRID_X_ENV=12,RANK_GRID_Z_ENV=8,GPUS_PER_NODE_ENV=4,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=3840,TARGET_HEIGHT_ENV=2160,ML_BATCH_SIZE_ENV=20000000,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${BUILD_DIR},FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1"

PREV_JID="${PREV_JID_ENV:-}"

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
echo "=== Submitting Score-P Suite: 96c4g hires (96-Core + 4-GPU Single Node, 3840x2160 watercnn, bs 20M, 202 steps, profiling+tracing) ==="

# 1. SmartSim Parallel Puts (c=0, DB_LAYOUT=shared, 1 DB node)
submit_suite_job "SP_96c4g_hires_smartsim_c0" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,DB_NODES_ENV=1,SMARTSIM_MPI_SEQUENTIAL_PUT=0,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=96c4g_hires_smartsim_c0"

# 2. SmartSim Per-Node Standalone DB (DB_LAYOUT=per-ml-node, 1 DB node)
submit_suite_job "SP_96c4g_hires_smartsim_per_node_db" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=per-ml-node,DB_NODES_ENV=1,SMARTSIM_MPI_SEQUENTIAL_PUT=0,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=96c4g_hires_smartsim_per_node_db"

# 3. SmartSim Per-GPU DB (DB_LAYOUT=per-ml-node, 4 DB instances each pinned to 1 GPU)
submit_suite_job "SP_96c4g_hires_smartsim_per_gpu_db" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=per-ml-node,DB_NODES_ENV=4,SMARTSIM_MPI_SEQUENTIAL_PUT=0,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=96c4g_hires_smartsim_per_gpu_db"

# 4. SmartSim Sequential Chain 1 (c=1)
submit_suite_job "SP_96c4g_hires_smartsim_c1" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,DB_NODES_ENV=1,SMARTSIM_MPI_SEQUENTIAL_PUT=1,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=96c4g_hires_smartsim_c1"

# 5. SmartSim Sequential Chain 3 (c=3)
submit_suite_job "SP_96c4g_hires_smartsim_c3" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,DB_NODES_ENV=1,SMARTSIM_MPI_SEQUENTIAL_PUT=3,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=96c4g_hires_smartsim_c3"

# 5b. SmartSim RedisAI auto-batching: BATCHSIZE=20M merges queued AI.RUN calls
#     per device queue (model.GPU:N); MINBATCHSIZE=0 -> batch executes
#     immediately with whatever is queued (no timeout needed; RedisAI rejects
#     MINBATCHTIMEOUT without MINBATCHSIZE).
submit_suite_job "SP_96c4g_hires_smartsim_c0_batch20m" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,DB_NODES_ENV=1,SMARTSIM_MPI_SEQUENTIAL_PUT=0,ML_DB_BATCH_SIZE_ENV=20000000,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=96c4g_hires_smartsim_c0_batch20m"

# 5c. SmartSim with RedisAI run-queue THREADS_PER_QUEUE=4 (default was 3 =
#     min(8, 24//8)): more DB worker threads drain the run queue concurrently.
submit_suite_job "SP_96c4g_hires_smartsim_c0_tpq4" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,DB_NODES_ENV=1,SMARTSIM_MPI_SEQUENTIAL_PUT=0,SMARTSIM_THREADS_PER_QUEUE_ENV=4,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=96c4g_hires_smartsim_c0_tpq4"

# 5d. SmartSim combined: RedisAI auto-batching (20M) + THREADS_PER_QUEUE=4.
submit_suite_job "SP_96c4g_hires_smartsim_c0_batch20m_tpq4" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,DB_NODES_ENV=1,SMARTSIM_MPI_SEQUENTIAL_PUT=0,ML_DB_BATCH_SIZE_ENV=20000000,SMARTSIM_THREADS_PER_QUEUE_ENV=4,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=96c4g_hires_smartsim_c0_batch20m_tpq4"

# 6. AIxelerator Collective (4 GPU controllers across 4 GPUs)
submit_suite_job "SP_96c4g_hires_aix_coll" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix.toml,MPI_RANKS_ENV=96,RANK_GRID_X_ENV=12,RANK_GRID_Z_ENV=8,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=96c4g_hires_aix_coll"

# 7. AIxelerator P2P / Pipelined (tracing OFF; CSV P2P timeline instead, OTF2 useless for async)
submit_suite_job "SP_96c4g_hires_aix_p2p" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,MPI_RANKS_ENV=96,RANK_GRID_X_ENV=12,RANK_GRID_Z_ENV=8,AIX_P2P_TIMELINE_ENV=1,SCOREP_DIR_TAG_ENV=96c4g_hires_aix_p2p"

# 8. PhyDLL C++ DL Client (4 DL workers across 4 GPUs)
submit_suite_job "SP_96c4g_hires_phydll_cpp" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll.toml,USE_PYTHON_DL_CLIENT=0,PHYDLL_NP_DL_ENV=4,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=96c4g_hires_phydll_cpp"

# 9. AIxelerator P2P / Pipelined with FULL initial credits (AIX_P2P_INITIAL_CREDITS=23):
#    all 23 workers of each 24-rank workgroup may start sending immediately instead
#    of the default credit=1 trickle (each received input grants +2 more).
#    Fully independent run: own timeline CSVs, profiling-only like case 7.
submit_suite_job "SP_96c4g_hires_aix_p2p_fullcredits" \
    "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,MPI_RANKS_ENV=96,RANK_GRID_X_ENV=12,RANK_GRID_Z_ENV=8,AIX_P2P_TIMELINE_ENV=1,AIX_P2P_INITIAL_CREDITS_ENV=23,SCOREP_DIR_TAG_ENV=96c4g_hires_aix_p2p_fullcredits"

# 10. PhyDLL Python DL Client — EXCLUDED (structural Score-P limitation, Sep 2026):
#    the Python client (standard mpi4py, scorep --mpp=none) cannot join Score-P's
#    world-wide init collectives (PMPI_Comm_dup/Allgather) or the clock ping-pong,
#    so the instrumented solver ranks deadlock inside MPI_Init in profiling AND
#    tracing mode. See the fail-fast guard in single_node_96c_4gpu.sh.
#    The C++ DL client case (SP_96c4g_hires_phydll_cpp) covers the DL-client insight.
# submit_suite_job "SP_96c4g_hires_phydll_py" \
#     "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll.toml,USE_PYTHON_DL_CLIENT=1,PHYDLL_NP_DL_ENV=4,PHYDLL_PY_SCOREP_WRAPPER=1,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=96c4g_hires_phydll_py"

echo ""
echo "=== All 12 96c4g hires Score-P benchmark cases submitted successfully ==="
echo "LAST_JOB_ID=${PREV_JID}"
