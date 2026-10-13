#!/usr/bin/env bash
# Het (c23mm + c23g) hires suite: 6 resource scenarios x 6-7 library configs.
#
# Scenario notation A+B (cores), G GPUs:
#   A = c23mm solver ranks (CPU-only memory nodes)
#   B = c23g solver/worker ranks
#   G = GPUs on the c23g node(s)
# AIX: the c23g group hosts max(B, 1) solver ranks; on any node with k GPUs,
#      k of its ranks auto-become GPU controllers (roundRobinDistribution.cpp:
#      node_rank < cudaGetDeviceCount()).
# SmartSim/PhyDLL: c23g group hosts B solver ranks (0 = only DB / DL workers).
# DL workers (PhyDLL) = G, one per GPU.
#
# Scenarios:
#   s1: 0+4,  1 GPU (c23g-only, account rwth0792)
#   s2: 4+0,  1 GPU (thes2181)
#   s3: 24+0, 1 GPU (thes2181)
#   s4: 24+24, 1 GPU (thes2181)
#   s5: 0+96, 4 GPUs (c23g-only, rwth0792; re-run of the 96c4g hires suite,
#                    now with --exclusive)
#   s6: 96+96, 4 GPUs (thes2181)
#
# All jobs are exclusive and chained (dependency=afterany). A=0 scenarios
# reuse single_node_96c_4gpu.sh; the others use het_hires_job.sh.
#
# Usage: SKIP_COMPILE_ENV=1 ./run_cmi_scorep_het_hires_suite.sh

# Exit on error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build_scorep_cmi"
HET_TEMPLATE="het_hires_job.sh"
SINGLE_TEMPLATE="single_node_96c_4gpu.sh"

# Pre-compilation phase (Score-P instrumented build)
if [ "${SKIP_COMPILE_ENV:-0}" -eq 1 ]; then
    echo "=== Skipping compilation (SKIP_COMPILE_ENV=1) ==="
else
    echo "=== Step 1: Pre-compiling solver with Score-P manual instrumentation into ${BUILD_DIR} ==="
    ./slurm_build.sh "${BUILD_DIR}"
    echo "=== Pre-compilation completed successfully ==="
fi

# id|mm_tasks|g_side_solver_tasks|gpus|account|time_limit
# g_side_solver_tasks is the c23g het-group allocation that carries SOLVER
# ranks (AIX: max(B,1); smartsim/phydll: B). Additional DB/DL capacity rides
# on the same group (DB as background/srun step, DL via --oversubscribe).
SCENARIOS=(
    "5|0|96|4|rwth0792|01:00:00"
    "1|0|4|1|rwth0792|02:00:00"
    "2|4|0|1|thes2181|02:00:00"
    "3|24|0|1|thes2181|02:00:00"
    "4|24|24|1|thes2181|02:00:00"
    "6|96|96|4|thes2181|01:00:00"
)

PREV_JID="${PREV_JID_ENV:-}"

submit_job() {
    local job_script="$1"
    local job_name="$2"
    local account="$3"
    local full_exports="$4"

    local sbatch_cmd="sbatch --parsable --job-name=${job_name} --account=${account} --export=${full_exports}"
    if [ -n "${PREV_JID}" ]; then
        sbatch_cmd="${sbatch_cmd} --dependency=afterany:${PREV_JID}"
    fi

    echo "Submitting ${job_name} (${job_script}, account ${account})..."
    local jid
    jid=$(eval ${sbatch_cmd} "${job_script}")

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

# --- helpers to build per-case temp job scripts ---

make_het_script() {
    # $1 = output path, $2 = mm tasks, $3 = g tasks, $4 = g cpus-per-task,
    # $5 = gpus, $6 = time limit
    sed -e "s/^#SBATCH --time=.*/#SBATCH --time=$6/" \
        -e "s/^#SBATCH --ntasks-per-node=__HET_MM_TASKS__/#SBATCH --ntasks-per-node=$2/" \
        -e "s/^#SBATCH --ntasks-per-node=__HET_G_TASKS__/#SBATCH --ntasks-per-node=$3/" \
        -e "s/^#SBATCH --cpus-per-task=__HET_G_CPUS__/#SBATCH --cpus-per-task=$4/" \
        -e "s/^#SBATCH --gres=gpu:__HET_G_GPUS__/#SBATCH --gres=gpu:$5/" \
        "${HET_TEMPLATE}" > "$1"
    chmod +x "$1"
}

make_single_script() {
    # $1 = output path, $2 = ntasks, $3 = gpus, $4 = time limit
    sed -e "s/^#SBATCH --time=.*/#SBATCH --time=$4/" \
        -e "s/^#SBATCH --ntasks-per-node=96/#SBATCH --ntasks-per-node=$2/" \
        -e "s/^#SBATCH --gres=gpu:4/#SBATCH --gres=gpu:$3/" \
        -e "s/^##SBATCH --exclusive/#SBATCH --exclusive/" \
        "${SINGLE_TEMPLATE}" > "$1"
    chmod +x "$1"
}

# Base exports for all runs (hires configuration: 3840x2160 watercnn, batch
# 20M, 202 steps = 101 Regular + 101 ML). Rank grid left empty (auto).
BASE_EXPORTS="ALL,USE_SCOREP_ENV=1,SCOREP_ENABLE_PROFILING_ENV=true,SKIP_PAPI_METRICS=1,SCOREP_METRIC_PAPI=,MLCOUPLING_MEM_LOG_VERBOSE=1,MODEL_NAME_ENV=watercnn,TOTAL_STEPS_ENV=202,SAVE_EVERY_ENV=202,RANK_GRID_X_ENV=,RANK_GRID_Z_ENV=,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=3840,TARGET_HEIGHT_ENV=2160,ML_BATCH_SIZE_ENV=20000000,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${BUILD_DIR},FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1"

echo ""
echo "=== Submitting Score-P Het Hires Suite (exclusive, chained; 6 scenarios) ==="

for scenario in "${SCENARIOS[@]}"; do
    IFS='|' read -r S_ID S_MM S_GTASKS S_GPUS S_ACCOUNT S_TIME <<< "${scenario}"
    echo ""
    echo "### Scenario s${S_ID}: ${S_MM} c23mm + ${S_GTASKS} c23g solver tasks, ${S_GPUS} GPU(s), ${S_ACCOUNT}, ${S_TIME}"

    if [[ "${S_MM}" -gt 0 ]]; then
        # Het submission via het_hires_job.sh
        # c23g allocation: solver tasks (1 CPU each) unless the group carries
        # only DB/DL/controller (then give that task the inference CPU budget).
        # The allocation always has >= 1 task (hosts AIX controller / DB / DL
        # workers even when B=0).
        g_alloc=$(( S_GTASKS > 0 ? S_GTASKS : 1 ))
        g_cpus=1
        if [[ "${S_GTASKS}" -le 1 ]]; then
            g_cpus=24
        fi
        tmp_script="$(mktemp .cmi_het_hires_s${S_ID}_XXXXXX.sh)"
        make_het_script "${tmp_script}" "${S_MM}" "${g_alloc}" "${g_cpus}" "${S_GPUS}" "${S_TIME}"
        # clean up on scenario loop end

        case_specific() {
            submit_job "${tmp_script}" "$1" "${S_ACCOUNT}" "${BASE_EXPORTS},$2"
        }

        # 1. AIxelerator Collective (auto controllers on the GPU node)
        case_specific "HH_s${S_ID}_aix_coll" \
            "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix.toml,HET_G_SIDE_TASKS_ENV=${g_alloc},SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=hethires_s${S_ID}_aix_coll"

        # 2. AIxelerator P2P / Pipelined (tracing OFF; CSV P2P timeline instead)
        case_specific "HH_s${S_ID}_aix_p2p" \
            "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,HET_G_SIDE_TASKS_ENV=${g_alloc},AIX_P2P_TIMELINE_ENV=1,SCOREP_DIR_TAG_ENV=hethires_s${S_ID}_aix_p2p"

        # 3. AIxelerator P2P with FULL initial credits
        case_specific "HH_s${S_ID}_aix_p2p_fullcredits" \
            "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,HET_G_SIDE_TASKS_ENV=${g_alloc},AIX_P2P_TIMELINE_ENV=1,AIX_P2P_INITIAL_CREDITS_ENV=23,SCOREP_DIR_TAG_ENV=hethires_s${S_ID}_aix_p2p_fullcredits"

        # 4. SmartSim Parallel Puts (c=0, shared DB)
        case_specific "HH_s${S_ID}_smartsim_c0" \
            "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,DB_NODES_ENV=1,SMARTSIM_MPI_SEQUENTIAL_PUT=0,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=hethires_s${S_ID}_smartsim_c0"

        # 5. SmartSim Sequential Chain 3 (c=3)
        case_specific "HH_s${S_ID}_smartsim_c3" \
            "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,DB_NODES_ENV=1,SMARTSIM_MPI_SEQUENTIAL_PUT=3,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=hethires_s${S_ID}_smartsim_c3"

        # 6. PhyDLL C++ DL Client (G DL workers, one per GPU)
        phydll_dl_over=0
        if [[ "${S_GTASKS}" -gt 0 ]]; then
            phydll_dl_over=1
        fi
        case_specific "HH_s${S_ID}_phydll_cpp" \
            "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll.toml,USE_PYTHON_DL_CLIENT=0,PHYDLL_NP_DL_ENV=${S_GPUS},PHYDLL_DL_OVERSUBSCRIBE_ENV=${phydll_dl_over},SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=hethires_s${S_ID}_phydll_cpp"

        # 7. Per-GPU SmartSim DB only for the 4-GPU scenarios (with 1 GPU it
        #    is identical to the shared-DB c0 case).
        if [[ "${S_GPUS}" -ge 4 ]]; then
            case_specific "HH_s${S_ID}_smartsim_per_gpu_db" \
                "USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=per-ml-node,DB_NODES_ENV=4,SMARTSIM_MPI_SEQUENTIAL_PUT=0,SMARTSIM_DB_GPU_COUNT_ENV=${S_GPUS},SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=hethires_s${S_ID}_smartsim_per_gpu_db"
        fi

        rm -f "${tmp_script}"
    else
        # A=0 scenarios: single-allocation c23g job via single_node_96c_4gpu.sh
        tmp_script="$(mktemp .cmi_het_hires_s${S_ID}_XXXXXX.sh)"
        make_single_script "${tmp_script}" "${S_GTASKS}" "${S_GPUS}" "${S_TIME}"

        single_common="USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,MPI_RANKS_ENV=${S_GTASKS},GPUS_PER_NODE_ENV=${S_GPUS}"

        submit_job "${tmp_script}" "HH_s${S_ID}_aix_coll" "${S_ACCOUNT}" \
            "${BASE_EXPORTS},${single_common},CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix.toml,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=hethires_s${S_ID}_aix_coll"

        submit_job "${tmp_script}" "HH_s${S_ID}_aix_p2p" "${S_ACCOUNT}" \
            "${BASE_EXPORTS},${single_common},CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,AIX_P2P_TIMELINE_ENV=1,SCOREP_DIR_TAG_ENV=hethires_s${S_ID}_aix_p2p"

        submit_job "${tmp_script}" "HH_s${S_ID}_aix_p2p_fullcredits" "${S_ACCOUNT}" \
            "${BASE_EXPORTS},${single_common},CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,AIX_P2P_TIMELINE_ENV=1,AIX_P2P_INITIAL_CREDITS_ENV=23,SCOREP_DIR_TAG_ENV=hethires_s${S_ID}_aix_p2p_fullcredits"

        submit_job "${tmp_script}" "HH_s${S_ID}_smartsim_c0" "${S_ACCOUNT}" \
            "${BASE_EXPORTS},${single_common},CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,DB_NODES_ENV=1,SMARTSIM_MPI_SEQUENTIAL_PUT=0,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=hethires_s${S_ID}_smartsim_c0"

        submit_job "${tmp_script}" "HH_s${S_ID}_smartsim_c3" "${S_ACCOUNT}" \
            "${BASE_EXPORTS},${single_common},CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,DB_NODES_ENV=1,SMARTSIM_MPI_SEQUENTIAL_PUT=3,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=hethires_s${S_ID}_smartsim_c3"

        submit_job "${tmp_script}" "HH_s${S_ID}_phydll_cpp" "${S_ACCOUNT}" \
            "${BASE_EXPORTS},${single_common},CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll.toml,USE_PYTHON_DL_CLIENT=0,PHYDLL_NP_DL_ENV=${S_GPUS},SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=hethires_s${S_ID}_phydll_cpp"

        if [[ "${S_GPUS}" -ge 4 ]]; then
            submit_job "${tmp_script}" "HH_s${S_ID}_smartsim_per_gpu_db" "${S_ACCOUNT}" \
                "${BASE_EXPORTS},${single_common},CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=per-ml-node,DB_NODES_ENV=4,SMARTSIM_MPI_SEQUENTIAL_PUT=0,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=hethires_s${S_ID}_smartsim_per_gpu_db"
        fi

        rm -f "${tmp_script}"
    fi
done

echo ""
echo "=== All het hires Score-P benchmark cases submitted successfully ==="
echo "LAST_JOB_ID=${PREV_JID}"
