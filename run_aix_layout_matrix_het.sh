#!/usr/bin/env bash
# 3x3 AIx rank-layout matrix for the 96+96c4g heterogeneous configuration:
#   96 c23mm solver ranks (CPU-only) + 96 c23g solver ranks (4 GPUs),
#   account thes2181. Same layouts/comms as run_aix_layout_matrix.sh:
#   layouts: default (RoundRobin), affine (WITH_HWLOC), numa (WITH_NUMA_LOCAL_GROUPS)
#   comms:   collective, p2p credit=1, p2p credit=23
# Each layout is a chain: the collective job compiles the solver against that
# layout's AIx install; the two P2P jobs reuse the build. Three chains in
# parallel. All runs use --cpu-bind=cores for deterministic placement.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

JOB_TEMPLATE="het_hires_job.sh"
AIX_ROOT="${SCRIPT_DIR}/../CPP-ML-Interface/extern/AIxeleratorService"
TIME_LIMIT="${TIME_LIMIT_ENV:-01:00:00}"
HET_MM_TASKS="${HET_MM_TASKS_ENV:-96}"
HET_G_TASKS="${HET_G_TASKS_ENV:-96}"
HET_G_CPUS="${HET_G_CPUS_ENV:-1}"
HET_G_GPUS="${HET_G_GPUS_ENV:-4}"
DEPEND_JID="${DEPEND_JID_ENV:-}"

COMMON_EXPORTS="ALL,USE_SCOREP_ENV=0,SKIP_PAPI_METRICS=1,SCOREP_METRIC_PAPI=,MLCOUPLING_MEM_LOG_VERBOSE=0,MODEL_NAME_ENV=watercnn,TOTAL_STEPS_ENV=202,SAVE_EVERY_ENV=202,GPUS_PER_NODE_ENV=4,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=3840,TARGET_HEIGHT_ENV=2160,ML_BATCH_SIZE_ENV=20000000,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,SOLVER_CPU_BIND_ENV=cores,HET_G_SIDE_TASKS_ENV=${HET_G_TASKS},AIX_P2P_TIMELINE_ENV=0,APPLY_CUDA_STUBS_ENV=1"

make_het_script() {
    local output="$1"
    sed -e "s/^#SBATCH --time=.*/#SBATCH --time=${TIME_LIMIT}/" \
        -e "s/__HET_MM_TASKS__/${HET_MM_TASKS}/" \
        -e "s/__HET_G_TASKS__/${HET_G_TASKS}/" \
        -e "s/__HET_G_CPUS__/${HET_G_CPUS}/" \
        -e "s/__HET_G_GPUS__/${HET_G_GPUS}/" \
        "${JOB_TEMPLATE}" > "${output}"
    chmod +x "${output}"
}

submit_chain() {
    local layout="$1"
    local install_prefix="$2"
    local build_dir="$3"
    local base="${COMMON_EXPORTS},AIX_INSTALL_PREFIX_ENV=${install_prefix},COMPILE_OUTPUT_PATH_ENV=${build_dir}"

    local tmp_script
    tmp_script="$(mktemp ".het_matrix_${layout}_XXXXXX.sh")"
    make_het_script "${tmp_script}"

    local coll_jid p2p1_jid p2p23_jid
    local coll_dep=()
    if [[ -n "${DEPEND_JID}" ]]; then
        coll_dep=("--dependency=afterok:${DEPEND_JID}")
    fi
    coll_jid=$(sbatch --parsable --job-name="MXH_${layout}_coll" --account=thes2181 "${coll_dep[@]}" \
        --export="${base},CPP_ML_CONFIG_ENV=config_aix.toml,SKIP_COMPILE_ENV=0" \
        "${tmp_script}")
    p2p1_jid=$(sbatch --parsable --job-name="MXH_${layout}_p2p1" --account=thes2181 \
        --dependency="afterok:${coll_jid}" \
        --export="${base},CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,AIX_P2P_INITIAL_CREDITS_ENV=1,SKIP_COMPILE_ENV=1" \
        "${tmp_script}")
    p2p23_jid=$(sbatch --parsable --job-name="MXH_${layout}_p2p23" --account=thes2181 \
        --dependency="afterok:${coll_jid}" \
        --export="${base},CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,AIX_P2P_INITIAL_CREDITS_ENV=23,SKIP_COMPILE_ENV=1" \
        "${tmp_script}")

    rm -f "${tmp_script}"
    echo "layout=${layout}: collective=${coll_jid}  p2p_credit1=${p2p1_jid}  p2p_credit23=${p2p23_jid}"
}

echo "=== Submitting 3x3 AIx layout matrix, 96 c23mm + 96 c23g c4g ($(date)) ==="
submit_chain default "${AIX_ROOT}/INSTALL"        build_het_matrix_default
submit_chain affine  "${AIX_ROOT}/INSTALL-AFFINE" build_het_matrix_affine
submit_chain numa    "${AIX_ROOT}/INSTALL-NUMA"   build_het_matrix_numa
echo ""
echo "Logs: logs/mini_app_output_<jobid>.txt (STEP_TIMING lines)."
echo "Analyze: python3 analyze_matrix_results.py --config-label 96p96c4g_c23mm_c23g \\"
echo "           --job-map 'default_coll=...,...' \\"
echo "           --out-csv results/aix_layout_matrix/96p96c4g/aix_layout_matrix_96p96c4g_c23mm_c23g_summary.csv"
