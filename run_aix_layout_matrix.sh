#!/usr/bin/env bash
# 3x3 AIx rank-layout matrix for the 96c4g WaterCNN production configuration:
#   layouts: default (RoundRobin / strided), gpu-affine (WITH_HWLOC),
#            numa-local (WITH_HWLOC + WITH_NUMA_LOCAL_GROUPS)
#   comms:   collective, p2p credit=1, p2p credit=23 (full credits)
# Each layout is a chain: the collective job compiles the solver against that
# layout's AIx install; the two P2P jobs reuse the build. The three chains run
# in parallel. All runs use --cpu-bind=cores for deterministic rank placement.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

JOB_SCRIPT="single_node_96c_4gpu.sh"
AIX_ROOT="${SCRIPT_DIR}/../CPP-ML-Interface/extern/AIxeleratorService"
TIME_LIMIT="${TIME_LIMIT_ENV:-00:45:00}"

COMMON_EXPORTS="ALL,USE_SCOREP_ENV=0,SKIP_PAPI_METRICS=1,SCOREP_METRIC_PAPI=,MLCOUPLING_MEM_LOG_VERBOSE=0,MODEL_NAME_ENV=watercnn,TOTAL_STEPS_ENV=202,SAVE_EVERY_ENV=202,MPI_RANKS_ENV=96,RANK_GRID_X_ENV=12,RANK_GRID_Z_ENV=8,GPUS_PER_NODE_ENV=4,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=3840,TARGET_HEIGHT_ENV=2160,ML_BATCH_SIZE_ENV=20000000,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,SOLVER_CPU_BIND_ENV=cores,AIX_P2P_TIMELINE_ENV=0"

submit_chain() {
    local layout="$1"
    local install_prefix="$2"
    local build_dir="$3"
    local base="${COMMON_EXPORTS},AIX_INSTALL_PREFIX_ENV=${install_prefix},COMPILE_OUTPUT_PATH_ENV=${build_dir}"

    local coll_jid p2p1_jid p2p23_jid
    coll_jid=$(sbatch --parsable --job-name="MX_${layout}_coll" --account=rwth0792 --time="${TIME_LIMIT}" \
        --export="${base},CPP_ML_CONFIG_ENV=config_aix.toml,SKIP_COMPILE_ENV=0" \
        "${JOB_SCRIPT}")
    p2p1_jid=$(sbatch --parsable --job-name="MX_${layout}_p2p1" --account=rwth0792 --time="${TIME_LIMIT}" \
        --dependency="afterok:${coll_jid}" \
        --export="${base},CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,AIX_P2P_INITIAL_CREDITS_ENV=1,SKIP_COMPILE_ENV=1" \
        "${JOB_SCRIPT}")
    p2p23_jid=$(sbatch --parsable --job-name="MX_${layout}_p2p23" --account=rwth0792 --time="${TIME_LIMIT}" \
        --dependency="afterok:${coll_jid}" \
        --export="${base},CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,AIX_P2P_INITIAL_CREDITS_ENV=23,SKIP_COMPILE_ENV=1" \
        "${JOB_SCRIPT}")

    echo "layout=${layout}: collective=${coll_jid}  p2p_credit1=${p2p1_jid}  p2p_credit23=${p2p23_jid}"
}

echo "=== Submitting 3x3 AIx layout matrix ($(date)) ==="
submit_chain default "${AIX_ROOT}/INSTALL"        build_matrix_default
submit_chain affine  "${AIX_ROOT}/INSTALL-AFFINE" build_matrix_affine
submit_chain numa    "${AIX_ROOT}/INSTALL-NUMA"   build_matrix_numa
echo ""
echo "Logs: logs/mini_app_output_<jobid>.txt (STEP_TIMING lines)."
echo "Compare with: python3 compare_step_timing.py <logs...>"
