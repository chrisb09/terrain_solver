#!/usr/bin/env bash
# Smoke-test the two new AIx installs (GPU-affine, NUMA-local) on a 96c4g node
# with a very short WaterCNN run. Verifies that the service starts, that the
# GPU controller selection found the GPUs via lstopo, and that the group
# placement prints look as expected, before spending the full 3x3 matrix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

JOB_SCRIPT="single_node_96c_4gpu.sh"
AIX_ROOT="${SCRIPT_DIR}/../CPP-ML-Interface/extern/AIxeleratorService"

COMMON_EXPORTS="ALL,USE_SCOREP_ENV=0,SKIP_PAPI_METRICS=1,SCOREP_METRIC_PAPI=,MLCOUPLING_MEM_LOG_VERBOSE=0,MODEL_NAME_ENV=watercnn,TOTAL_STEPS_ENV=5,SAVE_EVERY_ENV=5,MPI_RANKS_ENV=96,RANK_GRID_X_ENV=12,RANK_GRID_Z_ENV=8,GPUS_PER_NODE_ENV=4,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=3840,TARGET_HEIGHT_ENV=2160,ML_BATCH_SIZE_ENV=20000000,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,USE_SMARTSIM=0,USE_CPP_ML_INTERFACE=1,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix.toml,SOLVER_CPU_BIND_ENV=cores"

echo "=== Submitting AIx layout smoke tests (5 steps, 96c4g) ==="

AFFINE_JID=$(sbatch --parsable --job-name=smoke_aix_affine --account=rwth0792 --time=00:30:00 \
    --export="${COMMON_EXPORTS},AIX_INSTALL_PREFIX_ENV=${AIX_ROOT}/INSTALL-AFFINE,COMPILE_OUTPUT_PATH_ENV=build_smoke_affine,SKIP_COMPILE_ENV=0" \
    "${JOB_SCRIPT}")
echo "GPU-affine smoke: ${AFFINE_JID}"

NUMA_JID=$(sbatch --parsable --job-name=smoke_aix_numa --account=rwth0792 --time=00:30:00 \
    --export="${COMMON_EXPORTS},AIX_INSTALL_PREFIX_ENV=${AIX_ROOT}/INSTALL-NUMA,COMPILE_OUTPUT_PATH_ENV=build_smoke_numa,SKIP_COMPILE_ENV=0" \
    "${JOB_SCRIPT}")
echo "NUMA-local smoke: ${NUMA_JID}"

echo ""
echo "Logs: logs/mini_app_output_${AFFINE_JID}.txt, logs/mini_app_output_${NUMA_JID}.txt"
echo "Look for: 'is GPU controller for GPU' and 'NUMA-local assignment' / 'will be in group'."
