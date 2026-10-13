#!/usr/bin/env bash
# Smoke test for the 3 Score-P AIx layout builds (default, affine, numa)
# Non-exclusive, 24 tasks, 1 GPU, 22 steps, account rwth0792, time 00:10:00.
# Tests rpath linking, layout controller selection, and credits=192.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BASE_TEMPLATE="single_node_96c_4gpu.sh"
TEMP_SMOKE_SCRIPT="$(mktemp .hh3_smoke_XXXXXX.sh)"

sed -e 's/^#SBATCH --time=.*/#SBATCH --time=00:10:00/' \
    -e 's/^#SBATCH --ntasks-per-node=.*/#SBATCH --ntasks-per-node=24/' \
    -e 's/^#SBATCH --gres=gpu:.*/#SBATCH --gres=gpu:1/' \
    -e 's/^#SBATCH --exclusive//' \
    "${BASE_TEMPLATE}" > "${TEMP_SMOKE_SCRIPT}"
chmod +x "${TEMP_SMOKE_SCRIPT}"
trap 'rm -f "${TEMP_SMOKE_SCRIPT}"' EXIT

BASE_EXPORTS="ALL,USE_SCOREP_ENV=1,SCOREP_ENABLE_PROFILING_ENV=true,SKIP_PAPI_METRICS=1,SCOREP_METRIC_PAPI=,MLCOUPLING_MEM_LOG_VERBOSE=1,MODEL_NAME_ENV=watercnn,TOTAL_STEPS_ENV=22,SAVE_EVERY_ENV=22,RANK_GRID_X_ENV=,RANK_GRID_Z_ENV=,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=3840,TARGET_HEIGHT_ENV=2160,ML_BATCH_SIZE_ENV=20000000,SKIP_COMPILE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,MPI_RANKS_ENV=24,GPUS_PER_NODE_ENV=1,SOLVER_STEP_GPUS_ENV=1,CUDA_VISIBLE_DEVICES=0"

echo "=== Submitting HH3 Smoke Tests (non-exclusive, 1 GPU, 22 steps) ==="

for layout in default affine numa; do
    job_name="HH3_smoke_${layout}"
    solver_build="build_scorep_cmi_${layout}"
    exports="${BASE_EXPORTS},COMPILE_OUTPUT_PATH_ENV=${solver_build},CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,AIX_P2P_TIMELINE_ENV=1,AIX_P2P_INITIAL_CREDITS_ENV=192,SCOREP_DIR_TAG_ENV=smoke_hethires3_${layout},JOB_NAME_ENV=hh3_smoke_${layout},RUN_ID_ENV=hh3_smoke_${layout}"

    echo "Submitting ${job_name} (${solver_build})..."
    jid=$(sbatch --parsable --job-name="${job_name}" --account=rwth0792 --export="${exports}" "${TEMP_SMOKE_SCRIPT}")
    echo "  -> Job ID: ${jid}"
    if [ -x "${HOME}/scripts/swatch" ]; then
        "${HOME}/scripts/swatch" --jobid "${jid}" 2>/dev/null || true
    fi
done

echo "=== Smoke tests submitted successfully ==="
