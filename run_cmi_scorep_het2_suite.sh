#!/usr/bin/env bash
# HH2 het hires suite: ONE job per library-config (not per scenario).
#
# Two allocation classes, each job runs several scenarios sequentially inside
# one exclusive allocation (steps request <= allocation):
#
#   G-class (c23g-only, account rwth0792, 1 node 96x1cpu + gpu:4, 02:00:00)
#     runs s1 (0+4, 1 GPU), s3 (0+24, 1 GPU), s6 (0+96, 4 GPUs)
#   H-class (het, account thes2181, 96 c23mm + 96 c23g + gpu:4, 02:30:00)
#     runs s2 (4+0, 1 GPU), s4 (24+0, 1 GPU), s5 (24+24, 1 GPU), s7 (96+96, 4 GPUs)
#
# Scenario numbering: s1=0+4, s2=4+0, s3=0+24(new), s4=24+0, s5=24+24,
# s6=0+96, s7=96+96. All jobs are submitted INDEPENDENTLY (parallel), with
# per-scenario output isolation via unique JOB_NAME/RUN_ID/Score-P tags.
#
# Usage: SKIP_COMPILE_ENV=1 ./run_cmi_scorep_het2_suite.sh [configs...]
#        (default: all 7 configs; pass a subset to run fewer)

# Exit on error
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BUILD_DIR="build_scorep_cmi"
G_WRAPPER="hh2_wrapper_G.sh"
H_WRAPPER="hh2_wrapper_H.sh"

if [ "${SKIP_COMPILE_ENV:-0}" -eq 1 ]; then
    echo "=== Skipping compilation (SKIP_COMPILE_ENV=1) ==="
else
    echo "=== Compiling solver into ${BUILD_DIR} ==="
    ./slurm_build.sh "${BUILD_DIR}"
fi

# Base exports (hires: 3840x2160 watercnn, batch 20M, 202 steps, auto grid).
# Scenario-specific values (MPI_RANKS, GPUS_PER_NODE, Score-P tags, DB
# layout, port offsets) are set by the wrappers per scenario.
BASE_EXPORTS="ALL,USE_SCOREP_ENV=1,SCOREP_ENABLE_PROFILING_ENV=true,SKIP_PAPI_METRICS=1,SCOREP_METRIC_PAPI=,MLCOUPLING_MEM_LOG_VERBOSE=1,MODEL_NAME_ENV=watercnn,TOTAL_STEPS_ENV=202,SAVE_EVERY_ENV=202,RANK_GRID_X_ENV=,RANK_GRID_Z_ENV=,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=3840,TARGET_HEIGHT_ENV=2160,ML_BATCH_SIZE_ENV=20000000,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${BUILD_DIR},FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1"

declare -A CONFIG_EXPORTS=(
    [aix_coll]="CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix.toml,SCOREP_ENABLE_TRACING_ENV=true"
    [aix_p2p]="CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,AIX_P2P_TIMELINE_ENV=1"
    [aix_p2p_fullcredits]="CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,AIX_P2P_TIMELINE_ENV=1,AIX_P2P_INITIAL_CREDITS_ENV=23"
    [smartsim_c0]="CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,SMARTSIM_MPI_SEQUENTIAL_PUT=0,SCOREP_ENABLE_TRACING_ENV=true"
    [smartsim_c3]="CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,SMARTSIM_MPI_SEQUENTIAL_PUT=3,SCOREP_ENABLE_TRACING_ENV=true"
    [smartsim_per_gpu_db]="CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=per-ml-node,DB_NODES_ENV=4,SMARTSIM_MPI_SEQUENTIAL_PUT=0,SMARTSIM_DB_GPU_COUNT_ENV=4,SCOREP_ENABLE_TRACING_ENV=true"
    [phydll_cpp]="CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll.toml,USE_PYTHON_DL_CLIENT=0,SCOREP_ENABLE_TRACING_ENV=true"
)
ALL_CONFIGS=(aix_coll aix_p2p aix_p2p_fullcredits smartsim_c0 smartsim_c3 phydll_cpp smartsim_per_gpu_db)

if [ $# -gt 0 ]; then
    SELECTED=("$@")
else
    SELECTED=("${ALL_CONFIGS[@]}")
fi

echo ""
echo "=== Submitting HH2 suite: ${#SELECTED[@]} G-class + (n-1) H-class parallel jobs ==="

for cfg in "${SELECTED[@]}"; do
    exports="${BASE_EXPORTS},${CONFIG_EXPORTS[${cfg}]},HH2_CONFIG_ENV=${cfg}"

    # G-class job: c23g-only (rwth0792)
    echo "Submitting HH2_G_${cfg}..."
    jid=$(sbatch --parsable --job-name="HH2_G_${cfg}" --account=rwth0792 \
        --export="${exports}" "${G_WRAPPER}")
    echo "  -> Job ID: ${jid}"
    [ -x "${HOME}/scripts/swatch" ] && "${HOME}/scripts/swatch" --jobid "${jid}" 2>/dev/null || true

    # H-class job (all configs except per_gpu_db, which is G-class-only)
    if [[ "${cfg}" != "smartsim_per_gpu_db" ]]; then
        echo "Submitting HH2_H_${cfg}..."
        jid=$(sbatch --parsable --job-name="HH2_H_${cfg}" --account=thes2181 \
            --export="${exports}" "${H_WRAPPER}")
        echo "  -> Job ID: ${jid}"
        [ -x "${HOME}/scripts/swatch" ] && "${HOME}/scripts/swatch" --jobid "${jid}" 2>/dev/null || true
    fi
done

echo ""
echo "=== HH2 submissions complete ==="
