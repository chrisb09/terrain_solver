#!/usr/bin/env bash
# HH3 Suite Submission Script:
#
# Part 1 (Base batch, s1-s7):
#   AIx layout cube (3 configs x 3 layouts x 2 classes = 18 jobs)
#   + s7 smartsim_per_gpu_db (1 H job)
#   (Disabled by default via SUBMIT_BASE_ENV=0 since jobs 4230067.. were already submitted)
#
# Part 2 (Scale batch, s8-s11):
#   Bundled 17-node het allocation (16 c23mm + 1 c23g, exclusive, thes2181)
#   Runs s8 (2:1), s9 (4:1), s10 (8:1), s11 (16:1) sequentially inside each job.
#   AIx cube: 3 configs x 3 layouts = 9 jobs (time 01:00:00)
#   Non-AIx: smartsim_c0, smartsim_c3, smartsim_per_gpu_db, phydll_cpp = 4 jobs (time 01:30:00)
#   Total scale jobs: 13
#
# Usage:
#   ./run_cmi_scorep_het3_suite.sh                    # Submits scale batch (13 jobs)
#   SUBMIT_BASE_ENV=1 ./run_cmi_scorep_het3_suite.sh  # Also submits base batch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

G_WRAPPER="hh3_wrapper_G.sh"
H_WRAPPER="hh3_wrapper_H.sh"
H_SCALE_WRAPPER="hh3_wrapper_H_scale.sh"

SUBMIT_BASE="${SUBMIT_BASE_ENV:-0}"
SUBMIT_SCALE="${SUBMIT_SCALE_ENV:-1}"

BASE_EXPORTS="ALL,USE_SCOREP_ENV=1,SCOREP_ENABLE_PROFILING_ENV=true,SKIP_PAPI_METRICS=1,SCOREP_METRIC_PAPI=,MLCOUPLING_MEM_LOG_VERBOSE=1,MODEL_NAME_ENV=watercnn,TOTAL_STEPS_ENV=202,SAVE_EVERY_ENV=202,RANK_GRID_X_ENV=,RANK_GRID_Z_ENV=,DB_NODES_ENV=1,ML_INFERENCE_CPU_CORES_ENV=24,TARGET_WIDTH_ENV=3840,TARGET_HEIGHT_ENV=2160,ML_BATCH_SIZE_ENV=20000000,SKIP_COMPILE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1"

declare -A AIX_CONFIG_EXPORTS=(
    [aix_coll]="CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix.toml,SCOREP_ENABLE_TRACING_ENV=true"
    [aix_p2p]="CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,AIX_P2P_TIMELINE_ENV=1"
    [aix_p2p_fullcredits]="CPP_ML_INTERFACE_PROVIDER_ENV=AIX,CPP_ML_CONFIG_ENV=config_aix_pipelined.toml,AIX_P2P_TIMELINE_ENV=1,AIX_P2P_INITIAL_CREDITS_ENV=192"
)

declare -A NON_AIX_CONFIG_EXPORTS=(
    [smartsim_c0]="CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,SMARTSIM_MPI_SEQUENTIAL_PUT=0,SCOREP_ENABLE_TRACING_ENV=true"
    [smartsim_c3]="CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=shared,SMARTSIM_MPI_SEQUENTIAL_PUT=3,SCOREP_ENABLE_TRACING_ENV=true"
    [smartsim_per_gpu_db]="CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config.toml,DB_LAYOUT_ENV=per-ml-node,DB_NODES_ENV=4,SMARTSIM_MPI_SEQUENTIAL_PUT=0,SMARTSIM_DB_GPU_COUNT_ENV=4,SCOREP_ENABLE_TRACING_ENV=true"
    [phydll_cpp]="CPP_ML_INTERFACE_PROVIDER_ENV=PHYDLL,CPP_ML_CONFIG_ENV=config_phydll.toml,USE_PYTHON_DL_CLIENT=0,SCOREP_ENABLE_TRACING_ENV=true"
)

AIX_CONFIGS=(aix_coll aix_p2p aix_p2p_fullcredits)
AIX_LAYOUTS=(default affine numa)
NON_AIX_CONFIGS=(smartsim_c0 smartsim_c3 smartsim_per_gpu_db phydll_cpp)

# Filter functions
want_config() {
    local c="$1"
    [[ "${RUN_CONFIGS_ENV:-all}" == "all" ]] || [[ ",${RUN_CONFIGS_ENV}," == *",${c},"* ]]
}

want_layout() {
    local l="$1"
    [[ "${RUN_LAYOUTS_ENV:-all}" == "all" ]] || [[ ",${RUN_LAYOUTS_ENV}," == *",${l},"* ]]
}

# ==============================================================================
# Part 1: Base Batch (s1 - s7)
# ==============================================================================
if [[ "${SUBMIT_BASE}" == "1" ]]; then
    echo "=== Submitting HH3 Base Batch (18 AIx jobs + 1 s7 per_gpu_db) ==="
    for layout in "${AIX_LAYOUTS[@]}"; do
        want_layout "${layout}" || continue
        for cfg in "${AIX_CONFIGS[@]}"; do
            want_config "${cfg}" || continue
            exports="${BASE_EXPORTS},${AIX_CONFIG_EXPORTS[${cfg}]},HH3_CONFIG_ENV=${cfg},HH3_LAYOUT_ENV=${layout}"

            job_g="HH3_G_${cfg}_${layout}"
            echo "Submitting ${job_g}..."
            jid_g=$(sbatch --parsable --job-name="${job_g}" --account=rwth0792 --time=00:12:00 --export="${exports}" "${G_WRAPPER}" | tail -n1 | cut -d';' -f1)
            echo "  -> Job ID: ${jid_g}"
            [ -x "${HOME}/scripts/swatch" ] && "${HOME}/scripts/swatch" --jobid "${jid_g}" 2>/dev/null || true

            job_h="HH3_H_${cfg}_${layout}"
            echo "Submitting ${job_h}..."
            jid_h=$(sbatch --parsable --job-name="${job_h}" --account=thes2181 --time=00:15:00 --export="${exports}" "${H_WRAPPER}" | tail -n1 | cut -d';' -f1)
            echo "  -> Job ID: ${jid_h}"
            [ -x "${HOME}/scripts/swatch" ] && "${HOME}/scripts/swatch" --jobid "${jid_h}" 2>/dev/null || true
        done
    done

    if want_config "smartsim_per_gpu_db"; then
        ss_exports="${BASE_EXPORTS},${NON_AIX_CONFIG_EXPORTS[smartsim_per_gpu_db]},HH3_CONFIG_ENV=smartsim_per_gpu_db,HH3_LAYOUT_ENV=default,HH3_SCENARIOS_ENV=7"
        job_ss="HH3_H_smartsim_per_gpu_db_s7"
        echo "Submitting ${job_ss}..."
        jid_ss=$(sbatch --parsable --job-name="${job_ss}" --account=thes2181 --time=00:20:00 --export="${ss_exports}" "${H_WRAPPER}" | tail -n1 | cut -d';' -f1)
        echo "  -> Job ID: ${jid_ss}"
        [ -x "${HOME}/scripts/swatch" ] && "${HOME}/scripts/swatch" --jobid "${jid_ss}" 2>/dev/null || true
    fi
fi

# ==============================================================================
# Part 2: Scale Batch (Bundled s8 - s11: 16 c23mm + 1 c23g nodes, 17 nodes total)
# ==============================================================================
if [[ "${SUBMIT_SCALE}" == "1" ]]; then
    echo ""
    echo "=== Submitting HH3 Scale Batch (Bundled 17-node het: s8, s9, s10, s11) ==="

    # 1. AIx Cube (3 configs x 3 layouts = 9 bundled jobs, time 01:00:00)
    for layout in "${AIX_LAYOUTS[@]}"; do
        want_layout "${layout}" || continue
        for cfg in "${AIX_CONFIGS[@]}"; do
            want_config "${cfg}" || continue
            exports="${BASE_EXPORTS},${AIX_CONFIG_EXPORTS[${cfg}]},HH3_CONFIG_ENV=${cfg},HH3_LAYOUT_ENV=${layout}"
            if [[ -n "${RUN_SCENARIOS_ENV:-}" ]]; then
                exports="${exports},HH3_SCENARIOS_ENV=${RUN_SCENARIOS_ENV}"
            fi

            job_scale="HH3_H_scale_${cfg}_${layout}"
            echo "Submitting ${job_scale} (01:00:00, 17 nodes)..."
            jid_scale=$(sbatch --parsable --job-name="${job_scale}" --account=thes2181 --time=01:00:00 --export="${exports}" "${H_SCALE_WRAPPER}" | tail -n1 | cut -d';' -f1)
            echo "  -> Job ID: ${jid_scale}"
            if [ -x "${HOME}/scripts/swatch" ]; then
                "${HOME}/scripts/swatch" --jobid "${jid_scale}" 2>/dev/null || true
            fi
        done
    done

    # 2. Non-AIx Bundled Jobs (layout=default, time 01:30:00, 17 nodes)
    for cfg in "${NON_AIX_CONFIGS[@]}"; do
        want_config "${cfg}" || continue
        exports="${BASE_EXPORTS},${NON_AIX_CONFIG_EXPORTS[${cfg}]},HH3_CONFIG_ENV=${cfg},HH3_LAYOUT_ENV=default"
        if [[ -n "${RUN_SCENARIOS_ENV:-}" ]]; then
            exports="${exports},HH3_SCENARIOS_ENV=${RUN_SCENARIOS_ENV}"
        fi

        job_scale="HH3_H_scale_${cfg}"
        echo "Submitting ${job_scale} (01:30:00, 17 nodes)..."
        jid_scale=$(sbatch --parsable --job-name="${job_scale}" --account=thes2181 --time=01:30:00 --export="${exports}" "${H_SCALE_WRAPPER}" | tail -n1 | cut -d';' -f1)
        echo "  -> Job ID: ${jid_scale}"
        if [ -x "${HOME}/scripts/swatch" ]; then
            "${HOME}/scripts/swatch" --jobid "${jid_scale}" 2>/dev/null || true
        fi
    done
fi

echo ""
echo "=== Submission script finished ==="
