#!/bin/bash
#SBATCH --job-name=cpu_scorep_smoke
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=logs/cpu_scorep_smoke_%j.log

set -uo pipefail

echo "=========================================================="
echo "CPU Terrain Solver Score-P Smoke Test"
echo "Job ID:    ${SLURM_JOB_ID:-$$}"
echo "Node:      $(hostname)"
echo "Date:      $(date)"
echo "Binary:    build_scorep_cmi_fresh/terrain_solver"
echo "=========================================================="

MINI_APP_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app"
BASE_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim"
CPP_DIR="${BASE_DIR}/CPP-ML-Interface"
cd "${MINI_APP_DIR}" || exit 1

source "${BASE_DIR}/set_env_claix23_cuda12.4.sh"
export CUDA_STUBS_DIR="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/stubs/lib64"
export LD_LIBRARY_PATH="${CUDA_STUBS_DIR}:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HDF5_USE_FILE_LOCKING=FALSE
export SR_MODEL_TIMEOUT=900000
export SR_CMD_TIMEOUT=900000
export SR_SOCKET_TIMEOUT=900000

# Score-P runtime environment
export CMI_DIR="${CPP_DIR}"
export SMARTSIM_PAPI_ROOT="${CPP_DIR}/tmp/opencode/papi-7.2.0-install"
export SMARTSIM_SCOREP_ROOT="${CPP_DIR}/tmp/opencode/scorep-8.4-papi72-install"
source "${CPP_DIR}/env_scorep.sh"
export SCOREP_ENABLE_PROFILING=true
export SCOREP_ENABLE_TRACING=false
export SCOREP_METRIC_PAPI=
export SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY=true

SOLVER_BIN="${MINI_APP_DIR}/solver_cpp/${SOLVER_BUILD:-build_scorep_cmi_fresh}/terrain_solver"
MODEL_PATH_FLAT="${MINI_APP_DIR}/train_models/model_a/watercnn_cpu.pt"
MODEL_PATH_SPLIT="${MINI_APP_DIR}/train_models/model_a/watercnn_split_cpu.pt"
PREP_H5="${PREP_H5_OVERRIDE:-${MINI_APP_DIR}/input/prep_watercnn_4k.h5}"
SMARTSIM_PYTHON="${CPP_DIR}/extern/python/smartsim_cuda-12/bin/python"
PHYDLL_DL_CLIENT="${CPP_DIR}/dl_clients/build-scorep/phydll_dl_client"

# Rank-count / grid are env-overridable so a reduced-rank run can fit a partial node.
MPI_RANKS="${MPI_RANKS:-96}"
RANK_GRID_X="${RANK_GRID_X:-12}"
RANK_GRID_Z="${RANK_GRID_Z:-8}"

mkdir -p logs

for f in "${SOLVER_BIN}" "${MODEL_PATH_FLAT}" "${MODEL_PATH_SPLIT}" "${PREP_H5}"; do
    if [[ ! -f "${f}" ]]; then
        echo "Error: required file not found: ${f}" >&2
        exit 1
    fi
done

RUN_ROOT="/hpcwork/thes2181/mini_app/scorep_smoke_${SLURM_JOB_ID:-$$}"
mkdir -p "${RUN_ROOT}"

COMMON_SOLVER_ARGS=(
    --device CPU
    --gpus-per-node 0
    --ml-nodes 1
    --model-backend TORCH
    --input-hdf5 "${PREP_H5}"
    --steps 6
    --save-every 6
    --save-mode periodic
    --triangular-scale 1
    --chunk-size 12
    --io-mode parallel_hdf5
    --mpi-sync-mode none
    --hdf5-xfer-mode independent
    --rank-grid-x "${RANK_GRID_X}"
    --rank-grid-z "${RANK_GRID_Z}"
    --ml-interface cpp
)

declare -A RESULTS_WARM
declare -A RESULTS_STATUS
declare -A RESULTS_SCOREP
declare -A RESULTS_WALL

# --------------------------------------------------------------
run_smartsim_config() {
    local cfg_name="$1" omit_threads="$2" tpq="$3" intra="$4" bind_cpus="$5"
    local omit_set_cpus="$6" db_bs="$7" db_min_bs="$8"

    local cfg_dir="${RUN_ROOT}/${cfg_name}"
    local scorep_dir="${cfg_dir}/scorep"
    mkdir -p "${cfg_dir}"; rm -rf "${scorep_dir}"
    local log_file="logs/scorep_${cfg_name}_${SLURM_JOB_ID:-$$}.log"
    local db_host_file="${cfg_dir}/db_host.txt"
    local run_id="${cfg_name}_${SLURM_JOB_ID:-$$}"
    local done_file="${MINI_APP_DIR}/close_driver_${run_id}.txt"
    rm -f "${done_file}" "${db_host_file}"

    echo ""
    echo ">>> ${cfg_name}: TPQ=${tpq} INTRA=${intra} BIND=${bind_cpus} omit_threads=${omit_threads} omit_set_cpus=${omit_set_cpus} db_batch=${db_bs} min_batch=${db_min_bs}"

    export SMARTSIM_OMIT_THREADS="${omit_threads}"
    export SMARTSIM_OMIT_SET_CPUS="${omit_set_cpus}"
    export SMARTSIM_THREADS_PER_QUEUE="${tpq}"
    export SMARTSIM_INTRA_OP_THREADS="${intra}"
    export SMARTSIM_INTER_OP_THREADS=1
    export RUN_ID_ENV="${run_id}"

    local port=$(( 7100 + (RANDOM % 1000) ))
    ${SMARTSIM_PYTHON} smartsim_controller.py \
        --db_nodes 1 --db_layout shared --port "${port}" \
        --cpu_cores_per_node "${bind_cpus}" \
        --hostname_file "${db_host_file}" > "logs/controller_${run_id}.log" 2>&1 &
    local ctrl_pid=$!

    local wait_s=0
    while [[ ! -s "${db_host_file}" ]]; do
        if ! ps -p "${ctrl_pid}" >/dev/null 2>&1; then
            echo "    Controller exited prematurely"; RESULTS_STATUS["${cfg_name}"]="FAILED (controller)"; return 1
        fi
        sleep 1; wait_s=$((wait_s+1))
        [[ ${wait_s} -ge 120 ]] && { kill "${ctrl_pid}" 2>/dev/null; RESULTS_STATUS["${cfg_name}"]="FAILED (db timeout)"; return 1; }
    done
    export SSDB="$(cat "${db_host_file}")"
    echo "    DB ready: ${SSDB}"

    local batch_args=(--ml-batch-size 10000000 --ml-db-batch-size "${db_bs}" --ml-db-min-batch-size "${db_min_bs}")

    export SCOREP_EXPERIMENT_DIRECTORY="${scorep_dir}"
    local t0=$SECONDS
    timeout 420 mpirun --bind-to none --oversubscribe -x LD_LIBRARY_PATH -x SCOREP_EXPERIMENT_DIRECTORY \
        -x SCOREP_ENABLE_PROFILING -x SCOREP_ENABLE_TRACING -x SCOREP_METRIC_PAPI \
        -x SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY \
        -n "${MPI_RANKS}" "${SOLVER_BIN}" \
        "${COMMON_SOLVER_ARGS[@]}" \
        --model-path "${MODEL_PATH_SPLIT}" --model-io-layout split_3x3 \
        --cpp-ml-config config.toml \
        --output-hdf5 "${cfg_dir}/traj.h5" \
        "${batch_args[@]}" > "${log_file}" 2>&1
    local rc=$?
    local wall=$((SECONDS - t0))
    RESULTS_WALL["${cfg_name}"]="${wall}"

    touch "${done_file}"; wait "${ctrl_pid}" 2>/dev/null || true
    pkill -u "${USER}" -f "redis-server" 2>/dev/null || true

    if [[ ${rc} -eq 0 ]]; then
        local warm_ms=$(grep "STEP_TIMING step=4" "${log_file}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p')
        local warm_s=$(awk "BEGIN {printf \"%.2f\", ${warm_ms:-0} / 1000.0}")
        RESULTS_WARM["${cfg_name}"]="${warm_s}"
        RESULTS_STATUS["${cfg_name}"]="SUCCESS"
        RESULTS_SCOREP["${cfg_name}"]="$(ls -d "${scorep_dir}"/*.cubex 2>/dev/null | head -1 | xargs -r basename)"
        echo "    SUCCESS step4=${warm_s}s wall=${wall}s  scorep=$(ls -d ${scorep_dir}/*.cubex 2>/dev/null | wc -l) cubex"
        echo "    rank_min/max: $(grep 'STEP_TIMING step=4' "${log_file}" | sed -n 's/.*\(rank_min_ms=[0-9.]* rank_max_ms=[0-9.]*\).*/\1/p')"
    else
        RESULTS_STATUS["${cfg_name}"]="FAILED (rc=${rc})"
        echo "    FAILED rc=${rc} wall=${wall}s (see ${log_file})"
    fi
}

# --------------------------------------------------------------
# 1. AIX
echo ""
echo ">>> AIX"
AIX_DIR="${RUN_ROOT}/AIX"; mkdir -p "${AIX_DIR}"; rm -rf "${AIX_DIR}/scorep"
LOG_AIX="logs/scorep_AIX_${SLURM_JOB_ID:-$$}.log"
export SCOREP_EXPERIMENT_DIRECTORY="${AIX_DIR}/scorep"
t0=$SECONDS
timeout 420 mpirun --bind-to none --oversubscribe -x LD_LIBRARY_PATH -x SCOREP_EXPERIMENT_DIRECTORY \
    -x SCOREP_ENABLE_PROFILING -x SCOREP_ENABLE_TRACING -x SCOREP_METRIC_PAPI \
    -x SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY \
    -n "${MPI_RANKS}" "${SOLVER_BIN}" "${COMMON_SOLVER_ARGS[@]}" \
    --model-path "${MODEL_PATH_FLAT}" --model-io-layout flat_contiguous \
    --cpp-ml-config config_aix.toml --ml-batch-size 10000000 \
    --output-hdf5 "${AIX_DIR}/traj.h5" > "${LOG_AIX}" 2>&1
rc=$?
wall=$((SECONDS - t0)); RESULTS_WALL["AIX"]="${wall}"
if [[ ${rc} -eq 0 ]]; then
    wm=$(grep 'STEP_TIMING step=4' "${LOG_AIX}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p'); wm=${wm:-0}
    ws=$(awk "BEGIN {printf \"%.2f\", ${wm} / 1000.0}")
    RESULTS_WARM["AIX"]="${ws}"; RESULTS_STATUS["AIX"]="SUCCESS"
    RESULTS_SCOREP["AIX"]="$(ls -d ${AIX_DIR}/scorep/*.cubex 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')"
    echo "    SUCCESS step4=${ws}s wall=${wall}s"
else
    RESULTS_STATUS["AIX"]="FAILED (rc=${rc})"; echo "    FAILED rc=${rc} wall=${wall}s"
fi

# --------------------------------------------------------------
# 2. PHYDLL_CPP
echo ""
echo ">>> PHYDLL_CPP"
PHY_DIR="${RUN_ROOT}/PHYDLL_CPP"; mkdir -p "${PHY_DIR}"; rm -rf "${PHY_DIR}/scorep"
LOG_PHY="logs/scorep_PHYDLL_CPP_${SLURM_JOB_ID:-$$}.log"
export MLCOUPLING_INTRA_OP_THREADS=1 MLCOUPLING_INTER_OP_THREADS=1 PHYDLL_DL_COUNT=1 PHYDLL_DL_FIELD_COUNT=1
export SCOREP_EXPERIMENT_DIRECTORY="${PHY_DIR}/scorep"
t0=$SECONDS
timeout 420 mpirun --bind-to none --oversubscribe \
    -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
    -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
    -x SCOREP_EXPERIMENT_DIRECTORY -x SCOREP_ENABLE_PROFILING -x SCOREP_ENABLE_TRACING \
    -x SCOREP_METRIC_PAPI -x SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY \
    -n "${MPI_RANKS}" "${SOLVER_BIN}" "${COMMON_SOLVER_ARGS[@]}" \
    --model-path "${MODEL_PATH_FLAT}" --model-io-layout flat_contiguous \
    --cpp-ml-config config_phydll.toml --ml-batch-size 10000000 \
    --output-hdf5 "${PHY_DIR}/traj.h5" \
    : -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
    -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
    -n "${MPI_RANKS}" "${PHYDLL_DL_CLIENT}" > "${LOG_PHY}" 2>&1
rc=$?
wall=$((SECONDS - t0)); RESULTS_WALL["PHYDLL_CPP"]="${wall}"
if [[ ${rc} -eq 0 ]]; then
    wm=$(grep 'STEP_TIMING step=4' "${LOG_PHY}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p'); wm=${wm:-0}
    ws=$(awk "BEGIN {printf \"%.2f\", ${wm} / 1000.0}")
    RESULTS_WARM["PHYDLL_CPP"]="${ws}"; RESULTS_STATUS["PHYDLL_CPP"]="SUCCESS"
    RESULTS_SCOREP["PHYDLL_CPP"]="$(ls -d ${PHY_DIR}/scorep/*.cubex 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')"
    echo "    SUCCESS step4=${ws}s wall=${wall}s"
else
    RESULTS_STATUS["PHYDLL_CPP"]="FAILED (rc=${rc})"; echo "    FAILED rc=${rc} wall=${wall}s"
fi

# --------------------------------------------------------------
# 3-5. SmartSim configs
# SS_DEFAULT = literal SmartSim default: db_batch=0, min_batch=0, threads omitted
run_smartsim_config "SS_DEFAULT" 1 1 1 96 1 0 0
# allow-batching "sane" default: db_batch=10M, min_batch=0
run_smartsim_config "SS_TPQ96_I1_B96" 0 96 1 96 0 10000000 0
run_smartsim_config "SS_TPQ1_I96_B96" 0 1 96 96 0 10000000 0

# --------------------------------------------------------------
echo ""
echo "=========================================================="
echo "SCORE-P SMOKE SUMMARY"
echo "=========================================================="
printf "%-22s | %-8s | %-8s | %-10s | %s\n" "Config" "step4(s)" "wall(s)" "Status" "Score-P experiment"
for cfg in AIX PHYDLL_CPP SS_DEFAULT SS_TPQ96_I1_B96 SS_TPQ1_I96_B96; do
    printf "%-22s | %-8s | %-8s | %-10s | %s\n" "${cfg}" "${RESULTS_WARM[$cfg]:-N/A}" "${RESULTS_WALL[$cfg]:-N/A}" "${RESULTS_STATUS[$cfg]:-NOT RUN}" "${RESULTS_SCOREP[$cfg]:-}"
done
echo "=========================================================="
echo "Score-P dirs under ${RUN_ROOT}/<cfg>/scorep"
echo "Completed at $(date)"
