#!/bin/bash
#SBATCH --job-name=cpu_smoke_7cfg
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=logs/cpu_smoke_7cfg_%j.log

set -uo pipefail

echo "=========================================================="
echo "Starting CPU Terrain Solver Smoke Test (7 Configurations)"
echo "Job ID:        ${SLURM_JOB_ID:-$$}"
echo "Node:          $(hostname)"
echo "Date:          $(date)"
echo "Account:       rwth2150"
echo "Partition:     c23mm (512 GB RAM)"
echo "=========================================================="

MINI_APP_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app"
BASE_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim"
CPP_DIR="${BASE_DIR}/CPP-ML-Interface"
cd "${MINI_APP_DIR}" || exit 1

# Environment setup
source "${BASE_DIR}/set_env_claix23_cuda12.4.sh"
export CUDA_STUBS_DIR="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/stubs/lib64"
export LD_LIBRARY_PATH="${CUDA_STUBS_DIR}:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HDF5_USE_FILE_LOCKING=FALSE
export SR_MODEL_TIMEOUT=900000
export SR_CMD_TIMEOUT=900000
export SR_SOCKET_TIMEOUT=900000

# Binaries and files
SOLVER_BIN="${MINI_APP_DIR}/solver_cpp/build_matrix_default/terrain_solver"
MODEL_PATH_FLAT="${MINI_APP_DIR}/train_models/model_a/watercnn_cpu.pt"
MODEL_PATH_SPLIT="${MINI_APP_DIR}/train_models/model_a/watercnn_split_cpu.pt"
PREP_H5="${MINI_APP_DIR}/input/prep_watercnn_4k.h5"
SMARTSIM_PYTHON="${CPP_DIR}/extern/python/smartsim_cuda-12/bin/python"
PHYDLL_DL_CLIENT="${BASE_DIR}/cpu_benchmark/provider_bench/build-provider-bench/dl-client/phydll_dl_client"
PHYDLL_DL_PY="${CPP_DIR}/dl_clients/phydll_dl_client.py"

mkdir -p logs

if [[ ! -x "${SOLVER_BIN}" ]]; then
    echo "Error: solver binary not found at ${SOLVER_BIN}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_PATH_FLAT}" ]]; then
    echo "Error: flat model file not found at ${MODEL_PATH_FLAT}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_PATH_SPLIT}" ]]; then
    echo "Error: split model file not found at ${MODEL_PATH_SPLIT}" >&2
    exit 1
fi
if [[ ! -f "${PREP_H5}" ]]; then
    echo "Error: input H5 not found at ${PREP_H5}" >&2
    exit 1
fi

RUN_ROOT="/hpcwork/thes2181/mini_app/smoke_cpu_${SLURM_JOB_ID:-$$}"
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
    --rank-grid-x 12
    --rank-grid-z 8
    --ml-interface cpp
)

declare -A RESULTS_WARM
declare -A RESULTS_STATUS

run_smartsim_config() {
    local cfg_name="$1"
    local omit_threads="$2"
    local tpq="$3"
    local intra="$4"
    local bind_cpus="$5"
    local omit_set_cpus="$6"
    local db_bs="$7"
    local db_min_bs="$8"
    local db_timeout="$9"

    local cfg_dir="${RUN_ROOT}/${cfg_name}"
    mkdir -p "${cfg_dir}"
    local log_file="logs/smoke_${cfg_name}_${SLURM_JOB_ID:-$$}.log"
    local db_host_file="${cfg_dir}/db_host.txt"
    local run_id="${cfg_name}_${SLURM_JOB_ID:-$$}"
    local done_file="${MINI_APP_DIR}/close_driver_${run_id}.txt"
    rm -f "${done_file}" "${db_host_file}"

    echo ""
    echo ">>> Running Config: ${cfg_name}"
    echo "    SmartSim: TPQ=${tpq}, INTRA=${intra}, BIND=${bind_cpus}, OmitThreads=${omit_threads}, OmitSetCpus=${omit_set_cpus}"
    echo "    Batching: DB_BATCH_SIZE=${db_bs}, MIN_BATCH_SIZE=${db_min_bs}, TIMEOUT=${db_timeout}"

    export SMARTSIM_OMIT_THREADS="${omit_threads}"
    export SMARTSIM_OMIT_SET_CPUS="${omit_set_cpus}"
    export SMARTSIM_THREADS_PER_QUEUE="${tpq}"
    export SMARTSIM_INTRA_OP_THREADS="${intra}"
    export SMARTSIM_INTER_OP_THREADS=1
    export RUN_ID_ENV="${run_id}"

    local port=$(( 7100 + (RANDOM % 1000) ))
    ${SMARTSIM_PYTHON} smartsim_controller.py \
        --db_nodes 1 \
        --db_layout shared \
        --port "${port}" \
        --cpu_cores_per_node "${bind_cpus}" \
        --hostname_file "${db_host_file}" > "logs/controller_${run_id}.log" 2>&1 &
    local ctrl_pid=$!

    echo "    Waiting for SmartSim database on port ${port}..."
    local wait_s=0
    while [[ ! -s "${db_host_file}" ]]; do
        if ! ps -p "${ctrl_pid}" >/dev/null 2>&1; then
            echo "    Error: Controller exited prematurely!"
            RESULTS_STATUS["${cfg_name}"]="FAILED (Controller crash)"
            return 1
        fi
        sleep 1
        wait_s=$((wait_s + 1))
        if [[ ${wait_s} -ge 120 ]]; then
            echo "    Error: Timeout waiting for database startup!"
            kill "${ctrl_pid}" 2>/dev/null || true
            RESULTS_STATUS["${cfg_name}"]="FAILED (DB Timeout)"
            return 1
        fi
    done
    local db_host=$(cat "${db_host_file}")
    echo "    DB is ready at: ${db_host}"

    export SSDB="${db_host}"
    local batch_args=(
        --ml-batch-size 10000000
        --ml-db-batch-size "${db_bs}"
        --ml-db-min-batch-size "${db_min_bs}"
    )
    if [[ "${db_timeout}" -gt 0 ]]; then
        batch_args+=(--ml-db-min-batch-timeout "${db_timeout}")
    fi

    local t_start=$(date +%s.%N)
    timeout 420 mpirun --bind-to none --oversubscribe -x LD_LIBRARY_PATH -n 96 "${SOLVER_BIN}" \
        "${COMMON_SOLVER_ARGS[@]}" \
        --model-path "${MODEL_PATH_SPLIT}" \
        --model-io-layout split_3x3 \
        --cpp-ml-config config.toml \
        --output-hdf5 "${cfg_dir}/traj.h5" \
        "${batch_args[@]}" > "${log_file}" 2>&1
    local rc=$?
    local t_end=$(date +%s.%N)
    local dur=$(awk "BEGIN {print ${t_end} - ${t_start}}")

    touch "${done_file}"
    wait "${ctrl_pid}" 2>/dev/null || true
    pkill -u "${USER}" -f "redis-server" 2>/dev/null || true

    if [[ ${rc} -eq 0 ]]; then
        local warm_ms=$(grep "STEP_TIMING step=4" "${log_file}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p')
        local warm_s=$(awk "BEGIN {printf \"%.2f\", ${warm_ms:-0} / 1000.0}")
        RESULTS_WARM["${cfg_name}"]="${warm_s} s"
        RESULTS_STATUS["${cfg_name}"]="SUCCESS (step4=${warm_s}s, total=${dur}s)"
        echo "    SUCCESS! Step 4 Warm Time: ${warm_s} s (Total Run: ${dur} s)"
    else
        RESULTS_STATUS["${cfg_name}"]="FAILED (rc=${rc})"
        echo "    FAILED with exit code ${rc}. Check ${log_file}"
    fi
}

# ==============================================================
# 1. AIX
# ==============================================================
echo ""
echo ">>> Running Config 1/7: AIX"
AIX_DIR="${RUN_ROOT}/AIX"
mkdir -p "${AIX_DIR}"
LOG_AIX="logs/smoke_AIX_${SLURM_JOB_ID:-$$}.log"
t_start=$(date +%s.%N)
timeout 420 mpirun --bind-to none --oversubscribe -x LD_LIBRARY_PATH -n 96 "${SOLVER_BIN}" \
    "${COMMON_SOLVER_ARGS[@]}" \
    --model-path "${MODEL_PATH_FLAT}" \
    --model-io-layout flat_contiguous \
    --cpp-ml-config config_aix.toml \
    --ml-batch-size 10000000 \
    --output-hdf5 "${AIX_DIR}/traj.h5" > "${LOG_AIX}" 2>&1
rc=$?
t_end=$(date +%s.%N)
dur=$(awk "BEGIN {print ${t_end} - ${t_start}}")
if [[ ${rc} -eq 0 ]]; then
    warm_ms=$(grep "STEP_TIMING step=4" "${LOG_AIX}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p')
    warm_s=$(awk "BEGIN {printf \"%.2f\", ${warm_ms:-0} / 1000.0}")
    RESULTS_WARM["AIX"]="${warm_s} s"
    RESULTS_STATUS["AIX"]="SUCCESS (step4=${warm_s}s, total=${dur}s)"
    echo "    SUCCESS! Step 4 Warm Time: ${warm_s} s (Total: ${dur} s)"
else
    RESULTS_STATUS["AIX"]="FAILED (rc=${rc})"
    echo "    FAILED with exit code ${rc}. Check ${LOG_AIX}"
fi

# ==============================================================
# 2. PHYDLL_CPP
# ==============================================================
echo ""
echo ">>> Running Config 2/7: PHYDLL_CPP"
PHY_CPP_DIR="${RUN_ROOT}/PHYDLL_CPP"
mkdir -p "${PHY_CPP_DIR}"
LOG_PHY_CPP="logs/smoke_PHYDLL_CPP_${SLURM_JOB_ID:-$$}.log"
export MLCOUPLING_INTRA_OP_THREADS=1
export MLCOUPLING_INTER_OP_THREADS=1
export PHYDLL_DL_COUNT=1
export PHYDLL_DL_FIELD_COUNT=1
export PHYDLL_MPMD_SHUTDOWN_BARRIER=1
t_start=$(date +%s.%N)
timeout 420 mpirun --bind-to none --oversubscribe \
    -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
    -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
    -x PHYDLL_MPMD_SHUTDOWN_BARRIER \
    -n 96 "${SOLVER_BIN}" \
    "${COMMON_SOLVER_ARGS[@]}" \
    --model-path "${MODEL_PATH_FLAT}" \
    --model-io-layout flat_contiguous \
    --cpp-ml-config config_phydll.toml \
    --ml-batch-size 10000000 \
    --output-hdf5 "${PHY_CPP_DIR}/traj.h5" \
    : -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
    -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
    -x PHYDLL_MPMD_SHUTDOWN_BARRIER \
    -n 96 "${PHYDLL_DL_CLIENT}" > "${LOG_PHY_CPP}" 2>&1
rc=$?
t_end=$(date +%s.%N)
dur=$(awk "BEGIN {print ${t_end} - ${t_start}}")
if [[ ${rc} -eq 0 ]]; then
    warm_ms=$(grep "STEP_TIMING step=4" "${LOG_PHY_CPP}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p')
    warm_s=$(awk "BEGIN {printf \"%.2f\", ${warm_ms:-0} / 1000.0}")
    RESULTS_WARM["PHYDLL_CPP"]="${warm_s} s"
    RESULTS_STATUS["PHYDLL_CPP"]="SUCCESS (step4=${warm_s}s, total=${dur}s)"
    echo "    SUCCESS! Step 4 Warm Time: ${warm_s} s (Total: ${dur} s)"
else
    RESULTS_STATUS["PHYDLL_CPP"]="FAILED (rc=${rc})"
    echo "    FAILED with exit code ${rc}. Check ${LOG_PHY_CPP}"
fi

# ==============================================================
# 3. PHYDLL_PY (optional: known teardown-barrier deadlock on this stack)
# ==============================================================
if [[ "${RUN_PHYDLL_PY:-0}" == "1" ]]; then
# ==============================================================
# 3. PHYDLL_PY
# ==============================================================
echo ""
echo ">>> Running Config 3/7: PHYDLL_PY"
PHY_PY_DIR="${RUN_ROOT}/PHYDLL_PY"
mkdir -p "${PHY_PY_DIR}"
LOG_PHY_PY="logs/smoke_PHYDLL_PY_${SLURM_JOB_ID:-$$}.log"
export PHYDLL_PY_FORCE_EXIT=1
t_start=$(date +%s.%N)
timeout 420 mpirun --bind-to none --oversubscribe \
    -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
    -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
    -x PHYDLL_MPMD_SHUTDOWN_BARRIER -x PHYDLL_PY_FORCE_EXIT \
    -n 96 "${SOLVER_BIN}" \
    "${COMMON_SOLVER_ARGS[@]}" \
    --model-path "${MODEL_PATH_FLAT}" \
    --model-io-layout flat_contiguous \
    --cpp-ml-config config_phydll.toml \
    --ml-batch-size 10000000 \
    --output-hdf5 "${PHY_PY_DIR}/traj.h5" \
    : -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
    -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
    -x PHYDLL_MPMD_SHUTDOWN_BARRIER -x PHYDLL_PY_FORCE_EXIT \
    -n 96 "${SMARTSIM_PYTHON}" "${PHYDLL_DL_PY}" > "${LOG_PHY_PY}" 2>&1
rc=$?
t_end=$(date +%s.%N)
dur=$(awk "BEGIN {print ${t_end} - ${t_start}}")
if [[ ${rc} -eq 0 ]]; then
    warm_ms=$(grep "STEP_TIMING step=4" "${LOG_PHY_PY}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p')
    warm_s=$(awk "BEGIN {printf \"%.2f\", ${warm_ms:-0} / 1000.0}")
    RESULTS_WARM["PHYDLL_PY"]="${warm_s} s"
    RESULTS_STATUS["PHYDLL_PY"]="SUCCESS (step4=${warm_s}s, total=${dur}s)"
    echo "    SUCCESS! Step 4 Warm Time: ${warm_s} s (Total: ${dur} s)"
else
    RESULTS_STATUS["PHYDLL_PY"]="FAILED (rc=${rc})"
    echo "    FAILED with exit code ${rc}. Check ${LOG_PHY_PY}"
fi

else
    echo ""
    echo ">>> Running Config 3/7: PHYDLL_PY"
    echo "    SKIPPED (RUN_PHYDLL_PY=0)"
    RESULTS_STATUS["PHYDLL_PY"]="SKIPPED (RUN_PHYDLL_PY=0)"
fi

# ==============================================================
# 4. SS_DEFAULT (Implicit defaults, unpinned, batch_size=10M, min_batch=0)
# ==============================================================
run_smartsim_config "SS_DEFAULT" 1 1 1 96 1 10000000 0 0

# ==============================================================
# 5. SS_TPQ96_I1_B96 (Optimal queue concurrency, batch_size=10M, min_batch=0)
# ==============================================================
run_smartsim_config "SS_TPQ96_I1_B96" 0 96 1 96 0 10000000 0 0

# ==============================================================
# 6. SS_TPQ1_I96_B96_BATCH0 (OpenMP thrashing cliff: batch_size=0 disabled)
# ==============================================================
run_smartsim_config "SS_TPQ1_I96_B96_BATCH0" 0 1 96 96 0 0 0 0

# ==============================================================
# 7. SS_TPQ1_I96_B96_BATCH100 (Batching rescue: min_batch=8294400 forced)
# ==============================================================
run_smartsim_config "SS_TPQ1_I96_B96_BATCH100" 0 1 96 96 0 10000000 8294400 5000

# ==============================================================
# Summary Report
# ==============================================================
echo ""
echo "=========================================================="
echo "SMOKE TEST SUMMARY REPORT"
echo "=========================================================="
printf "%-28s | %-12s | %-35s\n" "Configuration" "Step 4 Warm" "Status"
echo "-----------------------------+--------------+------------------------------------"
for cfg in "AIX" "PHYDLL_CPP" "PHYDLL_PY" "SS_DEFAULT" "SS_TPQ96_I1_B96" "SS_TPQ1_I96_B96_BATCH0" "SS_TPQ1_I96_B96_BATCH100"; do
    warm="${RESULTS_WARM[$cfg]:-N/A}"
    status="${RESULTS_STATUS[$cfg]:-NOT RUN}"
    printf "%-28s | %-12s | %-35s\n" "${cfg}" "${warm}" "${status}"
done
echo "=========================================================="
echo "Completed at $(date)"
