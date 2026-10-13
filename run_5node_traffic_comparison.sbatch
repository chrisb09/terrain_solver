#!/usr/bin/env bash
#SBATCH --job-name=ss_5node_traffic
#SBATCH --account=rwth2150
#SBATCH --partition=c23ms
#SBATCH --nodes=5
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --switches=1@00:05:00
#SBATCH --output=logs/5node_traffic_%j.out
#SBATCH --error=logs/5node_traffic_%j.err

set -uo pipefail

echo "========================================================================"
echo " 5-Node SmartSim Clustered vs Standalone Traffic Benchmark"
echo " Job ID:    ${SLURM_JOB_ID:-$$}"
echo " Date:      $(date)"
echo " Partition: ${SLURM_JOB_PARTITION:-c23ms}"
echo " Account:   ${SLURM_JOB_ACCOUNT:-rwth2150}"
echo " Order:     ${ORDER_ENV:-clustered_first}"
echo "========================================================================"

MINI_APP_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app"
BASE_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim"
CPP_DIR="${BASE_DIR}/CPP-ML-Interface"

cd "${MINI_APP_DIR}" || exit 1
mkdir -p logs

# Environment setup
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
export SCOREP_METRIC_PAPI=""
export SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY=true

SOLVER_BIN="${MINI_APP_DIR}/solver_cpp/build_scorep_cmi/terrain_solver"
MODEL_PATH="${MINI_APP_DIR}/train_models/model_a/watercnn_split_cpu.pt"
PREP_H5="${MINI_APP_DIR}/input/prep_watercnn_4k.h5"
CONFIG_TOML="${MINI_APP_DIR}/config_5node_traffic.toml"
SMARTSIM_PYTHON="${BASE_DIR}/python/smartsim_cpu/bin/python3"
ANALYSIS_SCRIPT="${MINI_APP_DIR}/analyze_traffic_comparison.py"

MPI_RANKS=96
RANK_GRID_X=12
RANK_GRID_Z=8
TOTAL_STEPS=102
SAVE_EVERY=102
CHUNK_SIZE=12
TARGET_WIDTH=3840
TARGET_HEIGHT=2160
DB_SHARDS=4
DB_CORES_PER_SHARD=96

# Verify required files
for f in "${SOLVER_BIN}" "${MODEL_PATH}" "${PREP_H5}" "${CONFIG_TOML}" "${SMARTSIM_PYTHON}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: Required file not found: ${f}" >&2
        exit 1
    fi
done

# Node discovery and partitioning
hosts=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
if [[ ${#hosts[@]} -lt 5 ]]; then
    echo "ERROR: Expected 5 nodes, got ${#hosts[@]} (${hosts[*]})" >&2
    exit 1
fi
solver_host="${hosts[0]}"
db_hosts=("${hosts[1]}" "${hosts[2]}" "${hosts[3]}" "${hosts[4]}")
db_nodelist=$(IFS=,; echo "${db_hosts[*]}")

echo "Topology assignment:"
echo "  Solver node (1):  ${solver_host} (co-located with batch script)"
echo "  DB nodes (4):     ${db_nodelist} (dedicated ML nodes)"

RUN_ROOT="${MINI_APP_DIR}/results/5node_traffic_${SLURM_JOB_ID:-$$}"
mkdir -p "${RUN_ROOT}"

sample_ib_dev() {
    local tag="$1"
    local outfile="$2"
    srun --overlap --exact --nodelist="${solver_host}" --nodes=1 --ntasks=1 bash -c "echo -n '${tag} ' && grep 'ib0:' /proc/net/dev" >> "${outfile}" 2>/dev/null
}

run_experiment() {
    local layout="$1" # "shared" or "per-ml-node"
    local label="$2"  # "clustered" or "standalone"

    local exp_dir="${RUN_ROOT}/${label}"
    local scorep_dir="${exp_dir}/scorep"
    mkdir -p "${exp_dir}"
    rm -rf "${scorep_dir}"

    local run_id="${label}_${SLURM_JOB_ID:-$$}"
    local db_host_file="${exp_dir}/db_host.txt"
    local done_file="${MINI_APP_DIR}/close_driver_${run_id}.txt"
    local solver_log="logs/solver_${label}_${SLURM_JOB_ID:-$$}.log"
    local driver_log="logs/driver_${label}_${SLURM_JOB_ID:-$$}.log"
    local net_snapshot="${exp_dir}/net_ib0_snapshot.txt"

    rm -f "${done_file}" "${db_host_file}" "${net_snapshot}"
    # Cleanup any remote stale redis-servers on DB nodes
    srun --nodelist="${db_nodelist}" --nodes=4 --ntasks-per-node=1 bash -c "pkill -u \${USER} -f redis-server || true" 2>/dev/null || true
    sleep 2

    echo ""
    echo "========================================================================"
    echo " RUNNING CASE: ${label^^} (db_layout=${layout}, db_nodes=${DB_SHARDS})"
    echo "========================================================================"

    export RUN_ID_ENV="${run_id}"
    export SMARTSIM_INTERFACE="ib0"
    export SMARTSIM_INTRA_OP_THREADS=8
    export SMARTSIM_INTER_OP_THREADS=1
    export SMARTSIM_THREADS_PER_QUEUE=12
    export MLCOUPLING_SMARTSIM_DB_LAYOUT="${layout}"
    export MLCOUPLING_SMARTSIM_NODES="${DB_SHARDS}"
    export SMARTSIM_PIN_DB_NODELIST=1
    export SMARTSIM_DB_NODELIST="${db_nodelist}"
    export SMARTSIM_DB_EXCLUDE_NODE="${solver_host}"

    local base_port=$(( 6800 + (RANDOM % 800) ))

    echo "Starting SmartSim controller on port ${base_port} (InfiniBand ib0)..."
    ${SMARTSIM_PYTHON} -u smartsim_controller.py \
        --db_nodes "${DB_SHARDS}" \
        --db_layout "${layout}" \
        --interface ib0 \
        --port "${base_port}" \
        --cpu_cores_per_node "${DB_CORES_PER_SHARD}" \
        --hostname_file "${db_host_file}" > "${driver_log}" 2>&1 &
    local ctrl_pid=$!

    local wait_s=0
    while [[ ! -s "${db_host_file}" ]]; do
        if ! ps -p "${ctrl_pid}" >/dev/null 2>&1; then
            echo "ERROR: SmartSim controller exited prematurely. Check ${driver_log}" >&2
            cat "${driver_log}" >&2
            return 1
        fi
        sleep 1
        wait_s=$((wait_s + 1))
        if [[ ${wait_s} -ge 180 ]]; then
            echo "ERROR: Database readiness timeout after ${wait_s}s. Dumping ${driver_log}:" >&2
            cat "${driver_log}" >&2
            kill "${ctrl_pid}" 2>/dev/null || true
            return 1
        fi
    done

    export SSDB="$(cat "${db_host_file}")"
    echo "Database ready. SSDB=${SSDB}"

    export SCOREP_EXPERIMENT_DIRECTORY="${scorep_dir}"

    echo "Recording initial /proc/net/dev ib0 snapshot on solver node ${solver_host}..."
    sample_ib_dev "START" "${net_snapshot}"

    local t_start=$SECONDS
    echo "Launching terrain solver on dedicated node ${solver_host} (96 ranks, 4K WaterCNN, ${TOTAL_STEPS} steps)..."
    srun --overlap --exact --nodelist="${solver_host}" --nodes=1 --ntasks="${MPI_RANKS}" --cpus-per-task=1 \
        --export=ALL \
        "${SOLVER_BIN}" \
        --device CPU \
        --gpus-per-node 0 \
        --ml-nodes "${DB_SHARDS}" \
        --model-backend TORCH \
        --model-path "${MODEL_PATH}" \
        --model-io-layout split_3x3 \
        --input-hdf5 "${PREP_H5}" \
        --output-hdf5 "${exp_dir}/trajectory.h5" \
        --steps "${TOTAL_STEPS}" \
        --save-every "${SAVE_EVERY}" \
        --save-mode periodic \
        --triangular-scale 1 \
        --chunk-size "${CHUNK_SIZE}" \
        --io-mode parallel_hdf5 \
        --mpi-sync-mode none \
        --hdf5-xfer-mode independent \
        --rank-grid-x "${RANK_GRID_X}" \
        --rank-grid-z "${RANK_GRID_Z}" \
        --ml-interface cpp \
        --cpp-ml-config "${CONFIG_TOML}" > "${solver_log}" 2>&1
    local solver_rc=$?
    local t_elapsed=$((SECONDS - t_start))

    echo "Recording final /proc/net/dev ib0 snapshot on solver node ${solver_host}..."
    sample_ib_dev "END" "${net_snapshot}"

    echo "Signaling database shutdown..."
    touch "${done_file}"
    wait "${ctrl_pid}" 2>/dev/null || true
    srun --overlap --exact --nodelist="${db_nodelist}" --nodes=4 --ntasks-per-node=1 bash -c "pkill -u \${USER} -f redis-server || true" 2>/dev/null || true
    sleep 2

    if [[ ${solver_rc} -eq 0 ]]; then
        echo "SUCCESS: ${label} solver completed in ${t_elapsed}s (rc=${solver_rc})"
        grep "NET_USAGE" "${solver_log}" || true
    else
        echo "FAILED: ${label} solver failed with rc=${solver_rc} in ${t_elapsed}s. Check ${solver_log}" >&2
        tail -n 30 "${solver_log}" >&2
        return 1
    fi
}

ORDER="${ORDER_ENV:-clustered_first}"
if [[ "${ORDER}" == "standalone_first" ]]; then
    run_experiment "per-ml-node" "standalone"
    run_experiment "shared" "clustered"
else
    run_experiment "shared" "clustered"
    run_experiment "per-ml-node" "standalone"
fi

echo ""
echo "========================================================================"
echo " RUNNING COMPARISON & VALIDATION ANALYSIS (InfiniBand ib0)"
echo "========================================================================"

CLUST_DIR="${RUN_ROOT}/clustered"
STAND_DIR="${RUN_ROOT}/standalone"

python3 "${ANALYSIS_SCRIPT}" \
    --clustered-log "logs/solver_clustered_${SLURM_JOB_ID:-$$}.log" \
    --standalone-log "logs/solver_standalone_${SLURM_JOB_ID:-$$}.log" \
    --clustered-net "${CLUST_DIR}/net_ib0_snapshot.txt" \
    --standalone-net "${STAND_DIR}/net_ib0_snapshot.txt" \
    --clustered-cubex "${CLUST_DIR}/scorep/profile.cubex" \
    --standalone-cubex "${STAND_DIR}/scorep/profile.cubex" \
    --ranks "${MPI_RANKS}" \
    --shards "${DB_SHARDS}" \
    --width "${TARGET_WIDTH}" \
    --height "${TARGET_HEIGHT}" \
    --steps "${TOTAL_STEPS}" \
    --interface "ib0"

echo "Benchmark run complete. Results stored in: ${RUN_ROOT}"
