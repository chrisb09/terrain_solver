#!/usr/bin/env bash
#SBATCH --job-name=probe_5n_ctrl
#SBATCH --account=rwth2150
#SBATCH --partition=c23ms
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:08:00
#SBATCH --output=logs/probe_5n_%j.out
#SBATCH --error=logs/probe_5n_%j.err

set -uo pipefail

echo "========================================================================"
echo " 5-Node SmartSim Controller Probe Test (Partial Node Allocation)"
echo " Job ID:    ${SLURM_JOB_ID:-$$}"
echo " Host:      $(hostname)"
echo " Partition: ${SLURM_JOB_PARTITION:-c23ms}"
echo " Account:   ${SLURM_JOB_ACCOUNT:-rwth2150}"
echo " Date:      $(date)"
echo "========================================================================"

MINI_APP_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app"
BASE_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim"
cd "${MINI_APP_DIR}" || exit 1
mkdir -p logs

source "${BASE_DIR}/set_env_claix23_cuda12.4.sh"
SMARTSIM_PYTHON="${BASE_DIR}/python/smartsim_cpu/bin/python3"

hosts=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
if [[ ${#hosts[@]} -lt 5 ]]; then
    echo "ERROR: Expected 5 nodes, got ${#hosts[@]} (${hosts[*]})" >&2
    exit 1
fi
solver_host="${hosts[0]}"
db_hosts=("${hosts[1]}" "${hosts[2]}" "${hosts[3]}" "${hosts[4]}")
db_nodelist=$(IFS=,; echo "${db_hosts[*]}")

echo "Topology assignment:"
echo "  Solver node (1): ${solver_host}"
echo "  DB nodes (4):    ${db_nodelist}"

test_controller_layout() {
    local layout="$1"
    echo ""
    echo "------------------------------------------------------------------------"
    echo " PROBING LAYOUT: ${layout^^}"
    echo "------------------------------------------------------------------------"

    local probe_dir="${MINI_APP_DIR}/results/probe_5n_${SLURM_JOB_ID:-$$}_${layout}"
    mkdir -p "${probe_dir}"
    local host_file="${probe_dir}/db_host.txt"
    local done_file="${MINI_APP_DIR}/close_driver_probe_${SLURM_JOB_ID:-$$}_${layout}.txt"
    local ctrl_log="logs/probe_ctrl_${layout}_${SLURM_JOB_ID:-$$}.log"
    rm -f "${host_file}" "${done_file}" "${ctrl_log}"

    export RUN_ID_ENV="probe_${SLURM_JOB_ID:-$$}_${layout}"
    export SMARTSIM_INTERFACE="ib0"
    export SMARTSIM_INTRA_OP_THREADS=2
    export SMARTSIM_INTER_OP_THREADS=1
    export SMARTSIM_THREADS_PER_QUEUE=1
    export MLCOUPLING_SMARTSIM_DB_LAYOUT="${layout}"
    export MLCOUPLING_SMARTSIM_NODES=4
    export SMARTSIM_PIN_DB_NODELIST=1
    export SMARTSIM_DB_NODELIST="${db_nodelist}"
    export SMARTSIM_DB_EXCLUDE_NODE="${solver_host}"

    local base_port=$(( 6800 + (RANDOM % 800) ))
    echo "Launching controller on port ${base_port}..."
    ${SMARTSIM_PYTHON} -u smartsim_controller.py \
        --db_nodes 4 \
        --db_layout "${layout}" \
        --interface ib0 \
        --port "${base_port}" \
        --cpu_cores_per_node 2 \
        --hostname_file "${host_file}" > "${ctrl_log}" 2>&1 &
    local ctrl_pid=$!
    echo "Controller started with PID: ${ctrl_pid}"

    local wait_s=0
    while [[ ! -s "${host_file}" ]]; do
        if ! ps -p "${ctrl_pid}" >/dev/null 2>&1; then
            echo "ERROR: Controller exited prematurely!" >&2
            cat "${ctrl_log}" >&2
            return 1
        fi
        sleep 1
        wait_s=$((wait_s + 1))
        if [[ ${wait_s} -ge 60 ]]; then
            echo "ERROR: Timeout after ${wait_s}s. Controller output:" >&2
            cat "${ctrl_log}" >&2
            kill "${ctrl_pid}" 2>/dev/null || true
            return 1
        fi
    done

    local ssdb=$(cat "${host_file}")
    echo "SUCCESS: Database ready! SSDB=${ssdb}"

    # Verify connectivity from solver host
    echo "Testing ping from solver host (${solver_host}) to database endpoints..."
    ${SMARTSIM_PYTHON} -c "
import redis
ssdb = '${ssdb}'
for addr in ssdb.split(','):
    host, port = addr.split(':')
    try:
        r = redis.Redis(host=host, port=int(port), socket_timeout=5)
        res = r.ping()
        print(f'  Ping {addr}: ok ({res})')
    except Exception as e:
        print(f'  Ping {addr}: FAIL ({e})')
"

    echo "Shutting down controller..."
    touch "${done_file}"
    wait "${ctrl_pid}" 2>/dev/null || true
    echo "Controller shut down cleanly."
}

test_controller_layout "shared"
test_controller_layout "per-ml-node"

echo ""
echo "========================================================================"
echo " Probe completed successfully!"
echo "========================================================================"
