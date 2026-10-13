#!/bin/bash
#SBATCH --job-name=smoke_pinning
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --time=00:35:00
#SBATCH --exclusive
#SBATCH --output=logs/smoke_pinning_%j.log

set -euo pipefail
MINI_APP_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app"
BASE_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim"
CPP_DIR="${BASE_DIR}/CPP-ML-Interface"
cd "${MINI_APP_DIR}"

MODEL="${MODEL:-watercnn}"
STEPS="${STEPS:-6}"
RUN_TIMEOUT="${RUN_TIMEOUT:-600}"
DB_START_TIMEOUT=180

source "${BASE_DIR}/set_env_claix23_cuda12.4.sh"
export CUDA_STUBS_DIR="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/stubs/lib64"
export LD_LIBRARY_PATH="${CUDA_STUBS_DIR}:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HDF5_USE_FILE_LOCKING=FALSE
export SR_MODEL_TIMEOUT=900000 SR_CMD_TIMEOUT=900000 SR_SOCKET_TIMEOUT=900000

# Score-P runtime environment
export CMI_DIR="${CPP_DIR}"
export SMARTSIM_PAPI_ROOT="${CPP_DIR}/tmp/opencode/papi-7.2.0-install"
export SMARTSIM_SCOREP_ROOT="${CPP_DIR}/tmp/opencode/scorep-8.4-papi72-install"
source "${CPP_DIR}/env_scorep.sh"
export SCOREP_ENABLE_PROFILING=true
export SCOREP_ENABLE_TRACING=false
export SCOREP_METRIC_PAPI=
export SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY=true

# Critical diagnostic toggles:
# 1. PhyDLL logging barriers disabled!
export PHYDLL_IO_LOG=0 PHYDLL_IO_LOG_BARRIERS=0 PHYDLL_TELEMETRY=0
# 2. SmartSim telemetry disabled!
export SMARTSIM_FLAG_TELEMETRY=0
# 3. Solver internal memory logging disabled!
export MLCOUPLING_MEM_LOG_VERBOSE=0 MLCOUPLING_MEM_LOG_COLLECTIVE=0 MLCOUPLING_STEP_TIMING_SYNC=1

SOLVER_BIN="${MINI_APP_DIR}/solver_cpp/build_scorep_cmi_fresh/terrain_solver"
PHYDLL_DL_CLIENT="${CPP_DIR}/dl_clients/build-scorep/phydll_dl_client"
SMARTSIM_PYTHON="${CPP_DIR}/extern/python/smartsim_cuda-12/bin/python"
PIN_WRAPPER="${MINI_APP_DIR}/pin_wrapper.sh"

case "${MODEL}" in
    watercnn)
        MODEL_PATH_FLAT="${MINI_APP_DIR}/train_models/model_a/watercnn_cpu.pt"
        MODEL_PATH_SPLIT="${MINI_APP_DIR}/train_models/model_a/watercnn_split_cpu.pt"
        PREP_H5="${MINI_APP_DIR}/input/prep_watercnn_4k.h5"
        ;;
    giant)
        MODEL_PATH_FLAT="${MINI_APP_DIR}/train_models/model_a/giant_cpu.pt"
        MODEL_PATH_SPLIT="${MINI_APP_DIR}/train_models/model_a/giant_split_cpu.pt"
        PREP_H5="${MINI_APP_DIR}/input/prep_giant_240x192.h5"
        ;;
    *) echo "Unknown MODEL=${MODEL}" >&2; exit 1 ;;
esac

RUN_ROOT="${MINI_APP_DIR}/results/smoke_pinning/job_${SLURM_JOB_ID}_${MODEL}"
mkdir -p "${RUN_ROOT}" logs
RESULTS_CSV="${RUN_ROOT}/results.csv"
echo "config,binding,step4_ms,step6_ms,mean_steady_ms,wall_s,cubex" > "${RESULTS_CSV}"

echo "=========================================================="
echo "Pinning Smoke Test | MODEL=${MODEL} PARTITION=${SLURM_JOB_PARTITION:-unknown}"
echo "Job ${SLURM_JOB_ID:-$$} on $(hostname) at $(date)"
echo "RUN_ROOT=${RUN_ROOT}"
echo "=========================================================="

COMMON_SOLVER_ARGS=(
    --device CPU --gpus-per-node 0 --ml-nodes 1
    --model-backend TORCH
    --input-hdf5 "${PREP_H5}"
    --steps "${STEPS}" --save-every "${STEPS}" --save-mode periodic
    --triangular-scale 1 --chunk-size 12 --io-mode parallel_hdf5
    --mpi-sync-mode none --hdf5-xfer-mode independent
    --rank-grid-x 12 --rank-grid-z 8
    --ml-interface cpp
    --ml-batch-size 10000000
)

SCOREP_EXPORTS=(
    -x LD_LIBRARY_PATH -x SCOREP_EXPERIMENT_DIRECTORY
    -x SCOREP_ENABLE_PROFILING -x SCOREP_ENABLE_TRACING -x SCOREP_METRIC_PAPI
    -x SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY
    -x MLCOUPLING_MEM_LOG_VERBOSE -x MLCOUPLING_MEM_LOG_COLLECTIVE -x MLCOUPLING_STEP_TIMING_SYNC
)

run_aix() {
    local binding="$1" # unbound | pinned
    local tag="AIX_${binding}"
    local cfg_dir="${RUN_ROOT}/${tag}"
    local scorep_dir="${cfg_dir}/scorep"
    local log_file="${cfg_dir}/run.log"
    mkdir -p "${cfg_dir}"; rm -rf "${scorep_dir}"
    export SCOREP_EXPERIMENT_DIRECTORY="${scorep_dir}"

    local cmd=(timeout --kill-after=30 "${RUN_TIMEOUT}" mpirun --bind-to none --oversubscribe "${SCOREP_EXPORTS[@]}")
    if [[ "${binding}" == "pinned" ]]; then
        cmd+=(-n 96 "${PIN_WRAPPER}" "${SOLVER_BIN}")
    else
        cmd+=(-n 96 "${SOLVER_BIN}")
    fi
    cmd+=(
        "${COMMON_SOLVER_ARGS[@]}"
        --model-path "${MODEL_PATH_FLAT}" --model-io-layout flat_contiguous
        --cpp-ml-config config_aix.toml
        --output-hdf5 "${cfg_dir}/traj.h5"
    )

    echo ">>> Running AIX [${binding}]..."
    local t0=$SECONDS
    "${cmd[@]}" < /dev/null > "${log_file}" 2>&1 || true
    local wall=$((SECONDS - t0))

    local s4=$(grep 'STEP_TIMING step=4 ' "${log_file}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p' || echo "")
    local s6=$(grep 'STEP_TIMING step=6 ' "${log_file}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p' || echo "")
    local mean=$(python3 -c "vals = [float(x) for x in ['$s4', '$s6'] if x]; print(f'{sum(vals)/len(vals):.2f}' if vals else 'FAILED')")
    local cub=$(ls -d "${scorep_dir}"/*.cubex 2>/dev/null | head -1 || echo "")
    echo "    AIX [${binding}]: step4=${s4:-?}ms step6=${s6:-?}ms mean=${mean}ms wall=${wall}s"
    echo "AIX,${binding},${s4:-},${s6:-},${mean},${wall},${cub}" >> "${RESULTS_CSV}"
}

run_phydll() {
    local binding="$1" # unbound | pinned
    local tag="PHYDLL_${binding}"
    local cfg_dir="${RUN_ROOT}/${tag}"
    local scorep_dir="${cfg_dir}/scorep"
    local log_file="${cfg_dir}/run.log"
    mkdir -p "${cfg_dir}"; rm -rf "${scorep_dir}"
    export SCOREP_EXPERIMENT_DIRECTORY="${scorep_dir}"
    export MLCOUPLING_INTRA_OP_THREADS=1 MLCOUPLING_INTER_OP_THREADS=1 PHYDLL_DL_COUNT=1 PHYDLL_DL_FIELD_COUNT=1

    local phy_exports=(
        "${SCOREP_EXPORTS[@]}"
        -x PHYDLL_IO_LOG -x PHYDLL_IO_LOG_BARRIERS -x PHYDLL_TELEMETRY
        -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS
        -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT
    )

    local cmd=(timeout --kill-after=30 "${RUN_TIMEOUT}" mpirun --bind-to none --oversubscribe "${phy_exports[@]}")
    if [[ "${binding}" == "pinned" ]]; then
        cmd+=(-n 96 "${PIN_WRAPPER}" "${SOLVER_BIN}")
    else
        cmd+=(-n 96 "${SOLVER_BIN}")
    fi
    cmd+=(
        "${COMMON_SOLVER_ARGS[@]}"
        --model-path "${MODEL_PATH_FLAT}" --model-io-layout flat_contiguous
        --cpp-ml-config config_phydll.toml
        --output-hdf5 "${cfg_dir}/traj.h5"
    )
    if [[ "${binding}" == "pinned" ]]; then
        cmd+=(: "${phy_exports[@]}" -n 96 "${PIN_WRAPPER}" "${PHYDLL_DL_CLIENT}")
    else
        cmd+=(: "${phy_exports[@]}" -n 96 "${PHYDLL_DL_CLIENT}")
    fi

    echo ">>> Running PhyDLL C++ [${binding}]..."
    local t0=$SECONDS
    "${cmd[@]}" < /dev/null > "${log_file}" 2>&1 || true
    local wall=$((SECONDS - t0))

    local s4=$(grep 'STEP_TIMING step=4 ' "${log_file}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p' || echo "")
    local s6=$(grep 'STEP_TIMING step=6 ' "${log_file}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p' || echo "")
    local mean=$(python3 -c "vals = [float(x) for x in ['$s4', '$s6'] if x]; print(f'{sum(vals)/len(vals):.2f}' if vals else 'FAILED')")
    local cub=$(ls -d "${scorep_dir}"/*.cubex 2>/dev/null | head -1 || echo "")
    echo "    PhyDLL [${binding}]: step4=${s4:-?}ms step6=${s6:-?}ms mean=${mean}ms wall=${wall}s"
    echo "PHYDLL_CPP,${binding},${s4:-},${s6:-},${mean},${wall},${cub}" >> "${RESULTS_CSV}"
}

run_smartsim() {
    local cfg_name="$1" # SS_TPQ96_I1_B96 or SS_DEFAULT
    local binding="$2"  # unbound | pinned
    local omit_threads="$3" tpq="$4" intra="$5" bind_cpus="$6" omit_set_cpus="$7"

    local tag="${cfg_name}_${binding}"
    local cfg_dir="${RUN_ROOT}/${tag}"
    local scorep_dir="${cfg_dir}/scorep"
    local log_file="${cfg_dir}/run.log"
    local db_host_file="${cfg_dir}/db_host.txt"
    local done_file="${cfg_dir}/close_driver.txt"
    mkdir -p "${cfg_dir}"; rm -rf "${scorep_dir}"
    rm -f "${done_file}" "${db_host_file}"
    export SCOREP_EXPERIMENT_DIRECTORY="${scorep_dir}"

    export SMARTSIM_OMIT_THREADS="${omit_threads}"
    export SMARTSIM_OMIT_SET_CPUS="${omit_set_cpus}"
    export SMARTSIM_THREADS_PER_QUEUE="${tpq}"
    export SMARTSIM_INTRA_OP_THREADS="${intra}"
    export SMARTSIM_INTER_OP_THREADS=1
    export RUN_ID_ENV="smoke_pin_${SLURM_JOB_ID}_${tag}"

    local port=$(( 7200 + (RANDOM % 800) ))
    (
        cd "${cfg_dir}"
        exec "${SMARTSIM_PYTHON}" "${MINI_APP_DIR}/smartsim_controller.py" \
            --db_nodes 1 --db_layout shared --port "${port}" \
            --cpu_cores_per_node "${bind_cpus}" \
            --hostname_file "${db_host_file}" < /dev/null > "${cfg_dir}/controller.log" 2>&1
    ) &
    local ctrl_pid=$!

    local wait_s=0
    while [[ ! -s "${db_host_file}" ]]; do
        if ! ps -p "${ctrl_pid}" >/dev/null 2>&1; then
            echo "    Controller failed to start"
            return 1
        fi
        sleep 1; wait_s=$((wait_s+1))
        if [[ ${wait_s} -ge ${DB_START_TIMEOUT} ]]; then
            echo "    DB startup timeout"
            kill -TERM "${ctrl_pid}" 2>/dev/null || true
            return 1
        fi
    done
    local ssdb=$(<"${db_host_file}")
    echo "    DB ready: ${ssdb}"

    local ss_exports=("${SCOREP_EXPORTS[@]}" -x SSDB)
    export SSDB="${ssdb}"

    local cmd=(timeout --kill-after=30 "${RUN_TIMEOUT}" mpirun --bind-to none --oversubscribe "${ss_exports[@]}")
    if [[ "${binding}" == "pinned" ]]; then
        cmd+=(-n 96 "${PIN_WRAPPER}" "${SOLVER_BIN}")
    else
        cmd+=(-n 96 "${SOLVER_BIN}")
    fi
    cmd+=(
        "${COMMON_SOLVER_ARGS[@]}"
        --model-path "${MODEL_PATH_SPLIT}" --model-io-layout split_3x3
        --cpp-ml-config config.toml
        --output-hdf5 "${cfg_dir}/traj.h5"
        --ml-db-batch-size 0 --ml-db-min-batch-size 0 --ml-db-min-batch-timeout 0
    )

    echo ">>> Running SmartSim ${cfg_name} [${binding}]..."
    local t0=$SECONDS
    "${cmd[@]}" < /dev/null > "${log_file}" 2>&1 || true
    local wall=$((SECONDS - t0))

    # Signal controller to shutdown cleanly
    touch "${cfg_dir}/close_driver_${RUN_ID_ENV}.txt"
    for ((j=0; j<30; j++)); do
        kill -0 "${ctrl_pid}" 2>/dev/null || break
        sleep 1
    done
    wait "${ctrl_pid}" 2>/dev/null || true
    rm -f "${cfg_dir}/close_driver_${RUN_ID_ENV}.txt"

    local s4=$(grep 'STEP_TIMING step=4 ' "${log_file}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p' || echo "")
    local s6=$(grep 'STEP_TIMING step=6 ' "${log_file}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p' || echo "")
    local mean=$(python3 -c "vals = [float(x) for x in ['$s4', '$s6'] if x]; print(f'{sum(vals)/len(vals):.2f}' if vals else 'FAILED')")
    local cub=$(ls -d "${scorep_dir}"/*.cubex 2>/dev/null | head -1 || echo "")
    echo "    SmartSim ${cfg_name} [${binding}]: step4=${s4:-?}ms step6=${s6:-?}ms mean=${mean}ms wall=${wall}s"
    echo "${cfg_name},${binding},${s4:-},${s6:-},${mean},${wall},${cub}" >> "${RESULTS_CSV}"
}

# Run the test matrix:
run_aix "unbound"
run_aix "pinned"

run_phydll "unbound"
run_phydll "pinned"

run_smartsim "SS_TPQ96_I1_B96" "unbound" 0 96 1 96 0
run_smartsim "SS_TPQ96_I1_B96" "pinned"  0 96 1 96 0

run_smartsim "SS_DEFAULT" "unbound" 1 1 1 96 1
run_smartsim "SS_DEFAULT" "pinned"  1 1 1 96 1

echo ""
echo "=========================================================="
echo "SMOKE PINNING SUMMARY (${MODEL})"
echo "=========================================================="
column -s, -t "${RESULTS_CSV}"
echo "=========================================================="

# Analyze Score-P breakdown
python3 - <<'PY'
import csv, os, sys
from pathlib import Path

# Add CPP-ML-Interface scripts to path
cmi_scripts = Path("/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/CPP-ML-Interface/scripts")
sys.path.insert(0, str(cmi_scripts))
try:
    from analyze_cmi_scorep_profiles import parse_cubex_file
except ImportError:
    print("Could not import Score-P parser, skipping profile breakdown.")
    sys.exit(0)

run_root = Path(os.environ.get("RUN_ROOT", "."))
results_csv = run_root / "results.csv"
if not results_csv.exists():
    sys.exit(0)

print("\n--- Detailed Score-P Breakdown (per-rank per-steady-step ms) ---")
print(f"{'Config':<18} {'Binding':<10} {'Mean Latency':<14} {'Key Profile Region (ms)':<40}")
print("-" * 85)

with results_csv.open() as f:
    for row in csv.DictReader(f):
        cfg = row["config"]
        binding = row["binding"]
        latency = row["mean_steady_ms"]
        cubex_path = row.get("cubex", "")
        if not cubex_path or not Path(cubex_path).exists():
            print(f"{cfg:<18} {binding:<10} {latency:<14} [No profile]")
            continue
        try:
            nodes_by_rank = parse_cubex_file(Path(cubex_path))
            steady_steps = 2
            # Find key regions
            if "AIX" in cfg:
                fwd_times = []
                for rank_nodes in nodes_by_rank:
                    for n in rank_nodes.values():
                        if n.name == "torchInference::forward" and n.visits:
                            fwd_times.append(n.incl_time * 1000.0 / steady_steps)
                mean_fwd = sum(fwd_times)/len(fwd_times) if fwd_times else 0
                detail = f"forward={mean_fwd:.2f}ms (n={len(fwd_times)} ranks)"
            elif "PHYDLL" in cfg:
                dl_fwd = []
                dl_send = []
                for rank_nodes in nodes_by_rank:
                    for n in rank_nodes.values():
                        if n.name == "dl_torch_forward" and n.visits:
                            dl_fwd.append(n.incl_time * 1000.0 / steady_steps)
                        if n.name == "dl_send_output" and n.visits:
                            dl_send.append(n.incl_time * 1000.0 / steady_steps)
                fwd_str = f"dl_fwd={sum(dl_fwd)/len(dl_fwd):.2f}ms" if dl_fwd else "dl_fwd=?"
                send_str = f"dl_send={sum(dl_send)/len(dl_send):.2f}ms" if dl_send else "dl_send=?"
                detail = f"{fwd_str}, {send_str}"
            elif "SS_" in cfg:
                run_model = []
                put_tensor = []
                for rank_nodes in nodes_by_rank:
                    for n in rank_nodes.values():
                        if n.name == "smartsim_run_model" and n.visits:
                            run_model.append(n.incl_time * 1000.0 / steady_steps)
                        if n.name == "smartsim_put_tensor" and n.visits:
                            put_tensor.append(n.incl_time * 1000.0 / steady_steps)
                rm_str = f"run_model={sum(run_model)/len(run_model):.2f}ms" if run_model else "run_model=?"
                put_str = f"put={sum(put_tensor)/len(put_tensor):.2f}ms" if put_tensor else "put=?"
                detail = f"{rm_str}, {put_str}"
            else:
                detail = ""
            print(f"{cfg:<18} {binding:<10} {latency:<14} {detail:<40}")
        except Exception as e:
            print(f"{cfg:<18} {binding:<10} {latency:<14} [Error parsing profile: {e}]")
print("-" * 85)
PY

echo "Smoke test complete at $(date)"
