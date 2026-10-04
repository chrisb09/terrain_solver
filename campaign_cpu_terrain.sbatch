#!/bin/bash
#SBATCH --job-name=cpu_terrain_nobatch
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --output=logs/terrain_nobatch_%j.log

# CPU terrain-solver production campaign.
# Default: both models, five repetitions, six configs, 22 steps, Score-P on.
# MODEL/REP select a single model/rep; CONFIGS selects a space-separated subset.
# See terrain_memory_README.md for usage and memory accounting limitations.
set -uo pipefail

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    [[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ && ${SLURM_ARRAY_TASK_ID} -lt 10 ]] || exit 1
    if [[ ${SLURM_ARRAY_TASK_ID} -lt 5 ]]; then MODEL=watercnn; else MODEL=giant; fi
    REP=$((SLURM_ARRAY_TASK_ID % 5 + 1))
fi
MODELS="${MODEL:-${MODELS:-watercnn giant}}"
FIRST_REP="${REP:-1}"
LAST_REP="${REP:-${REPS:-5}}"
CONFIGS="${CONFIGS:-AIX PHYDLL_CPP SS_DEFAULT SS_TPQ12_I8_B96 SS_TPQ96_I1_B96 SS_TPQ1_I96_B96}"
for cfg in ${CONFIGS}; do
    case "${cfg}" in AIX|PHYDLL_CPP|SS_DEFAULT|SS_TPQ12_I8_B96|SS_TPQ96_I1_B96|SS_TPQ1_I96_B96) ;; *) echo "Unknown config ${cfg}" >&2; exit 1 ;; esac
done

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

SOLVER_BIN="${SOLVER_BIN:-${MINI_APP_DIR}/solver_cpp/build_scorep_cmi_fresh/terrain_solver}"
SMARTSIM_PYTHON="${SMARTSIM_PYTHON:-${CPP_DIR}/extern/python/smartsim_cuda-12/bin/python}"
PHYDLL_DL_CLIENT="${PHYDLL_DL_CLIENT:-${CPP_DIR}/dl_clients/build-scorep/phydll_dl_client}"
# Requires a binary rebuilt from the current solver source. No timed memory MPI.
export MLCOUPLING_MEM_LOG_VERBOSE=0 MLCOUPLING_MEM_LOG_COLLECTIVE=0
MONITOR_PYTHON="${MONITOR_PYTHON:-python3}"
MEMORY_INTERVAL="${MEMORY_INTERVAL:-1.0}"
RUN_TIMEOUT="${RUN_TIMEOUT:-10800}"
DB_START_TIMEOUT="${DB_START_TIMEOUT:-180}"

MPI_RANKS="${MPI_RANKS:-96}"
RANK_GRID_X="${RANK_GRID_X:-12}"
RANK_GRID_Z="${RANK_GRID_Z:-8}"
STEPS="${STEPS:-22}"
for value in "${FIRST_REP}" "${LAST_REP}" "${STEPS}" "${MPI_RANKS}" "${RANK_GRID_X}" "${RANK_GRID_Z}" "${RUN_TIMEOUT}" "${DB_START_TIMEOUT}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "Expected positive integer: ${value}" >&2; exit 1; }
done
[[ ${FIRST_REP} -le ${LAST_REP} && $((RANK_GRID_X * RANK_GRID_Z)) -eq ${MPI_RANKS} ]] || exit 1
OUTPUT_BASE="${OUTPUT_BASE:-${MINI_APP_DIR}/results/terrain_nobatch}"
CAMPAIGN_ROOT="${RUN_ROOT:-${OUTPUT_BASE}/job_${SLURM_JOB_ID:-local}_$(date +%Y%m%dT%H%M%S)_$$}"
[[ ! -e "${CAMPAIGN_ROOT}" ]] || { echo "Refusing existing RUN_ROOT ${CAMPAIGN_ROOT}" >&2; exit 1; }
LOG_ROOT="${LOG_ROOT:-${CAMPAIGN_ROOT}/logs}"
mkdir -p "${CAMPAIGN_ROOT}" "${LOG_ROOT}" || exit 1
echo "CAMPAIGN_ROOT=${CAMPAIGN_ROOT} LOG_ROOT=${LOG_ROOT}"
selected() { [[ " ${CONFIGS} " == *" $1 "* ]]; }
monitor_pid= ctrl_pid= memory_stop= done_file=
stop_memory() {
    if [[ -n "${ctrl_pid}" ]]; then
        touch "${done_file}"
        for ((j=0; j<30; j++)); do
            kill -0 "${ctrl_pid}" 2>/dev/null || break
            sleep 1
        done
    fi
    if [[ -n "${monitor_pid}" ]]; then
        touch "${memory_stop}"
        wait "${monitor_pid}" || { echo "Memory sampler failed" >&2; exit 1; }
        monitor_pid=
    fi
    if [[ -n "${ctrl_pid}" ]]; then
        # The sampler cleans up surviving controller/DB PID-start identities.
        # Do not signal a bare PID that Bash may already have reaped and reused.
        wait "${ctrl_pid}" 2>/dev/null || true
        ctrl_pid=
    fi
    [[ -z "${done_file}" ]] || rm -f "${done_file}"
}
start_memory() {
    local dir="$1" token="$2" port="${3:-0}"
    local expected_dl=0 expected_db=0
    [[ "${dir##*/}" != PHYDLL_CPP ]] || expected_dl="${MPI_RANKS}"
    [[ "${port}" == 0 ]] || expected_db=1
    memory_stop="${dir}/memory.stop"
    export TERRAIN_MEMORY_TOKEN="${token}"
    "${MONITOR_PYTHON}" terrain_memory_sampler.py --output "${dir}" --token "${token}" \
        --job-id "${SLURM_JOB_ID:-}" --solver "${SOLVER_BIN}" --dl "${PHYDLL_DL_CLIENT}" \
        --db-port "${port}" --interval "${MEMORY_INTERVAL}" --expected-solver "${MPI_RANKS}" \
        --expected-dl "${expected_dl}" --expected-db "${expected_db}" --cleanup > "${dir}/memory_monitor.log" 2>&1 &
    monitor_pid=$!
    for ((j=0; j<100; j++)); do
        [[ ! -f "${dir}/memory.ready" ]] || break
        kill -0 "${monitor_pid}" 2>/dev/null || { echo "Sampler startup failed" >&2; exit 1; }
        sleep 0.1
    done
    [[ -f "${dir}/memory.ready" ]] || { echo "Sampler readiness timeout" >&2; exit 1; }
}
trap stop_memory EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
failures=0
for MODEL in ${MODELS}; do
for ((REP=FIRST_REP; REP<=LAST_REP; REP++)); do
unset MLCOUPLING_INTRA_OP_THREADS MLCOUPLING_INTER_OP_THREADS PHYDLL_DL_COUNT PHYDLL_DL_FIELD_COUNT SSDB

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

RUN_ROOT="${CAMPAIGN_ROOT}/${MODEL}/rep${REP}"
mkdir -p "${RUN_ROOT}" logs
RESULTS_CSV="${RUN_ROOT}/results.csv"
echo "model,rep,config,step4_warm_s,wall_s,rank_min_ms,rank_max_ms,status,cubex" > "${RESULTS_CSV}"

echo "=========================================================="
echo "CPU Terrain Solver Campaign | MODEL=${MODEL} REP=${REP}"
echo "Job ${SLURM_JOB_ID:-$$} on $(hostname) at $(date)"
echo "RUN_ROOT=${RUN_ROOT}"
echo "=========================================================="

for f in "${SOLVER_BIN}" "${MODEL_PATH_FLAT}" "${MODEL_PATH_SPLIT}" "${PREP_H5}"; do
    [[ -f "${f}" ]] || { echo "Error: missing ${f}" >&2; exit 1; }
done
if selected PHYDLL_CPP; then
    [[ -x "${PHYDLL_DL_CLIENT}" ]] || { echo "Missing DL client ${PHYDLL_DL_CLIENT}" >&2; exit 1; }
fi

COMMON_SOLVER_ARGS=(
    --device CPU --gpus-per-node 0 --ml-nodes 1
    --model-backend TORCH
    --input-hdf5 "${PREP_H5}"
    --steps "${STEPS}" --save-every "${STEPS}" --save-mode periodic
    --triangular-scale 1 --chunk-size 12 --io-mode parallel_hdf5
    --mpi-sync-mode none --hdf5-xfer-mode independent
    --rank-grid-x "${RANK_GRID_X}" --rank-grid-z "${RANK_GRID_Z}"
    --ml-interface cpp
)

record() {  # name step4 wall min max status cubex
    echo "${MODEL},${REP},$1,$2,$3,$4,$5,$6,$7" >> "${RESULTS_CSV}"
    [[ "$6" == SUCCESS ]] || failures=$((failures+1))
}

# --------------------------------------------------------------
run_smartsim_config() {
    local cfg_name="$1" omit_threads="$2" tpq="$3" intra="$4" bind_cpus="$5"
    local omit_set_cpus="$6"
    selected "${cfg_name}" || return 0

    local cfg_dir="${RUN_ROOT}/${cfg_name}"
    local scorep_dir="${cfg_dir}/scorep"
    mkdir -p "${cfg_dir}"; rm -rf "${scorep_dir}"
    local log_file="${LOG_ROOT}/terrain_nobatch_${MODEL}_r${REP}_${cfg_name}_${SLURM_JOB_ID:-$$}.log"
    local db_host_file="${cfg_dir}/db_host.txt"
    local run_id="terrain_nobatch_${MODEL}_r${REP}_${cfg_name}_${SLURM_JOB_ID:-$$}_$$"
    done_file="${MINI_APP_DIR}/close_driver_${run_id}.txt"
    rm -f "${done_file}" "${db_host_file}"

    echo ""
    echo ">>> ${cfg_name}: TPQ=${tpq} INTRA=${intra} BIND=${bind_cpus} omit_threads=${omit_threads} omit_set_cpus=${omit_set_cpus} db_batch=0 min_batch=0 timeout=0"

    export SMARTSIM_OMIT_THREADS="${omit_threads}"
    export SMARTSIM_OMIT_SET_CPUS="${omit_set_cpus}"
    export SMARTSIM_THREADS_PER_QUEUE="${tpq}"
    export SMARTSIM_INTRA_OP_THREADS="${intra}"
    export SMARTSIM_INTER_OP_THREADS=1
    export RUN_ID_ENV="${run_id}"

    local port=$(( 7100 + (RANDOM % 1000) ))
    start_memory "${cfg_dir}" "${run_id}" "${port}"
    SMARTSIM_FLAG_TELEMETRY=0 "${SMARTSIM_PYTHON}" smartsim_controller.py \
        --db_nodes 1 --db_layout shared --port "${port}" \
        --cpu_cores_per_node "${bind_cpus}" \
        --hostname_file "${db_host_file}" > "${LOG_ROOT}/controller_${run_id}.log" 2>&1 &
    ctrl_pid=$!

    local wait_s=0
    while [[ ! -s "${db_host_file}" ]]; do
        if ! ps -p "${ctrl_pid}" >/dev/null 2>&1; then
            echo "    Controller exited prematurely"; record "${cfg_name}" "" "" "" "" "FAILED(controller)" ""
            stop_memory
            return 1
        fi
        sleep 1; wait_s=$((wait_s+1))
        if [[ ${wait_s} -ge ${DB_START_TIMEOUT} ]]; then
            record "${cfg_name}" "" "" "" "" "FAILED(db-timeout)" ""
            stop_memory
            return 1
        fi
    done
    SSDB=$(<"${db_host_file}"); export SSDB
    echo "    DB ready: ${SSDB}"

    local batch_args=(--ml-batch-size 10000000 --ml-db-batch-size 0 --ml-db-min-batch-size 0 --ml-db-min-batch-timeout 0)

    export SCOREP_EXPERIMENT_DIRECTORY="${scorep_dir}"
    local t0=$SECONDS
    timeout --kill-after=30 "${RUN_TIMEOUT}" mpirun --bind-to none --oversubscribe -x LD_LIBRARY_PATH -x SCOREP_EXPERIMENT_DIRECTORY \
        -x TERRAIN_MEMORY_TOKEN -x MLCOUPLING_MEM_LOG_VERBOSE -x MLCOUPLING_MEM_LOG_COLLECTIVE \
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

    stop_memory

    if [[ ${rc} -eq 0 ]]; then
        local wm=$(grep 'STEP_TIMING step=4 ' "${log_file}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p')
        local ws=""; [[ -z "${wm}" ]] || ws=$(awk "BEGIN {printf \"%.2f\", ${wm} / 1000.0}")
        local mn=$(grep 'STEP_TIMING step=4 ' "${log_file}" | sed -n 's/.*rank_min_ms=\([0-9.]*\).*/\1/p')
        local mx=$(grep 'STEP_TIMING step=4 ' "${log_file}" | sed -n 's/.*rank_max_ms=\([0-9.]*\).*/\1/p')
        local cub=$(ls -d "${scorep_dir}"/*.cubex 2>/dev/null | head -1 | xargs -r basename)
        echo "    SUCCESS step4=${ws}s wall=${wall}s rank ${mn:-?}/${mx:-?} ms"
        record "${cfg_name}" "${ws}" "${wall}" "${mn}" "${mx}" "SUCCESS" "${cub}"
    else
        echo "    FAILED rc=${rc} wall=${wall}s (see ${log_file})"
        record "${cfg_name}" "" "${wall}" "" "" "FAILED(rc=${rc})" ""
    fi
}

# --------------------------------------------------------------
# 1. AIX
if selected AIX; then
echo ""; echo ">>> AIX"
AIX_DIR="${RUN_ROOT}/AIX"; mkdir -p "${AIX_DIR}"; rm -rf "${AIX_DIR}/scorep"
LOG_AIX="${LOG_ROOT}/terrain_nobatch_${MODEL}_r${REP}_AIX_${SLURM_JOB_ID:-$$}.log"
start_memory "${AIX_DIR}" "terrain_nobatch_${MODEL}_r${REP}_AIX_${SLURM_JOB_ID:-$$}_$$"
export SCOREP_EXPERIMENT_DIRECTORY="${AIX_DIR}/scorep"
t0=$SECONDS
timeout --kill-after=30 "${RUN_TIMEOUT}" mpirun --bind-to none --oversubscribe -x LD_LIBRARY_PATH -x SCOREP_EXPERIMENT_DIRECTORY \
    -x TERRAIN_MEMORY_TOKEN -x MLCOUPLING_MEM_LOG_VERBOSE -x MLCOUPLING_MEM_LOG_COLLECTIVE \
    -x SCOREP_ENABLE_PROFILING -x SCOREP_ENABLE_TRACING -x SCOREP_METRIC_PAPI \
    -x SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY \
    -n "${MPI_RANKS}" "${SOLVER_BIN}" "${COMMON_SOLVER_ARGS[@]}" \
    --model-path "${MODEL_PATH_FLAT}" --model-io-layout flat_contiguous \
    --cpp-ml-config config_aix.toml --ml-batch-size 10000000 \
    --output-hdf5 "${AIX_DIR}/traj.h5" > "${LOG_AIX}" 2>&1
rc=$?; wall=$((SECONDS - t0))
stop_memory
if [[ ${rc} -eq 0 ]]; then
    wm=$(grep 'STEP_TIMING step=4 ' "${LOG_AIX}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p')
    ws=""; [[ -z "${wm}" ]] || ws=$(awk "BEGIN {printf \"%.2f\", ${wm} / 1000.0}")
    mn=$(grep 'STEP_TIMING step=4 ' "${LOG_AIX}" | sed -n 's/.*rank_min_ms=\([0-9.]*\).*/\1/p')
    mx=$(grep 'STEP_TIMING step=4 ' "${LOG_AIX}" | sed -n 's/.*rank_max_ms=\([0-9.]*\).*/\1/p')
    cub=$(ls -d "${AIX_DIR}"/scorep/*.cubex 2>/dev/null | head -1 | xargs -r basename)
    echo "    SUCCESS step4=${ws}s wall=${wall}s rank ${mn:-?}/${mx:-?} ms"
    record "AIX" "${ws}" "${wall}" "${mn}" "${mx}" "SUCCESS" "${cub}"
else
    echo "    FAILED rc=${rc} wall=${wall}s"
    record "AIX" "" "${wall}" "" "" "FAILED(rc=${rc})" ""
fi
fi

# --------------------------------------------------------------
# 2. PHYDLL_CPP
if selected PHYDLL_CPP; then
echo ""; echo ">>> PHYDLL_CPP"
PHY_DIR="${RUN_ROOT}/PHYDLL_CPP"; mkdir -p "${PHY_DIR}"; rm -rf "${PHY_DIR}/scorep"
LOG_PHY="${LOG_ROOT}/terrain_nobatch_${MODEL}_r${REP}_PHYDLL_CPP_${SLURM_JOB_ID:-$$}.log"
start_memory "${PHY_DIR}" "terrain_nobatch_${MODEL}_r${REP}_PHYDLL_CPP_${SLURM_JOB_ID:-$$}_$$"
export MLCOUPLING_INTRA_OP_THREADS=1 MLCOUPLING_INTER_OP_THREADS=1 PHYDLL_DL_COUNT=1 PHYDLL_DL_FIELD_COUNT=1
export SCOREP_EXPERIMENT_DIRECTORY="${PHY_DIR}/scorep"
t0=$SECONDS
timeout --kill-after=30 "${RUN_TIMEOUT}" mpirun --bind-to none --oversubscribe \
    -x TERRAIN_MEMORY_TOKEN -x MLCOUPLING_MEM_LOG_VERBOSE -x MLCOUPLING_MEM_LOG_COLLECTIVE \
    -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
    -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
    -x SCOREP_EXPERIMENT_DIRECTORY -x SCOREP_ENABLE_PROFILING -x SCOREP_ENABLE_TRACING \
    -x SCOREP_METRIC_PAPI -x SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY \
    -n "${MPI_RANKS}" "${SOLVER_BIN}" "${COMMON_SOLVER_ARGS[@]}" \
    --model-path "${MODEL_PATH_FLAT}" --model-io-layout flat_contiguous \
    --cpp-ml-config config_phydll.toml --ml-batch-size 10000000 \
    --output-hdf5 "${PHY_DIR}/traj.h5" \
    : -x LD_LIBRARY_PATH -x TERRAIN_MEMORY_TOKEN -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
    -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
    -n "${MPI_RANKS}" "${PHYDLL_DL_CLIENT}" > "${LOG_PHY}" 2>&1
rc=$?; wall=$((SECONDS - t0))
stop_memory
if [[ ${rc} -eq 0 ]]; then
    wm=$(grep 'STEP_TIMING step=4 ' "${LOG_PHY}" | sed -n 's/.*step_ms=\([0-9.]*\).*/\1/p')
    ws=""; [[ -z "${wm}" ]] || ws=$(awk "BEGIN {printf \"%.2f\", ${wm} / 1000.0}")
    mn=$(grep 'STEP_TIMING step=4 ' "${LOG_PHY}" | sed -n 's/.*rank_min_ms=\([0-9.]*\).*/\1/p')
    mx=$(grep 'STEP_TIMING step=4 ' "${LOG_PHY}" | sed -n 's/.*rank_max_ms=\([0-9.]*\).*/\1/p')
    cub=$(ls -d "${PHY_DIR}"/scorep/*.cubex 2>/dev/null | head -1 | xargs -r basename)
    echo "    SUCCESS step4=${ws}s wall=${wall}s rank ${mn:-?}/${mx:-?} ms"
    record "PHYDLL_CPP" "${ws}" "${wall}" "${mn}" "${mx}" "SUCCESS" "${cub}"
else
    echo "    FAILED rc=${rc} wall=${wall}s"
    record "PHYDLL_CPP" "" "${wall}" "" "" "FAILED(rc=${rc})" ""
fi
fi

# --------------------------------------------------------------
# 3-6. RedisAI batching disabled explicitly in every SmartSim config.
unset MLCOUPLING_INTRA_OP_THREADS MLCOUPLING_INTER_OP_THREADS PHYDLL_DL_COUNT PHYDLL_DL_FIELD_COUNT
run_smartsim_config "SS_DEFAULT"      1 1  1  96 1
run_smartsim_config "SS_TPQ12_I8_B96" 0 12 8  96 0
run_smartsim_config "SS_TPQ96_I1_B96" 0 96 1  96 0
run_smartsim_config "SS_TPQ1_I96_B96" 0 1  96 96 0

# --------------------------------------------------------------
echo ""
echo "=========================================================="
echo "CAMPAIGN SUMMARY | MODEL=${MODEL} REP=${REP}"
echo "=========================================================="
cat "${RESULTS_CSV}"
echo "=========================================================="
echo "Completed at $(date)"
done
done
echo "Campaign failures=${failures}"
[[ ${failures} -eq 0 ]]
