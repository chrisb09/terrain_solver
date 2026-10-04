#!/bin/bash
#SBATCH --job-name=terrain_cpu_diagnostics
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --exclusive
#SBATCH --output=logs/terrain_cpu_diagnostics_%j.log

# One suite and model per allocation; never build or submit from this runner.
set -eo pipefail
MINI_APP_DIR=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
BASE_DIR=${MINI_APP_DIR%/mini_app}
CPP_DIR=${BASE_DIR}/CPP-ML-Interface
cd "${MINI_APP_DIR}"
: "${SLURM_JOB_ID:?Submit with sbatch}"
: "${SUITE:?Set SUITE=aix|phydll|smartsim}"
: "${MODEL:?Set MODEL=watercnn|giant}"
STEPS=${STEPS:-10}
RUN_TIMEOUT=${RUN_TIMEOUT:-1800}
DB_START_TIMEOUT=${DB_START_TIMEOUT:-180}
MEMORY_INTERVAL=${MEMORY_INTERVAL:-1.0}
HELPER_PYTHON=${HELPER_PYTHON:-/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python}
MONITOR_PYTHON=${MONITOR_PYTHON:-$HELPER_PYTHON}
export HELPER_PYTHON MONITOR_PYTHON
for python in "$HELPER_PYTHON" "$MONITOR_PYTHON"; do
    [[ $python == /* && -x $python ]] || { printf 'Python must be an absolute executable: %s\n' "$python" >&2; exit 1; }
done
HELPER=${MINI_APP_DIR}/terrain_cpu_diagnostics.py
jobname=${SLURM_JOB_NAME//[^A-Za-z0-9_.-]/_}
ROOT=${MINI_APP_DIR}/results/terrain_cpu_diagnostics/${jobname}_${SLURM_JOB_ID}
[[ ! -e "$ROOT" ]] || { printf 'Refusing existing root %s\n' "$ROOT" >&2; exit 1; }
mkdir -p "$ROOT"
"$HELPER_PYTHON" "$HELPER" --plan "$SUITE" --model "$MODEL" --cycles "${ABBA_CYCLES:-0}" \
    --repeats "${REPEATS:-2}" --steps "$STEPS" > "$ROOT/plan.tsv"
for value in "$RUN_TIMEOUT" "$DB_START_TIMEOUT"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || exit 1
done
[[ ${SLURM_JOB_NUM_NODES:-1} == 1 ]] || exit 1

unset USE_SCOREP TERRAIN_MEMORY_TOKEN SSDB
# Keep executable overrides, but do not inherit a previous benchmark's treatments.
while read -r name; do
    case "$name" in
        SMARTSIM_PYTHON|PHYDLL_DL_CLIENT) ;;
        SMARTSIM_*|SCOREP_*|MLCOUPLING_*|PHYDLL_*|OMP_*|MKL_*|OPENBLAS_*|KMP_*|GOMP_*|OMPI_*|PMIX_*) unset "$name" ;;
    esac
done < <(compgen -e)
source "${BASE_DIR}/set_env_claix23_cuda12.4.sh"
export CUDA_STUBS_DIR=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/stubs/lib64
export LD_LIBRARY_PATH="${CUDA_STUBS_DIR}:${LD_LIBRARY_PATH:-}"
export CMI_DIR=$CPP_DIR
export SMARTSIM_PAPI_ROOT=${CPP_DIR}/tmp/opencode/papi-7.2.0-install
export SMARTSIM_SCOREP_ROOT=${CPP_DIR}/tmp/opencode/scorep-8.4-papi72-install
source "${CPP_DIR}/env_scorep.sh"
export SCOREP_ENABLE_PROFILING=true SCOREP_ENABLE_TRACING=false SCOREP_METRIC_PAPI=
export SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY=false
export HDF5_USE_FILE_LOCKING=FALSE SR_MODEL_TIMEOUT=900000 SR_CMD_TIMEOUT=900000 SR_SOCKET_TIMEOUT=900000
export MLCOUPLING_MEM_LOG_VERBOSE=0 MLCOUPLING_MEM_LOG_COLLECTIVE=0 MLCOUPLING_STEP_TIMING_SYNC=1
export SMARTSIM_FLAG_TELEMETRY=0 PHYDLL_TELEMETRY=0
SOLVER_BIN=${SOLVER_BIN:-${MINI_APP_DIR}/solver_cpp/build_scorep_cmi_fresh/terrain_solver}
PHYDLL_DL_CLIENT=${PHYDLL_DL_CLIENT:-${CPP_DIR}/dl_clients/build-scorep/phydll_dl_client}
SMARTSIM_PYTHON=${SMARTSIM_PYTHON:-${CPP_DIR}/extern/python/smartsim_cuda-12/bin/python}
case "$MODEL" in
    watercnn) prep=$MINI_APP_DIR/input/prep_watercnn_4k.h5; total_bs=8294400 ;;
    giant) prep=$MINI_APP_DIR/input/prep_giant_240x192.h5; total_bs=46080 ;;
esac
flat=$MINI_APP_DIR/train_models/model_a/${MODEL}_cpu.pt
split=$MINI_APP_DIR/train_models/model_a/${MODEL}_split_cpu.pt
[[ -x "$SOLVER_BIN" ]] || { printf 'Missing executable: %s\n' "$SOLVER_BIN" >&2; exit 1; }
for file in "$prep" "$flat" "$split"; do [[ -f "$file" ]] || { printf 'Missing %s\n' "$file" >&2; exit 1; }; done
[[ $SUITE != phydll || -x $PHYDLL_DL_CLIENT ]] || exit 1
[[ $SUITE != smartsim || -x $SMARTSIM_PYTHON ]] || exit 1
sha256sum "$SOLVER_BIN" "$prep" "$flat" "$split" config_aix.toml config_phydll.toml config.toml \
    terrain_memory_sampler.py "$HELPER" > "$ROOT/inputs.sha256"
if [[ $SUITE == phydll ]]; then sha256sum "$PHYDLL_DL_CLIENT" >> "$ROOT/inputs.sha256"; fi
ldd "$SOLVER_BIN" > "$ROOT/solver_libraries.txt"
if [[ $SUITE == phydll ]]; then
    ldd "$PHYDLL_DL_CLIENT" > "$ROOT/dl_libraries.txt"
    "$HELPER_PYTHON" "$HELPER" --hash-phydll-libraries "$ROOT/solver_libraries.txt" "$ROOT/dl_libraries.txt" >> "$ROOT/inputs.sha256"
fi
env | sort > "$ROOT/allocation.env"
lscpu > "$ROOT/lscpu.txt"
taskset -pc $$ > "$ROOT/batch_affinity.txt"
scontrol show job "$SLURM_JOB_ID" > "$ROOT/slurm_job.txt"
printf 'model,suite,run,rep,config,condition,tpq,intra,db_batch,sampler,barriers,steps,wall_s,status,log,cubex\n' > "$ROOT/results.csv"
printf 'ROOT=%s\n' "$ROOT"

monitor_pid= ctrl_pid= dir= done_file= controller_active=0 active_run=0
declare -A baseline_steps=()
controller_alive() {
    local stat saved
    local -a fields
    [[ -r $dir/controller_start && -r /proc/$ctrl_pid/stat ]] || return 1
    read -r saved < "$dir/controller_start" || return 1
    read -r stat < "/proc/$ctrl_pid/stat" || return 1
    read -ra fields <<< "${stat##*) }"
    [[ ${fields[19]} == "$saved" && ${fields[0]} != Z && ${fields[0]} != X ]]
}
# No process-name kills. Only new numeric steps in this allocation are eligible.
current_steps() {
    local output step
    output=$(squeue --steps --noheader --jobs "$SLURM_JOB_ID" --format='%i' < /dev/null) || return 1
    while read -r step; do
        [[ $step =~ ^${SLURM_JOB_ID}\.[0-9]+$ ]] && printf '%s\n' "$step"
    done <<< "$output"
    return 0
}
cleanup() {
    local failed=0 step alive=0
    if [[ -n $monitor_pid ]]; then
        touch "$dir/memory.stop"
        wait "$monitor_pid" || failed=1
        monitor_pid=
    fi
    if (( controller_active )); then
        touch "$done_file"
        for ((j=0; j<60; j++)); do
            controller_alive || break
            sleep 1
        done
        # Capture even a step created during a failed/timed-out DB startup.
        local steps
        steps=$(current_steps) || failed=1
        while read -r step; do
            [[ -n $step && -z ${baseline_steps[$step]+yes} ]] || continue
            printf 'Controller step survived natural shutdown: %s\n' "$step" >> "$dir/cleanup.log"
            scancel "$step" < /dev/null || failed=1
            failed=1
        done <<< "$steps"
        if controller_alive; then
            kill -TERM "$ctrl_pid" 2>/dev/null || true
            failed=1
        fi
        wait "$ctrl_pid" || failed=1
        ctrl_pid=; controller_active=0
        rm -f "$done_file"
        # Do not proceed into another treatment while a cancelled DB step remains.
        for ((j=0; j<30; j++)); do
            alive=0
            steps=$(current_steps) || { failed=1; break; }
            while read -r step; do
                [[ -z $step || -n ${baseline_steps[$step]+yes} ]] || alive=1
            done <<< "$steps"
            (( alive )) || break
            sleep 1
        done
        (( ! alive )) || failed=1
    fi
    return "$failed"
}
record_run() {
    local status=$1 cubex= record_wall=${finished_wall:-$((SECONDS - t0))}
    shopt -s nullglob
    local profiles=("$dir"/scorep/*.cubex)
    (( ${#profiles[@]} != 1 )) || cubex=${profiles[0]}
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$MODEL" "$SUITE" "$run" "$rep" "$config" "$condition" "$tpq" "$intra" "$db_batch" \
        "$sampler" "$barriers" "$STEPS" "$record_wall" "$status" "$dir/solver.log" "$cubex" >> "$ROOT/results.csv"
    active_run=0
}
trap 'exit_rc=$?; trap - EXIT; cleanup || exit_rc=1; if (( active_run )); then record_run "FAILED_exit_$exit_rc"; fi; exit "$exit_rc"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
common=(--device CPU --gpus-per-node 0 --ml-nodes 1 --model-backend TORCH
    --input-hdf5 "$prep" --steps "$STEPS" --save-every "$STEPS" --save-mode periodic
    --triangular-scale 1 --chunk-size 12 --io-mode parallel_hdf5 --mpi-sync-mode none
    --hdf5-xfer-mode independent --rank-grid-x 12 --rank-grid-z 8 --ml-interface cpp
    --ml-batch-size 10000000)
exports=(-x LD_LIBRARY_PATH -x SCOREP_EXPERIMENT_DIRECTORY -x SCOREP_ENABLE_PROFILING
    -x SCOREP_ENABLE_TRACING -x SCOREP_METRIC_PAPI -x SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY
    -x MLCOUPLING_MEM_LOG_VERBOSE -x MLCOUPLING_MEM_LOG_COLLECTIVE -x MLCOUPLING_STEP_TIMING_SYNC)
while IFS=$'\t' read -r run rep config condition tpq intra sampler barriers; do
    dir=$ROOT/$run
    mkdir "$dir"
    : > "$dir/solver.log"
    export SCOREP_EXPERIMENT_DIRECTORY=$dir/scorep PHYDLL_IO_LOG_BARRIERS=$barriers
    unset TERRAIN_MEMORY_TOKEN SSDB
    db_batch=0; rc=0; t0=$SECONDS; finished_wall=; active_run=1
    if [[ $SUITE == aix ]]; then
        model_path=$flat; layout=flat_contiguous; cfg=config_aix.toml
        if [[ $sampler == 1 ]]; then
            export TERRAIN_MEMORY_TOKEN=terrain_cpu_diagnostics_${SLURM_JOB_ID}_${run}
            "$MONITOR_PYTHON" "$MINI_APP_DIR/terrain_memory_sampler.py" --output "$dir" \
                --token "$TERRAIN_MEMORY_TOKEN" --job-id "$SLURM_JOB_ID" --solver "$SOLVER_BIN" \
                --dl "$PHYDLL_DL_CLIENT" --interval "$MEMORY_INTERVAL" --expected-solver 96 \
                --expected-dl 0 --expected-db 0 --cleanup < /dev/null > "$dir/memory_monitor.log" 2>&1 &
            monitor_pid=$!
            for ((j=0; j<100; j++)); do
                [[ ! -f $dir/memory.ready ]] || break
                kill -0 "$monitor_pid" 2>/dev/null || { rc=1; break; }
                sleep 0.1
            done
            [[ -f $dir/memory.ready ]] || rc=1
        fi
    elif [[ $SUITE == phydll ]]; then
        model_path=$flat; layout=flat_contiguous; cfg=config_phydll.toml
        export PHYDLL_IO_LOG=1
        export MLCOUPLING_INTRA_OP_THREADS=1 MLCOUPLING_INTER_OP_THREADS=1 PHYDLL_DL_COUNT=1 PHYDLL_DL_FIELD_COUNT=1
    else
        model_path=$split; layout=split_3x3; cfg=config.toml
        [[ $condition == off ]] || db_batch=$((total_bs / tpq))
        export SMARTSIM_OMIT_THREADS=0 SMARTSIM_OMIT_SET_CPUS=0 SMARTSIM_THREADS_PER_QUEUE=$tpq
        export SMARTSIM_INTRA_OP_THREADS=$intra SMARTSIM_INTER_OP_THREADS=1
        export RUN_ID_ENV=terrain_cpu_diagnostics_${SLURM_JOB_ID}_${run}
        # Isolated cwd also isolates controller experiment directories and done files.
        done_file=$dir/close_driver_${RUN_ID_ENV}.txt
        baseline_steps=()
        before=$(current_steps)
        while read -r step; do [[ -z $step ]] || baseline_steps[$step]=1; done <<< "$before"
        port=$((7100 + SLURM_JOB_ID % 1000))
        controller=(timeout --kill-after=30 "$((RUN_TIMEOUT + DB_START_TIMEOUT + 120))" "$SMARTSIM_PYTHON"
            "$MINI_APP_DIR/smartsim_controller.py" --db_nodes 1 --db_layout shared --port "$port"
            --cpu_cores_per_node 96 --hostname_file "$dir/db_host.txt")
        printf '%q ' "${controller[@]}" > "$dir/controller_command.sh"; printf '\n' >> "$dir/controller_command.sh"
        (
            cd "$dir"
            # Save the wrapper's identity before exec; never signal a reused PID.
            read -r stat < /proc/self/stat
            read -ra fields <<< "${stat##*) }"
            printf '%s\n' "${fields[19]}" > controller_start
            exec "${controller[@]}" < /dev/null
        ) > "$dir/controller.log" 2>&1 &
        ctrl_pid=$!; controller_active=1
        for ((j=0; j<DB_START_TIMEOUT; j++)); do
            [[ ! -s $dir/db_host.txt ]] || break
            kill -0 "$ctrl_pid" 2>/dev/null || { rc=1; break; }
            sleep 1
        done
        if [[ -s $dir/db_host.txt ]]; then export SSDB=$(<"$dir/db_host.txt"); else rc=1; fi
    fi
    extra_exports=()
    [[ $sampler != 1 ]] || extra_exports+=(-x TERRAIN_MEMORY_TOKEN)
    if [[ $SUITE == smartsim ]]; then extra_exports+=(-x SSDB); fi
    if [[ $SUITE == phydll ]]; then
        extra_exports+=(-x PHYDLL_IO_LOG -x PHYDLL_IO_LOG_BARRIERS -x PHYDLL_TELEMETRY -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS
            -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT)
    fi
    command=(timeout --kill-after=30 "$RUN_TIMEOUT" mpirun --bind-to none --report-bindings --oversubscribe
        "${exports[@]}" "${extra_exports[@]}" -n 96 "$SOLVER_BIN" "${common[@]}"
        --model-path "$model_path" --model-io-layout "$layout" --cpp-ml-config "$MINI_APP_DIR/$cfg"
        --output-hdf5 "$dir/traj.h5")
    if [[ $SUITE == smartsim ]]; then
        command+=(--ml-db-batch-size "$db_batch" --ml-db-min-batch-size 0 --ml-db-min-batch-timeout 0)
    elif [[ $SUITE == phydll ]]; then
        # Exactly the same Score-P and treatment exports for both MPMD programs.
        command+=(: "${exports[@]}" "${extra_exports[@]}" -n 96 "$PHYDLL_DL_CLIENT")
    fi
    env | sort > "$dir/run.env"
    printf '%q ' "${command[@]}" > "$dir/command.sh"; printf '\n' >> "$dir/command.sh"
    t0=$SECONDS
    if (( rc == 0 )); then
        # MPI must not consume the remaining rows from the loop's plan.tsv stdin.
        "${command[@]}" < /dev/null > "$dir/solver.log" 2>&1 || rc=$?
    else
        printf 'Solver not launched: sampler/controller startup failed (rc=%s). See memory_monitor.log or controller.log.\n' "$rc" > "$dir/solver.log"
    fi
    wall=$((SECONDS - t0))
    cleanup || rc=1
    if (( rc == 0 )); then
        "$HELPER_PYTHON" "$HELPER" --validate-log "$dir/solver.log" --steps "$STEPS" < /dev/null || rc=1
        if [[ $sampler == 1 ]]; then
            "$HELPER_PYTHON" "$HELPER" --validate-memory "$dir/memory_summary.json" < /dev/null || rc=1
        fi
    fi
    shopt -s nullglob
    profiles=("$dir"/scorep/*.cubex)
    (( ${#profiles[@]} == 1 )) || rc=1
    status=SUCCESS; (( rc == 0 )) || status=FAILED_rc_$rc
    # Preserve solver-only wall time rather than the lifecycle/validation duration.
    finished_wall=$wall
    record_run "$status"
    printf '%s %s wall=%ss\n' "$run" "$status" "$wall"
    (( rc == 0 )) || exit "$rc"
done < "$ROOT/plan.tsv"
"$HELPER_PYTHON" "$HELPER" --root "$ROOT"
