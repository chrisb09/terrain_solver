#!/bin/bash
#SBATCH --job-name=terrain_cpu_final
#SBATCH --account=rwth2150
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --array=0-9%2
#SBATCH --output=logs/terrain_cpu_final_%A_%a.log

set -eo pipefail
MINI_APP_DIR=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
BASE_DIR=${MINI_APP_DIR%/mini_app}
CPP_DIR=$BASE_DIR/CPP-ML-Interface
HELPER_PYTHON=$CPP_DIR/extern/python/smartsim_cuda-12/bin/python
MODE=${MODE:-performance}
DATASET_ID=${DATASET_ID:-final}
[[ $DATASET_ID =~ ^[A-Za-z0-9_-]+$ ]] || exit 1
case $MODE in
    performance) STEPS=${STEPS:-22}; CONFIGS=${CONFIGS:-AIx PHYDLL_CPP SS_DEFAULT SS_TPQ1_I96_B96 SS_TPQ12_I8_B96 SS_TPQ96_I1_B96} ;;
    memory) STEPS=${STEPS:-6}; CONFIGS=${CONFIGS:-SS_TPQ1_I96_B96 SS_TPQ12_I8_B96 PHYDLL_CPP} ;;
    timing) STEPS=${STEPS:-22}; CONFIGS=${CONFIGS:-AIx SS_DEFAULT SS_TPQ1_I96_B96 SS_TPQ12_I8_B96 SS_TPQ96_I1_B96} ;;
    *) exit 1 ;;
esac
[[ $STEPS =~ ^[0-9]+$ ]] && (( STEPS >= 6 && STEPS % 2 == 0 )) || exit 1
task=${SLURM_ARRAY_TASK_ID:-0}
[[ $task =~ ^[0-9]$ ]] || exit 1
if (( task < 5 )); then MODEL=watercnn; total_bs=8294400; prep=input/prep_watercnn_4k.h5
else MODEL=giant; total_bs=46080; prep=input/prep_giant_240x192.h5; fi
REP=$((task % 5 + 1))
declare -A selected=()
for config in $CONFIGS; do
    case $config in AIx|PHYDLL_CPP|SS_DEFAULT|SS_TPQ1_I96_B96|SS_TPQ12_I8_B96|SS_TPQ96_I1_B96) ;; *) exit 1 ;; esac
    [[ -z ${selected[$config]+yes} ]] || exit 1
    selected[$config]=1
done
[[ ${#selected[@]} -gt 0 ]] || exit 1
# This offline plan path never loads modules or launches a process in the allocation.
if [[ ${PLAN_ONLY:-0} == 1 ]]; then
    printf 'mode=%s model=%s rep=%s steps=%s configs=%s\n' "$MODE" "$MODEL" "$REP" "$STEPS" "$CONFIGS"
    exit 0
fi
: "${SLURM_JOB_ID:?Submit with sbatch}"
[[ ${SLURM_JOB_NUM_NODES:-1} == 1 && -x $HELPER_PYTHON ]] || exit 1
RUN_TIMEOUT=${RUN_TIMEOUT:-3000}
DB_START_TIMEOUT=${DB_START_TIMEOUT:-180}
MEMORY_INTERVAL=${MEMORY_INTERVAL:-1.0}
for value in "$RUN_TIMEOUT" "$DB_START_TIMEOUT"; do [[ $value =~ ^[1-9][0-9]*$ ]] || exit 1; done
cd "$MINI_APP_DIR"
while read -r name; do
    case $name in
        SMARTSIM_PYTHON|PHYDLL_DL_CLIENT) ;;
        SMARTSIM_*|SCOREP_*|MLCOUPLING_*|PHYDLL_*|OMP_*|MKL_*|OPENBLAS_*|KMP_*|GOMP_*|OMPI_*|PMIX_*) unset "$name" ;;
    esac
done < <(compgen -e)
unset PYTHONHOME PYTHONPATH PYTHONOPTIMIZE LD_PRELOAD
unset TERRAIN_MEMORY_TOKEN SSDB USE_SCOREP
source "$BASE_DIR/set_env_claix23_cuda12.4.sh"
: "${LD_LIBRARY_PATH:?Missing library path}"
for name in EBROOTOPENSSL EBROOTCLANG EBROOTGCCCORE EBROOTOPENMPI EBROOTHDF5 EBROOTPNETCDF EBROOTCUDA EBROOTCUDNN EBROOTIMKL EBROOTHWLOC; do
    root=${!name:-}
    [[ -n $root && -d $root ]] || { printf 'Missing module root: %s\n' "$name" >&2; exit 1; }
done
for command in mpirun timeout ldd; do command -v "$command" > /dev/null; done
export LD_LIBRARY_PATH="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/stubs/lib64:${LD_LIBRARY_PATH:-}"
export CMI_DIR=$CPP_DIR SMARTSIM_PAPI_ROOT=$CPP_DIR/tmp/opencode/papi-7.2.0-install
export SMARTSIM_SCOREP_ROOT=$CPP_DIR/tmp/opencode/scorep-8.4-papi72-install
if [[ $MODE == timing ]]; then
    [[ -z ${TERRAIN_RETEST_BUILD:-} ]] || exit 1
    export TERRAIN_DB_LIBRARY_PATH=$LD_LIBRARY_PATH
    source "$MINI_APP_DIR/terrain_cpu_plain_gate.sh"
else
    source "$CPP_DIR/env_scorep.sh"
    if [[ -n ${TERRAIN_RETEST_BUILD:-} ]]; then
        source "$MINI_APP_DIR/terrain_phydll_retest_gate.sh"
    fi
    export SCOREP_ENABLE_PROFILING=true SCOREP_ENABLE_TRACING=false SCOREP_METRIC_PAPI=
    export SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY=false
fi
export MLCOUPLING_MEM_LOG_VERBOSE=0 MLCOUPLING_MEM_LOG_COLLECTIVE=0 MLCOUPLING_STEP_TIMING_SYNC=1
export PHYDLL_IO_LOG=0 PHYDLL_IO_LOG_BARRIERS=0 PHYDLL_TELEMETRY=0 SMARTSIM_FLAG_TELEMETRY=0
export HDF5_USE_FILE_LOCKING=FALSE SR_MODEL_TIMEOUT=900000 SR_CMD_TIMEOUT=900000 SR_SOCKET_TIMEOUT=900000
SOLVER_BIN=${SOLVER_BIN:-$MINI_APP_DIR/solver_cpp/build_scorep_cmi_fresh/terrain_solver}
PHYDLL_DL_CLIENT=${PHYDLL_DL_CLIENT:-$CPP_DIR/dl_clients/build-scorep/phydll_dl_client}
SMARTSIM_PYTHON=${SMARTSIM_PYTHON:-$HELPER_PYTHON}
ROOT=$MINI_APP_DIR/results/terrain_cpu_final/$DATASET_ID/$MODE/job_${SLURM_JOB_ID}_${task}
[[ ! -e $ROOT ]] || exit 1
mkdir -p "$ROOT/$MODEL/rep$REP"
RUN_ROOT=$ROOT/$MODEL/rep$REP
export MODE MODEL REP STEPS ROOT RUN_ROOT SOLVER_BIN PHYDLL_DL_CLIENT CPP_DIR MINI_APP_DIR
ldd "$SOLVER_BIN" < /dev/null > "$ROOT/solver_libraries.txt"
ldd "$PHYDLL_DL_CLIENT" < /dev/null > "$ROOT/dl_libraries.txt"
# Pin the reviewed rebuilt C library; the same library must resolve for both programs.
# APPROVED_PHYDLL_SHA256 is deliberately not named PHYDLL_* (treatment env is cleared).
"$HELPER_PYTHON" - "$ROOT" "$prep" < /dev/null <<'PY'
import hashlib, json, os, re, sys
from pathlib import Path
root = Path(sys.argv[1])
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
libraries = []
for name in ('solver', 'dl'):
    text = (root / f'{name}_libraries.txt').read_text()
    if 'not found' in text:
        raise ValueError('Unresolved loaded library')
    matches = re.findall(r'^\s*libphydll\.so(?:\.\d+)*\s+=>\s+(\S+)', text, re.M)
    if len(matches) != 1:
        raise ValueError('Expected one resolved C PhyDLL library')
    libraries.append(Path(matches[0]).resolve())
if libraries[0] != libraries[1]:
    raise ValueError('MPMD library mismatch')
library = libraries[0]
approved = os.environ.get('APPROVED_PHYDLL_SHA256', '')
if not re.fullmatch('[0-9a-f]{64}', approved) or sha(library) != approved:
    raise ValueError('Set APPROVED_PHYDLL_SHA256 to the reviewed logging-switch library hash')
binary = library.read_bytes()
if any(name not in binary for name in (b'PHYDLL_IO_LOG\0', b'PHYDLL_IO_LOG_BARRIERS\0')):
    raise ValueError('Loaded library lacks logging switches')
source = Path(os.environ['CPP_DIR']) / 'extern/phydll/src/core/io.c'
text = source.read_text()
for name in ('io_log_enabled', 'io_log_barriers_enabled'):
    body = re.search(r'static bool ' + name + r'\(void\)\s*\{([^}]+)\}', text).group(1)
    if 'value != NULL && strcmp(value, "1") == 0' not in body:
        raise ValueError('Reviewed C logging defaults changed')
mini = Path(os.environ['MINI_APP_DIR'])
files = [library, source, Path(os.environ['SOLVER_BIN']), Path(os.environ['PHYDLL_DL_CLIENT']),
         mini / sys.argv[2], mini / 'terrain_cpu_final.sbatch', mini / 'terrain_cpu_diagnostics.py',
         mini / 'terrain_memory_sampler.py', mini / 'smartsim_controller.py', mini / 'pin_wrapper.sh']
files += [mini / 'train_models/model_a' / f'{os.environ["MODEL"]}{suffix}_cpu.pt' for suffix in ('', '_split')]
files += [mini / name for name in ('config.toml', 'config_aix.toml', 'config_phydll.toml')]
if os.environ['MODE'] == 'timing':
    build = Path(os.environ['TERRAIN_PLAIN_BUILD'])
    provenance = Path(os.environ['TERRAIN_PLAIN_PROVENANCE'])
    files += [build / 'BUILD_READY', build / 'CMakeCache.txt', build / 'compile_commands.json',
              provenance / 'source.sha256', provenance / 'binaries.sha256',
              mini / 'terrain_cpu_plain_gate.sh', mini / 'terrain_cpu_missing.sbatch',
              mini / 'config_terrain_phydll_false.toml', mini / 'config_terrain_phydll_true.toml']
if os.environ.get('TERRAIN_RETEST_BUILD'):
    build = Path(os.environ['TERRAIN_RETEST_BUILD'])
    files += [Path(os.environ['TERRAIN_PHYDLL_CONFIG']), build / 'BUILD_READY',
              build / 'provenance/source.sha256', build / 'provenance/binaries.sha256',
              mini / 'terrain_phydll_retest.sbatch', mini / 'terrain_phydll_retest_gate.sh',
              mini / 'validate_terrain_phydll_profile.py']
hashes = {str(p.resolve()): sha(p) for p in files}
(root / 'inputs.sha256').write_text(''.join(f'{h}  {p}\n' for p, h in hashes.items()))
(root / 'provenance.json').write_text(json.dumps(dict(inputs=hashes, phydll_default_logging=0,
    phydll_default_barriers=0, approved_phydll_sha256=approved,
    device='CPU', mode=os.environ['MODE'], memory_sampler=int(os.environ['MODE'] == 'memory'),
    phydll_logging=0, phydll_io_log=0, phydll_io_log_barriers=0, internal_memory=0,
    internal_mem_verbose=0, internal_mem_collective=0, telemetry=0, timing_sync=1,
    solver_ranks=96, rank_grid='12x8', db_min_batch=0, db_timeout=0,
    policy='final_total_per_queue_except_default_and_TPQ96',
    helper_python=sys.executable, environment=dict(os.environ),
    retest_build=os.environ.get('TERRAIN_RETEST_BUILD', ''),
    plain_build=os.environ.get('TERRAIN_PLAIN_BUILD', '') if os.environ['MODE'] == 'timing' else '',
    scorep=int(os.environ['MODE'] != 'timing'),
    solver_readiness_wait=os.environ.get('TERRAIN_READINESS', 'false')), indent=2) + '\n')
PY
env | sort > "$ROOT/allocation.env"
sha256sum "$ROOT/allocation.env" > "$ROOT/environment.sha256"
lscpu < /dev/null > "$ROOT/lscpu.txt"
scontrol show job "$SLURM_JOB_ID" < /dev/null > "$ROOT/slurm_job.txt"

ctrl_pid= monitor_pid= dir= active=0
declare -A baseline_steps=()
current_steps() {
    local output step
    output=$(squeue --steps --noheader --jobs "$SLURM_JOB_ID" --format='%i' < /dev/null) || return 1
    while read -r step; do
        [[ $step =~ ^${SLURM_JOB_ID}\.[0-9]+$ ]] && printf '%s\n' "$step"
    done <<< "$output"
    return 0
}
controller_alive() {
    local stat saved
    local -a fields
    [[ -r $dir/controller_start && -r /proc/$ctrl_pid/stat ]] || return 1
    read -r saved < "$dir/controller_start" || return 1
    read -r stat < "/proc/$ctrl_pid/stat" || return 1
    read -ra fields <<< "${stat##*) }"
    [[ ${fields[19]} == "$saved" && ${fields[0]} != Z && ${fields[0]} != X ]]
}
cleanup() {
    local failed=0 steps step alive=0 j
    if [[ -n $ctrl_pid ]]; then
        touch "$done_file"
        for ((j=0; j<60; j++)); do controller_alive || break; sleep 1; done
        steps=$(current_steps) || failed=1
        while read -r step; do
            [[ -n $step && -z ${baseline_steps[$step]+yes} ]] || continue
            scancel "$step" < /dev/null || true
            failed=1
        done <<< "$steps"
        if controller_alive; then kill -TERM "$ctrl_pid" 2>/dev/null || true; failed=1; fi
        wait "$ctrl_pid" || failed=1
        ctrl_pid=
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
        rm -f "$done_file"
    fi
    # Sampling includes natural DB shutdown. Only then may tracked survivors be cleaned.
    if [[ -n $monitor_pid ]]; then
        touch "$dir/memory.stop"
        wait "$monitor_pid" || failed=1
        monitor_pid=
    fi
    return "$failed"
}
record() {
    "$HELPER_PYTHON" - "$dir" "$status" "$wall" "$tpq" "$intra" "$db_batch" "$omit" < /dev/null <<'PY'
import csv, hashlib, os, sys
from decimal import Decimal
from pathlib import Path
from terrain_cpu_diagnostics import timings
directory, status, wall, tpq, intra, batch, omit = sys.argv[1:]
d = Path(directory).resolve()
root = Path(os.environ['ROOT']).resolve()
run_environment = d / 'run.env'
profiles = list((d / 'scorep').glob('*.cubex'))
row = dict(model=os.environ['MODEL'], rep=int(os.environ['REP']), config=d.name,
    mode=os.environ['MODE'], monitoring=int(os.environ['MODE'] == 'memory'),
    memory_sampler=int(os.environ['MODE'] == 'memory'), phydll_logging=0, internal_memory=0,
    binding='pinned' if d.name in ('AIx', 'PHYDLL_CPP') else 'unbound',
    policy='final_total_per_queue_except_default_and_TPQ96', phydll_io_log=0,
    phydll_io_log_barriers=0, telemetry=0, internal_mem_verbose=0, internal_mem_collective=0,
    timing_sync=1, solver_ranks=96, dl_ranks=96 if d.name == 'PHYDLL_CPP' else 0,
    rank_grid='12x8', tpq=tpq, intra=intra, inter=0 if d.name == 'AIx' else 1,
    thread_settings='unforced' if d.name == 'AIx' else 'omitted_nominal' if omit == '1' else 'explicit',
    omit_threads=omit, omit_set_cpus=omit,
    total_bs=8294400 if os.environ['MODEL'] == 'watercnn' else 46080,
    db_batch=batch, db_min_batch=0, db_timeout=0, steps=int(os.environ['STEPS']),
    steady_steps=len(range(4, int(os.environ['STEPS']) + 1, 2)), wall_s=wall,
    status=status, log=str(d / 'solver.log'), cubex=str(profiles[0]) if len(profiles) == 1 else '',
    provenance=str(root / 'provenance.json'), run_environment=str(run_environment),
    environment_sha256=str(d / 'environment.sha256'),
    run_environment_sha256=hashlib.sha256(run_environment.read_bytes()).hexdigest() if run_environment.is_file() else '',
    mean_step_ms='', step4_ms='', rank_min_ms='', rank_max_ms='')
if os.environ.get('TERRAIN_RETEST_BUILD'):
    row['solver_readiness_wait'] = os.environ['TERRAIN_READINESS']
    row['retest_build'] = os.environ['TERRAIN_RETEST_BUILD']
if os.environ['MODE'] == 'timing':
    row['solver_readiness_wait'] = os.environ.get('TERRAIN_READINESS', 'false') if d.name == 'PHYDLL_CPP' else ''
    row['plain_build'] = os.environ['TERRAIN_PLAIN_BUILD']
    row['scorep'] = 0
if status == 'SUCCESS':
    timings(d / 'solver.log', row['steps'])
    if row['mode'] in ('performance', 'timing'):
        values = []
        for line in (d / 'solver.log').read_text().splitlines():
            if line.startswith('STEP_TIMING '):
                fields = dict(v.split('=', 1) for v in line.split()[1:] if '=' in v)
                if fields.get('solver') == 'ML' and int(fields['step']) >= 4:
                    values.append(Decimal(fields['step_ms']))
                    if fields['step'] == '4':
                        row.update(step4_ms=fields['step_ms'], rank_min_ms=fields['rank_min_ms'], rank_max_ms=fields['rank_max_ms'])
        row['mean_step_ms'] = str(sum(values) / len(values))
path = Path(os.environ['RUN_ROOT']) / 'results.csv'
new = not path.exists()
with path.open('a', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(row))
    if new: writer.writeheader()
    writer.writerow(row)
PY
    active=0
}
trap 'rc=$?; trap - EXIT; cleanup || rc=1; if (( active )); then status=FAILED_exit_$rc; wall=$((SECONDS-t0)); record || rc=1; fi; exit "$rc"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
common=(--device CPU --gpus-per-node 0 --ml-nodes 1 --model-backend TORCH
    --input-hdf5 "$MINI_APP_DIR/$prep" --steps "$STEPS" --save-every "$STEPS" --save-mode periodic
    --triangular-scale 1 --chunk-size 12 --io-mode parallel_hdf5 --mpi-sync-mode none
    --hdf5-xfer-mode independent --rank-grid-x 12 --rank-grid-z 8 --ml-interface cpp --ml-batch-size 10000000)
exports=(-x LD_LIBRARY_PATH -x SCOREP_EXPERIMENT_DIRECTORY -x SCOREP_ENABLE_PROFILING
    -x SCOREP_ENABLE_TRACING -x SCOREP_METRIC_PAPI -x SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY
    -x MLCOUPLING_MEM_LOG_VERBOSE -x MLCOUPLING_MEM_LOG_COLLECTIVE -x MLCOUPLING_STEP_TIMING_SYNC
    -x PHYDLL_IO_LOG -x PHYDLL_IO_LOG_BARRIERS -x PHYDLL_TELEMETRY -x SMARTSIM_FLAG_TELEMETRY)
if [[ $MODE == timing ]]; then
    exports=(-x LD_LIBRARY_PATH -x MLCOUPLING_MEM_LOG_VERBOSE -x MLCOUPLING_MEM_LOG_COLLECTIVE
        -x MLCOUPLING_STEP_TIMING_SYNC -x PHYDLL_IO_LOG -x PHYDLL_IO_LOG_BARRIERS
        -x PHYDLL_TELEMETRY -x SMARTSIM_FLAG_TELEMETRY)
fi
for config in $CONFIGS; do
    unset MLCOUPLING_INTRA_OP_THREADS MLCOUPLING_INTER_OP_THREADS PHYDLL_DL_COUNT PHYDLL_DL_FIELD_COUNT SSDB TERRAIN_MEMORY_TOKEN
    dir=$RUN_ROOT/$config
    mkdir "$dir"
    tpq=0; intra=0; db_batch=0; omit=0; expected_dl=0; expected_db=0; port=0
    model_path=$MINI_APP_DIR/train_models/model_a/${MODEL}_cpu.pt
    layout=flat_contiguous; cfg=config_aix.toml
    extra=()
    case $config in
        PHYDLL_CPP)
            cfg=config_phydll.toml; intra=1; expected_dl=96
            [[ -z ${TERRAIN_RETEST_BUILD:-} ]] || cfg=$TERRAIN_PHYDLL_CONFIG
            [[ $MODE != timing ]] || cfg=$TERRAIN_PHYDLL_CONFIG
            export MLCOUPLING_INTRA_OP_THREADS=1 MLCOUPLING_INTER_OP_THREADS=1 PHYDLL_DL_COUNT=1 PHYDLL_DL_FIELD_COUNT=1
            extra+=(-x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT) ;;
        SS_*)
            cfg=config.toml; layout=split_3x3; expected_db=1
            model_path=$MINI_APP_DIR/train_models/model_a/${MODEL}_split_cpu.pt
            case $config in
                SS_DEFAULT) omit=1; tpq=1; intra=1 ;;
                SS_TPQ1_I96_B96) tpq=1; intra=96; db_batch=$total_bs ;;
                SS_TPQ12_I8_B96) tpq=12; intra=8; db_batch=$((total_bs / 12)) ;;
                SS_TPQ96_I1_B96) tpq=96; intra=1 ;;
            esac
            export SMARTSIM_OMIT_THREADS=$omit SMARTSIM_OMIT_SET_CPUS=$omit
            export SMARTSIM_THREADS_PER_QUEUE=$tpq SMARTSIM_INTRA_OP_THREADS=$intra SMARTSIM_INTER_OP_THREADS=1
            export RUN_ID_ENV=terrain_cpu_final_${SLURM_JOB_ID}_${task}_${config}
            done_file=$dir/close_driver_${RUN_ID_ENV}.txt
            port=$((7100 + SLURM_JOB_ID % 1000)) ;;
    esac
    export SCOREP_EXPERIMENT_DIRECTORY=$dir/scorep
    [[ $MODE != timing ]] || unset SCOREP_EXPERIMENT_DIRECTORY
    active=1; t0=$SECONDS
    if [[ $MODE == memory ]]; then
        export TERRAIN_MEMORY_TOKEN=terrain_cpu_final_${SLURM_JOB_ID}_${task}_${config}
        extra+=(-x TERRAIN_MEMORY_TOKEN)
        "$HELPER_PYTHON" "$MINI_APP_DIR/terrain_memory_sampler.py" --output "$dir" \
            --token "$TERRAIN_MEMORY_TOKEN" --job-id "$SLURM_JOB_ID" --solver "$SOLVER_BIN" \
            --dl "$PHYDLL_DL_CLIENT" --db-port "$port" --interval "$MEMORY_INTERVAL" \
            --expected-solver 96 --expected-dl "$expected_dl" --expected-db "$expected_db" --cleanup \
            < /dev/null > "$dir/memory_monitor.log" 2>&1 &
        monitor_pid=$!
        for ((j=0; j<100; j++)); do
            [[ ! -f $dir/memory.ready ]] || break
            kill -0 "$monitor_pid" 2>/dev/null || exit 1
            sleep 0.1
        done
        [[ -f $dir/memory.ready ]] || exit 1
    fi
    if [[ $config == SS_* ]]; then
        baseline_steps=()
        before=$(current_steps)
        while read -r step; do [[ -z $step ]] || baseline_steps[$step]=1; done <<< "$before"
        controller=(timeout --kill-after=30 "$((RUN_TIMEOUT + DB_START_TIMEOUT + 120))" "$SMARTSIM_PYTHON"
            "$MINI_APP_DIR/smartsim_controller.py" --db_nodes 1 --db_layout shared --port "$port"
            --cpu_cores_per_node 96 --hostname_file "$dir/db_host.txt")
        printf '%q ' "${controller[@]}" > "$dir/controller_command.sh"
        (
            cd "$dir"
            # RedisAI must resolve its own Torch backend, not the solver's libtorch.
            if [[ $MODE == timing ]]; then
                export LD_LIBRARY_PATH=$TERRAIN_DB_LIBRARY_PATH
                env | sort > controller.env
                sha256sum controller.env > controller_environment.sha256
            fi
            read -r stat < /proc/self/stat
            read -ra fields <<< "${stat##*) }"
            printf '%s\n' "${fields[19]}" > controller_start
            exec "${controller[@]}" < /dev/null
        ) > "$dir/controller.log" 2>&1 &
        ctrl_pid=$!
        for ((j=0; j<DB_START_TIMEOUT; j++)); do
            [[ ! -s $dir/db_host.txt ]] || break
            kill -0 "$ctrl_pid" 2>/dev/null || exit 1
            sleep 1
        done
        [[ -s $dir/db_host.txt ]] || exit 1
        export SSDB=$(<"$dir/db_host.txt")
        extra+=(-x SSDB)
    fi
    solver_cmd=("$SOLVER_BIN")
    dl_cmd=("$PHYDLL_DL_CLIENT")
    if [[ $config == AIx || $config == PHYDLL_CPP ]]; then
        solver_cmd=("$MINI_APP_DIR/pin_wrapper.sh" "$SOLVER_BIN")
        dl_cmd=("$MINI_APP_DIR/pin_wrapper.sh" "$PHYDLL_DL_CLIENT")
    fi
    config_path=$cfg
    [[ $config_path == /* ]] || config_path=$MINI_APP_DIR/$config_path
    command=(timeout --kill-after=30 "$RUN_TIMEOUT" mpirun --bind-to none --report-bindings --oversubscribe
        "${exports[@]}" "${extra[@]}" -n 96 "${solver_cmd[@]}" "${common[@]}"
        --model-path "$model_path" --model-io-layout "$layout" --cpp-ml-config "$config_path" --output-hdf5 "$dir/traj.h5")
    if [[ $config == SS_* ]]; then
        command+=(--ml-db-batch-size "$db_batch" --ml-db-min-batch-size 0 --ml-db-min-batch-timeout 0)
    elif [[ $config == PHYDLL_CPP ]]; then
        command+=(: "${exports[@]}" "${extra[@]}" -n 96 "${dl_cmd[@]}")
    fi
    env | sort > "$dir/run.env"
    sha256sum "$dir/run.env" > "$dir/environment.sha256"
    printf '%q ' "${command[@]}" > "$dir/command.sh"
    t0=$SECONDS; rc=0
    (cd "$dir"; exec "${command[@]}" < /dev/null) > "$dir/solver.log" 2>&1 || rc=$?
    wall=$((SECONDS-t0))
    cleanup || rc=1
    if (( rc == 0 )); then
        "$HELPER_PYTHON" "$MINI_APP_DIR/terrain_cpu_diagnostics.py" --validate-log "$dir/solver.log" --steps "$STEPS" < /dev/null || rc=1
        if [[ $MODE == memory ]]; then
            "$HELPER_PYTHON" "$MINI_APP_DIR/terrain_cpu_diagnostics.py" --validate-memory "$dir/memory_summary.json" \
                --memory-config "$config" --solver-ranks 96 < /dev/null || rc=1
        fi
    fi
    shopt -s nullglob
    profiles=("$dir"/scorep/*.cubex)
    if [[ $MODE == timing ]]; then
        [[ ! -e $dir/scorep && ! -e $dir/memory_samples.csv && ! -e $dir/memory_summary.json ]] || rc=1
        sha256sum --check --quiet "$TERRAIN_PLAIN_PROVENANCE/binaries.sha256" > "$dir/build_binary_check.txt" || rc=1
    else
        (( ${#profiles[@]} == 1 )) || rc=1
    fi
    if [[ -n ${TERRAIN_RETEST_BUILD:-} && $rc == 0 ]]; then
        "$HELPER_PYTHON" "$MINI_APP_DIR/validate_terrain_phydll_profile.py" "${profiles[0]}" "$STEPS" \
            > "$dir/profile_validation.json" || rc=1
        sha256sum --check "$TERRAIN_RETEST_BUILD/provenance/binaries.sha256" > "$dir/build_binary_check.txt" || rc=1
    fi
    status=SUCCESS; (( rc == 0 )) || status=FAILED_rc_$rc
    record
    printf '%s %s wall=%ss\n' "$config" "$status" "$wall"
    (( rc == 0 )) || exit "$rc"
done
