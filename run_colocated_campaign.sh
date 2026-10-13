#!/usr/bin/env bash
set -euo pipefail
ROOT="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
phase=${1:?phase required}; platform=${2:?platform required}; rep=${3:?replicate required}
case "$phase" in smoke) steps=6; warmup=0;; pilot) steps=20; warmup=3;; production) steps=66; warmup=3;; *) exit 2;; esac
case "$platform" in
    cpu) width=384; height=216; gpus=0; tpq=6; intra=16;;
    gpu) width=1920; height=1080; gpus=4; tpq=1; intra=8;;
    *) exit 2;;
esac
[[ "$rep" =~ ^[1-5]$ ]] || exit 2
if [[ -n "${CAMPAIGN_BUILD_JOB:-}" ]]; then
    [[ "$CAMPAIGN_BUILD_JOB" =~ ^[0-9]+$ ]] || exit 2
    source "$ROOT/logs/build_colocated_campaign_${CAMPAIGN_BUILD_JOB}/build-dirs.env"
fi
: "${CAMPAIGN_ID:?}" "${CAMPAIGN_DIRECT_BUILD:?}" "${CAMPAIGN_CMI_BUILD:?}" "${CAMPAIGN_SCOREP_BUILD:?}"
for build in "$CAMPAIGN_DIRECT_BUILD" "$CAMPAIGN_CMI_BUILD" "$CAMPAIGN_SCOREP_BUILD"; do
    [[ "$build" =~ ^[A-Za-z0-9_-]+$ && -f "$ROOT/solver_cpp/$build/BUILD_READY" ]] || exit 1
done
[[ "${SLURM_JOB_NUM_NODES:?}" == 1 && "${SLURM_NTASKS:?}" == 96 ]] || { printf 'Expected one node and 96 tasks\n' >&2; exit 1; }
RUN_ROOT="$ROOT/logs/colocated_campaign/${CAMPAIGN_ID}/${platform}_${phase}_r${rep}_${SLURM_JOB_ID}"
mkdir -p "$(dirname "$RUN_ROOT")"
mkdir "$RUN_ROOT"
export SKIP_COMPILE_ENV=1 SKIP_RENDERING_ENV=1 NIC_SAMPLING_ENV=0
export SMARTSIM_RUNTIME_ROOT_ENV="$ROOT/../python"
export EXTERNAL_DIR_ENV="$RUN_ROOT/trajectories"
export TOTAL_STEPS_ENV=$steps SAVE_EVERY_ENV=$steps MPI_RANKS_ENV=96 RANK_GRID_X_ENV=12 RANK_GRID_Z_ENV=8
export TARGET_WIDTH_ENV=$width TARGET_HEIGHT_ENV=$height CHUNK_SIZE_ENV=12 ML_BATCH_SIZE_ENV=50000
export DB_NODES_ENV=1 DB_LAYOUT_ENV=shared ML_INFERENCE_CPU_CORES_ENV=96 SMARTSIM_DEDICATED_DB_NODES_ENV=0
export GPUS_PER_NODE_ENV=$gpus SMARTSIM_BALANCED_KEYS=1 SMARTSIM_THREADS_PER_QUEUE_ENV=$tpq SMARTSIM_THREADS_PER_QUEUE=$tpq
export SMARTSIM_INTRA_OP_THREADS=$intra SMARTSIM_INTER_OP_THREADS=1
export MODEL_NAME_ENV=benchmark_giant_mlp MODEL_BACKEND_ENV=TORCH MODEL_IO_LAYOUT_ENV=split_3x3
export MODEL_ARTIFACT_MANIFEST_ENV="$ROOT/train_models/model_a/artifact_manifest_giant_split.json"
export MODEL_PATH_ENV="$ROOT/train_models/model_a/giant_split_cuda.pt"
export CPP_ML_CONFIG_ENV="$ROOT/config_colocated_campaign.toml" CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM
export FORCE_FRESH_RUN_ENV=1 OVERWRITE_OUTPUT_ENV=1 USE_LOCAL_RUNTIME_STAGE_ENV=0 USE_LOCAL_MODEL_CACHE_ENV=1
export SKIP_PAPI_METRICS=1 SCOREP_METRIC_PAPI= SCOREP_METRIC_PAPI_PER_PROCESS= CUDA_LAUNCH_BLOCKING=1
export APPLY_CUDA_STUBS_ENV=1
export BASH_ENV="$ROOT/colocated_campaign_env.sh"
unset LD_PRELOAD SMARTSIM_DB_NODELIST SMARTSIM_DB_EXCLUDE_NODE
bash -n "$ROOT/single_node_96c_4gpu.sh"
conditions=(direct_nopreload direct_preload cmi_baseline cmi_profiling cmi_tracing)
printf 'condition\trun_id\tlog\tstatus\twarmup\tsteps\n' > "$RUN_ROOT/runs.tsv"
for ((position=0; position<5; position++)); do
    condition=${conditions[$(((rep - 1 + position) % 5))]}
    export RUN_ID_ENV="${CAMPAIGN_ID}_${platform}_${phase}_r${rep}_${SLURM_JOB_ID}_${condition}"
    export JOB_NAME_ENV="$RUN_ID_ENV" SCOREP_DIR_TAG_ENV="colocated_${condition}"
    export CAMPAIGN_ENV_DIR="$RUN_ROOT/${condition}_env"
    mkdir "$CAMPAIGN_ENV_DIR"
    export USE_SCOREP_ENV=0 USE_SCOREP=0 ENABLE_SCOREP_USER=0
    export SCOREP_ENABLE_PROFILING_ENV=false SCOREP_ENABLE_TRACING_ENV=false SCOREP_ENABLE_PROFILING=false SCOREP_ENABLE_TRACING=false
    unset SCOREP_EXPERIMENT_DIRECTORY
    export FORCE_TERRAIN_UPLOAD_EACH_STEP_ENV=1
    case "$condition" in
        direct_*) export ML_INTERFACE_ENV=smartsim USE_SMARTSIM=1 USE_CPP_ML_INTERFACE=0 COMPILE_OUTPUT_PATH_ENV="$CAMPAIGN_DIRECT_BUILD";;
        cmi_*) export ML_INTERFACE_ENV=cpp USE_SMARTSIM=0 USE_CPP_ML_INTERFACE=1 COMPILE_OUTPUT_PATH_ENV="$CAMPAIGN_CMI_BUILD";;
    esac
    [[ "$condition" != direct_preload ]] || export FORCE_TERRAIN_UPLOAD_EACH_STEP_ENV=0
    if [[ "$condition" == cmi_profiling || "$condition" == cmi_tracing ]]; then
        export USE_SCOREP_ENV=1 USE_SCOREP=1 ENABLE_SCOREP_USER=1 COMPILE_OUTPUT_PATH_ENV="$CAMPAIGN_SCOREP_BUILD" SCOREP_MPP_ENV=mpi
        if [[ "$condition" == cmi_profiling ]]; then
            export SCOREP_ENABLE_PROFILING_ENV=true SCOREP_ENABLE_PROFILING=true
        else
            export SCOREP_ENABLE_TRACING_ENV=true SCOREP_ENABLE_TRACING=true
        fi
    fi
    binary="$ROOT/solver_cpp/$COMPILE_OUTPUT_PATH_ENV/terrain_solver"
    [[ -x "$binary" ]] || { printf 'Missing binary: %s\n' "$binary" >&2; exit 1; }
    {
        printf 'phase=%s platform=%s rep=%s position=%s condition=%s warmup_ml_calls=%s\n' "$phase" "$platform" "$rep" "$position" "$condition" "$warmup"
        git -C "$ROOT" rev-parse HEAD
        git -C "$ROOT" diff -- solver_cpp
        git -C "$ROOT/../CPP-ML-Interface" rev-parse HEAD
        git -C "$ROOT/../CPP-ML-Interface" diff
        sha256sum "$binary" "$ROOT/solver_cpp/$COMPILE_OUTPUT_PATH_ENV/CMakeCache.txt" "$MODEL_PATH_ENV" "$MODEL_ARTIFACT_MANIFEST_ENV" "$CPP_ML_CONFIG_ENV" "$ROOT/single_node_96c_4gpu.sh"
        env | LC_ALL=C sort | while IFS= read -r line; do
            case "$line" in *TOKEN=*|*PASSWORD=*|*SECRET=*|*KEY=*) continue;; esac
            case "$line" in *ENV=*|SMARTSIM_*=*|SCOREP_*=*|SKIP_PAPI_METRICS=*|CUDA_LAUNCH_BLOCKING=*) printf '%s\n' "$line";; esac
        done
    } > "$RUN_ROOT/${condition}_identity.txt"
    # Validate Slurm export before the child; the BASH_ENV guard validates actual ranks.
    srun --export=ALL --nodes=1 --ntasks=1 --cpus-per-task=1 bash -c \
        '[[ "$SKIP_PAPI_METRICS" == 1 && -z "$SCOREP_METRIC_PAPI" && "$CUDA_LAUNCH_BLOCKING" == 1 ]]' \
        > "$RUN_ROOT/${condition}_export_check.txt" 2>&1
    log="$RUN_ROOT/${condition}.log"
    status=0
    bash "$ROOT/single_node_96c_4gpu.sh" > "$log" 2>&1 || status=$?
    rank_files=("$CAMPAIGN_ENV_DIR"/rank_*.txt)
    [[ ${#rank_files[@]} == 96 ]] || status=1
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$condition" "$RUN_ID_ENV" "$log" "$status" "$warmup" "$steps" >> "$RUN_ROOT/runs.tsv"
    # A failed child may leave its DB alive. Never run another condition then.
    [[ "$status" == 0 ]] || { printf 'Failed condition %s; see %s\n' "$condition" "$log" >&2; exit "$status"; }
done
