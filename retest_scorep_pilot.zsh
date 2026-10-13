#!/bin/zsh
# One authorized pilot per allocation. --list and --describe never write files.
set -euo pipefail
mini=${0:A:h}
cmi=${mini:h}/CPP-ML-Interface
fail() { print -u2 -- "ERROR: $*"; exit 1; }
cases=()
for cfg in 24c1g 96c4g 24p24c1g 96p96c4g; do
  cases+=("$cfg watercnn8k PHYDLL default" "$cfg watercnn8k PHYDLL readiness" "$cfg watercnn8k SMARTSIM shared")
done
cases+=("96c4g watercnn8k SMARTSIM pergpu" "96p96c4g watercnn8k SMARTSIM pergpu"
        "24c1g watercnn8k AIX default" "96p96c4g watercnn8k AIX default"
        "24p24c1g giantmlp288 PHYDLL default" "24p24c1g giantmlp288 PHYDLL readiness"
        "24p24c1g giantmlp288 SMARTSIM shared" "96p96c4g giantmlp288 SMARTSIM shared"
        "96c4g giantmlp288 SMARTSIM pergpu" "96p96c4g giantmlp288 SMARTSIM pergpu")
(( ${#cases} == 22 )) || fail 'Internal pilot count is not 22'
if [[ ${1:-} == --list ]]; then
  (( $# == 1 )) || fail 'Usage: --list'
  printf '%s\n' "${cases[@]}"
  exit 0
fi
describe=0
if [[ ${1:-} == --describe ]]; then describe=1; shift; fi
(( $# == 4 )) || fail 'Usage: [--describe] CONFIG MODEL PROVIDER VARIANT'
cfg=$1 model=$2 provider=$3 variant=$4
authorized=0
for entry in "${cases[@]}"; do
  [[ $entry == "$cfg $model $provider $variant" ]] && authorized=1
done
(( authorized )) || fail 'Unauthorized pilot'
pilot_mode=${PILOT_MODE_ENV-profile}
tpq=${PILOT_TPQ_ENV-1}
[[ $pilot_mode == profile || $pilot_mode == trace ]] || fail 'Invalid PILOT_MODE_ENV (profile|trace)'
[[ $tpq == 1 || $tpq == 2 ]] || fail 'Invalid PILOT_TPQ_ENV (1|2)'
steps=12; calls=6; steady=5; tracing=false
if [[ $pilot_mode != profile || $tpq != 1 ]]; then
  [[ $cfg == 96c4g || $cfg == 96p96c4g ]] && [[ $model == giantmlp288 && $provider == SMARTSIM && $variant == pergpu ]] || fail 'Mode/TPQ overrides require an authorized small SmartSim pilot'
  if [[ $pilot_mode == trace ]]; then
    [[ $tpq == 1 ]] || fail 'Trace pilots require TPQ1'
    steps=8; calls=4; steady=3; tracing=true
  fi
fi
export PILOT_MODE_ENV=$pilot_mode PILOT_TPQ_ENV=$tpq
case $cfg in
  24c1g|24p24c1g) ranks=24; gpus=1; gx=6; gz=4 ;;
  96c4g|96p96c4g) ranks=96; gpus=4; gx=12; gz=8 ;;
esac
driver=$mini/5r_driver_single.zsh
if [[ $cfg == *p*c*g ]]; then
  driver=$mini/5r_driver_hetsurrogate.zsh
  gx=0; gz=0
fi
case $model in
  watercnn8k) mn=watercnn; width=7680; height=4320; chunk=60; batch=20000000 ;;
  giantmlp288) mn=benchmark_giant_mlp; width=480; height=288; chunk=16; batch=500000 ;;
esac
export USE_SCOREP_ENV=1 SCOREP_MPP_ENV=mpi SKIP_COMPILE_ENV=1
export SCOREP_ENABLE_TRACING_ENV=$tracing SCOREP_ENABLE_PROFILING_ENV=true
export SCOREP_ENABLE_TRACING=$tracing SCOREP_ENABLE_PROFILING=true
export SCOREP_SHARED_PROFILE_ENV=1 GPU_MEM_SAMPLING_ENV=0
export SKIP_PAPI_METRICS=1 SCOREP_METRIC_PAPI=
export MLCOUPLING_MEM_LOG_VERBOSE=0 MLCOUPLING_MEM_LOG_COLLECTIVE=0 MLCOUPLING_STEP_TIMING_SYNC=1
export USE_SMARTSIM=0 USE_CPP_ML_INTERFACE=1 ML_INTERFACE_ENV=cpp
export CPP_ML_INTERFACE_PROVIDER_ENV=$provider
export COMPILE_OUTPUT_PATH_ENV=build_scorep_pilots
export AIX_INSTALL_PREFIX_ENV=$cmi/extern/AIxeleratorService/INSTALL-SCOREP
export MODEL_NAME_ENV=$mn TARGET_WIDTH_ENV=$width TARGET_HEIGHT_ENV=$height
export CHUNK_SIZE_ENV=$chunk ML_BATCH_SIZE_ENV=$batch TOTAL_STEPS_ENV=$steps SAVE_EVERY_ENV=$steps
export MPI_RANKS_ENV=$ranks RANK_GRID_X_ENV=$gx RANK_GRID_Z_ENV=$gz GPUS_PER_NODE_ENV=$gpus
if [[ $cfg == *p*c*g ]]; then
  # Empty overrides preserve the driver's per-node count and its automatic grid.
  unset RANK_GRID_X_ENV RANK_GRID_Z_ENV
fi
export HET_G_SIDE_TASKS_ENV=$ranks ML_INFERENCE_CPU_CORES_ENV=24
export FORCE_FRESH_RUN_ENV=1 OVERWRITE_OUTPUT_ENV=1 SKIP_RENDERING_ENV=1
export USE_LOCAL_RUNTIME_STAGE_ENV=0 USE_LOCAL_MODEL_CACHE_ENV=1 SOLVER_CPU_BIND_ENV=cores
export NIC_SAMPLING_ENV=0 OVERWRITE_JOB_NAME_ENV=1 AIX_P2P_TIMELINE_ENV=0
export DB_NODES_ENV=1 DB_LAYOUT_ENV=shared
export SMARTSIM_THREADS_PER_QUEUE_ENV=$tpq SMARTSIM_THREADS_PER_QUEUE=$tpq
export ML_DB_BATCH_SIZE_ENV=$batch SMARTSIM_MPI_SEQUENTIAL_PUT=0
export SMARTSIM_INTRA_OP_THREADS=8 SMARTSIM_INTER_OP_THREADS=1
export PHYDLL_PIN_ENV=1 PHYDLL_PIN_LOG_ENV=1 PHYDLL_DL_NODES_ENV=1
export PHYDLL_NP_DL_ENV=$gpus PHYDLL_NP_DL=$gpus PHYDLL_DL_COUNT=1 PHYDLL_DL_FIELD_COUNT=1
export PHYDLL_REBUILD_DL_CLIENT_ENV=0 PHYDLL_REBUILD_ENV=0 USE_PYTHON_DL_CLIENT=0
export PHYDLL_DL_BUILD_DIR=$cmi/dl_clients/build-scorep-pilots
export PHYDLL_DL_CLIENT=$PHYDLL_DL_BUILD_DIR/phydll_dl_client
unset SMARTSIM_DB_GPU_COUNT AIX_P2P_INITIAL_CREDITS_ENV
# Do not force AIx's asynchronous path into synchronous inference.
case $provider in
  PHYDLL)
    export MLCOUPLING_INTRA_OP_THREADS=1 MLCOUPLING_INTER_OP_THREADS=1
    export CPP_ML_CONFIG_ENV=config_phydll_pilot_${variant}.toml ;;
  AIX) export CPP_ML_CONFIG_ENV=config_aix.toml ;;
  SMARTSIM)
    export CPP_ML_CONFIG_ENV=config.toml
    if [[ $variant == pergpu ]]; then
      export DB_LAYOUT_ENV=per-ml-node DB_NODES_ENV=4 SMARTSIM_DB_GPU_COUNT=4
    fi ;;
esac
export SCOREP_DIR_TAG_ENV=scorep_pilot
run_tag=
if [[ $pilot_mode != profile || $tpq != 1 ]]; then
  run_tag=_${pilot_mode}_tpq${tpq}
  export SCOREP_DIR_TAG_ENV=scorep_pilot${run_tag}
fi
if (( describe )); then
  printf 'pilot=%s model=%s provider=%s variant=%s steps=%s ml_calls=%s warm=1 steady=%s dimensions=%sx%s chunk=%s cap=%s ranks_per_component=%s grid=%sx%s gpus=%s db_layout=%s db_nodes=%s config=%s driver=%s mode=%s tpq=%s tracing=%s tag=%s\n' \
    "$cfg" "$mn" "$provider" "$variant" "$steps" "$calls" "$steady" "$width" "$height" "$chunk" "$batch" "$ranks" "$gx" "$gz" "$gpus" "$DB_LAYOUT_ENV" "$DB_NODES_ENV" "$CPP_ML_CONFIG_ENV" "$driver" "$pilot_mode" "$tpq" "$tracing" "$SCOREP_DIR_TAG_ENV"
  exit 0
fi
[[ -n ${SLURM_JOB_ID:-} ]] || fail 'A Slurm allocation is required'
[[ ${PILOT_BUILD_JOB_ID_ENV:-} == <-> ]] || fail 'Invalid/missing build job ID'
[[ -n ${PILOT_CAMPAIGN_ENV:-} && $PILOT_CAMPAIGN_ENV == [A-Za-z0-9]* && $PILOT_CAMPAIGN_ENV != *[^A-Za-z0-9_.-]* ]] || fail 'Invalid/missing PILOT_CAMPAIGN_ENV'
export RUN_ID_ENV=scorep_pilot_${PILOT_CAMPAIGN_ENV}_${cfg}_${model}_${provider:l}_${variant}${run_tag}_${SLURM_JOB_ID}
export JOB_NAME_ENV=$RUN_ID_ENV
out=$mini/results/scorep_pilots/$PILOT_CAMPAIGN_ENV/$RUN_ID_ENV
[[ -d $mini/results && -d $mini/logs ]] || fail 'results and logs parent directories must exist'
mkdir -p "$mini/results/scorep_pilots/$PILOT_CAMPAIGN_ENV"
[[ ! -e $out ]] || fail "Pilot output already exists: $out"
mkdir "$out"
# Queued small-campaign jobs read this runner at start; no sbatch update needed.
unset SMARTSIM_DB_AFFINITY_CAPTURE SMARTSIM_DB_AFFINITY_OUTPUT
if [[ $PILOT_CAMPAIGN_ENV == smartsim_small_* && $provider == SMARTSIM && $model == giantmlp288 && $variant == pergpu && -n $run_tag ]]; then
  export SMARTSIM_DB_AFFINITY_CAPTURE=1 SMARTSIM_DB_AFFINITY_OUTPUT=$out/db_affinity.json
fi
# Always leave one result manifest, including failures before the driver starts.
rc=1
finish() {
  local profiles=( $mini/scorep_runs/${SCOREP_DIR_TAG_ENV}_${RUN_ID_ENV}*/**/*.cubex(N) )
  local traces=() events=() candidate
  for candidate in $mini/scorep_runs/${SCOREP_DIR_TAG_ENV}_${RUN_ID_ENV}*/**/*.otf2(N); do
    [[ -s $candidate ]] && traces+=("$candidate")
  done
  for candidate in $mini/scorep_runs/${SCOREP_DIR_TAG_ENV}_${RUN_ID_ENV}*/**/*.evt(N); do
    [[ -s $candidate ]] && events+=("$candidate")
  done
  : > "$out/profiles.txt"
  if (( ${#profiles} )); then printf '%s\n' "${profiles[@]}" > "$out/profiles.txt"; fi
  : > "$out/traces.txt"
  if (( ${#traces} )); then printf '%s\n' "${traces[@]}" > "$out/traces.txt"; fi
  : > "$out/trace_events.txt"
  if (( ${#events} )); then printf '%s\n' "${events[@]}" > "$out/trace_events.txt"; fi
  if [[ $pilot_mode == trace ]] && (( ! ${#traces} || ! ${#events} )); then
    print -u2 -- 'ERROR: Required nonempty archived OTF2 anchor/event data missing'
    (( rc != 0 )) || rc=1
  fi
  printf 'run_id,config,model,provider,variant,rc,profiles_file,mode,tpq,steps,ml_calls,warm,steady,traces_file,trace_events_file\n%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1,%s,%s,%s\n' \
    "$RUN_ID_ENV" "$cfg" "$model" "$provider" "$variant" "$rc" "$out/profiles.txt" "$pilot_mode" "$tpq" "$steps" "$calls" "$steady" "$out/traces.txt" "$out/trace_events.txt" > "$out/manifest.csv"
  print -- "PILOT_RESULT run_id=$RUN_ID_ENV rc=$rc manifest=$out/manifest.csv profiles=${#profiles} traces=${#traces}"
  trap - EXIT
  exit "$rc"
}
trap finish EXIT
solver=$mini/solver_cpp/build_scorep_pilots
dl=$PHYDLL_DL_BUILD_DIR
[[ -r $solver/BUILD_READY && -r $dl/BUILD_READY && -r $solver/build.env ]] || fail 'Pilot BUILD_READY markers/build.env missing'
cmp "$solver/BUILD_READY" "$dl/BUILD_READY" || fail 'Build markers differ'
grep -Eq "^BUILD_READY job=${PILOT_BUILD_JOB_ID_ENV:?} completed=" "$solver/BUILD_READY" || fail 'Build marker does not match dependency job'
source "$solver/build.env"
[[ $SOLVER_BIN == $solver/terrain_solver && $PHYDLL_DL_CLIENT == $dl/phydll_dl_client && $USE_SCOREP == 1 && $SCOREP_MPP == mpi && $PROVENANCE == $solver/provenance ]] || fail 'Unexpected pilot build environment'
[[ -x $SOLVER_BIN && -x $PHYDLL_DL_CLIENT ]] || fail 'Pilot executables missing'
for cache in $solver/CMakeCache.txt $dl/CMakeCache.txt; do
  grep -qx 'WITH_SCOREP:BOOL=ON' "$cache" || fail 'Score-P cache missing'
  grep -qx 'PHYDLL_DL_WITH_TORCH:BOOL=ON' "$cache" || fail 'Native Torch disabled'
  grep -Eq '^CPPML_SCOREP_MPP:.*=mpi$' "$cache" || fail 'MPI instrumentation missing'
done
for flag in USE_CPP_ML_INTERFACE WITH_PHYDLL WITH_SMARTSIM WITH_AIX; do
  grep -qx "$flag:BOOL=ON" "$solver/CMakeCache.txt" || fail "Solver cache missing $flag (would risk driver rebuild)"
done
grep -Fq solver_readiness_wait "$solver/generated_registry.hpp" || fail 'Readiness registry missing'
grep -Fq 'PhyDLL readiness sequence mismatch' "$PROVENANCE/solver-strings.txt" || fail 'Solver readiness implementation missing'
grep -Fq 'invalid solver_readiness_wait flag' "$PROVENANCE/dl-strings.txt" || fail 'DL readiness implementation missing'
cmp "$solver/generated_registry.hpp" "$solver/cpp-ml-interface-build/generated_registry.hpp" || fail 'Readiness registries differ'
sha256sum --check "$PROVENANCE/binaries.sha256" > "$out/binary_check.log" || fail 'Pilot binaries changed since build'
for name in terrain_solver phydll_dl_client; do
  grep -q libscorep_mpp_mpi "$PROVENANCE/$name.build-only.ldd.txt" || fail 'Native MPI Score-P linkage missing'
done
# Driver passes this TOML unchanged; terrain_solver overrides model/device/layout,
# AIx library.batchsize and SmartSim library.batch_size at create_from_config.
[[ -r $mini/$CPP_ML_CONFIG_ENV && $TOTAL_STEPS_ENV == $steps && $SAVE_EVERY_ENV == $steps && $SMARTSIM_THREADS_PER_QUEUE_ENV == $tpq && $SMARTSIM_THREADS_PER_QUEUE == $tpq && $SCOREP_ENABLE_TRACING_ENV == $tracing && $SCOREP_ENABLE_TRACING == $tracing ]] || fail 'Invalid runtime config/steps/mode/TPQ'
if [[ $provider == PHYDLL ]]; then
  expected=false; [[ $variant == readiness ]] && expected=true
  grep -qx "solver_readiness_wait = $expected" "$mini/$CPP_ML_CONFIG_ENV" || fail 'Readiness config mismatch'
fi
if [[ $cfg == *p*c*g ]]; then
  [[ ${SLURM_HET_SIZE:-0} == 2 && ${SLURM_NTASKS_HET_GROUP_0:-0} == $ranks && ${SLURM_NTASKS_HET_GROUP_1:-0} == $ranks && ${SLURM_JOB_NUM_NODES_HET_GROUP_0:-0} == 1 && ${SLURM_JOB_NUM_NODES_HET_GROUP_1:-0} == 1 && $gx == 0 && $gz == 0 ]] || fail 'Het allocation/grid mismatch (auto grid across both components)'
else
  [[ ${SLURM_NTASKS:-0} == $ranks && ${SLURM_JOB_NUM_NODES:-0} == 1 ]] && (( gx * gz == ranks )) || fail 'Single allocation/grid mismatch'
fi
cp "$mini/$CPP_ML_CONFIG_ENV" "$out/config.toml"
if [[ $provider == PHYDLL ]]; then
  # The solver does NOT override PhyDLL library.batch_size with --ml-batch-size.
  # Keep checked-in clones intact and apply the inference cap to this run only.
  sed "s/^batch_size = 500000$/batch_size = $batch/" "$mini/$CPP_ML_CONFIG_ENV" > "$out/config.toml"
  grep -qx "batch_size = $batch" "$out/config.toml" || fail 'Runtime PhyDLL inference cap mismatch'
fi
export CPP_ML_CONFIG_ENV=$out/config.toml
[[ $MODEL_NAME_ENV == $mn && $TARGET_WIDTH_ENV == $width && $TARGET_HEIGHT_ENV == $height && $CHUNK_SIZE_ENV == $chunk && $ML_BATCH_SIZE_ENV == $batch ]] || fail 'Runtime model/dimensions/cap mismatch'
env | LC_ALL=C sort > "$out/runtime.env"
cd "$mini"
rc=0
zsh "$driver" > "$out/driver.log" 2>&1 || rc=$?
sha256sum --check "$PROVENANCE/binaries.sha256" >> "$out/binary_check.log" || rc=1
exit "$rc"
