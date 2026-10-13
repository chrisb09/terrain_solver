#!/usr/bin/env zsh

# 2-node Standalone CPU Score-P instrumentation-overhead benchmark.
# Measures the overhead of Score-P profiling (CUBE4) vs tracing (OTF2) on the
# CMI SmartSim path, to be compared against the uninstrumented CMI runs.
# - 1 c23ms solver node (48 ranks, 8x6) + 1 c23ms DB node (96 cores) = 2 nodes.
# - DB_NODES=1, shared layout -> SmartSim Standalone (no clustering/sharding).
# - Settings: TPQ=6, INTRA=16 (6x16 = 96 threads/node).
# - Workload identical to the other 2-node campaigns:
#   384x216, CHUNK_SIZE=12, ML_BATCH_SIZE=50000, benchmark_giant_mlp (TORCH).
# - Conditions: profiling (10) and tracing (10), submitted independently.
# - Uses the Score-P CMI build (build_scorep_cmi), which must be rebuilt first
#   via the devel build job; pass its id via DEPENDENCY_ENV=afterok:<buildjob>.
# - Account: rwth2150, Partition: c23ms.
# - Overrides: REPLICATES_ENV, TIME_LIMIT_ENV, DEPENDENCY_ENV.
#   SCOREP_TOTAL_MEMORY is intentionally NOT set here: the launcher sets it
#   (524288K = 512M per rank) when USE_SCOREP=1.

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
MINI_APP_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app"
RUNNER="${MINI_APP_DIR}/run_key_balancing_cpu_debug.sh"
OUT_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/interface_overhead"
MANIFEST_PROF="${OUT_DIR}/cpu_iface_profiling_jobs.txt"
MANIFEST_TRACE="${OUT_DIR}/cpu_iface_tracing_jobs.txt"

SLURM_ACCOUNT="rwth2150"
DB_NODES=1
MPI_RANKS=48
ALLOC_CPUS_PER_NODE=96
DB_CPU_CORES=96
TPQ=6
INTRA=16
TARGET_WIDTH=384
TARGET_HEIGHT=216
TIME_LIMIT="${TIME_LIMIT_ENV:-00:20:00}"
RANK_GRID_X=8
RANK_GRID_Z=6
REPLICATES="${REPLICATES_ENV:-10}"

if [[ ! -x "${MINI_APP_DIR}/solver_cpp/build_scorep_cmi/terrain_solver" ]]; then
  print -u2 "Missing build: ${MINI_APP_DIR}/solver_cpp/build_scorep_cmi/terrain_solver"
  print -u2 "Rebuild it via the devel build job first (slurm_build.sh build_scorep_cmi)."
  exit 1
fi

mkdir -p "${OUT_DIR}"
rm -f "${MANIFEST_PROF}" "${MANIFEST_TRACE}"

print "========================================================================"
print " Submitting CPU CMI Score-P instrumentation-overhead suite"
print " Account: ${SLURM_ACCOUNT} | Partition: c23ms (exclusive 2 nodes)"
print " Grid: ${TARGET_WIDTH}x${TARGET_HEIGHT} | Ranks: ${MPI_RANKS} (8x6)"
print " DB Nodes: ${DB_NODES} (96 cores, Standalone) | TPQ=${TPQ} | Intra=${INTRA}"
print " Build: build_scorep_cmi | Conditions: profiling, tracing"
print " Replicates: ${REPLICATES} each | Dependency: ${DEPENDENCY_ENV:-none}"
print "========================================================================"

common_exports="USE_SCOREP_ENV=1,SCOREP_MPP_ENV=mpi,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=build_scorep_cmi,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,TOTAL_STEPS_ENV=10,SAVE_EVERY_ENV=10,MPI_RANKS_ENV=${MPI_RANKS},DB_NODES_ENV=${DB_NODES},ML_INFERENCE_CPU_CORES_ENV=${DB_CPU_CORES},SMARTSIM_DEDICATED_DB_NODES_ENV=1,SMARTSIM_PIN_DB_NODELIST=1,RANK_GRID_X_ENV=${RANK_GRID_X},RANK_GRID_Z_ENV=${RANK_GRID_Z},TARGET_WIDTH_ENV=${TARGET_WIDTH},TARGET_HEIGHT_ENV=${TARGET_HEIGHT},CHUNK_SIZE_ENV=12,ML_BATCH_SIZE_ENV=50000,MODEL_NAME_ENV=benchmark_giant_mlp,MODEL_BACKEND_ENV=TORCH,SMARTSIM_BALANCED_KEYS=1,SMARTSIM_INTRA_OP_THREADS=${INTRA},SMARTSIM_INTER_OP_THREADS=1,SMARTSIM_THREADS_PER_QUEUE=${TPQ},USE_SMARTSIM=0,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config_keybalance.toml"

dependency_args=()
if [[ -n "${DEPENDENCY_ENV:-}" ]]; then
  dependency_args=("--dependency=${DEPENDENCY_ENV}")
fi
job_count=0

# Submit from mini_app so SLURM_SUBMIT_DIR points at the launcher directory.
cd "${MINI_APP_DIR}"

for rep in $(seq 1 "${REPLICATES}"); do
  for condition in profiling tracing; do
    if [[ "${condition}" == "profiling" ]]; then
      job_name="cpu_iface_prof_r${rep}"
      exports="ALL,${common_exports},SCOREP_ENABLE_PROFILING_ENV=true,SCOREP_ENABLE_TRACING_ENV=false,SCOREP_DIR_TAG_ENV=iface_cpp_prof,JOB_NAME_ENV=${job_name}"
      manifest="${MANIFEST_PROF}"
    else
      job_name="cpu_iface_trace_r${rep}"
      exports="ALL,${common_exports},SCOREP_ENABLE_PROFILING_ENV=false,SCOREP_ENABLE_TRACING_ENV=true,SCOREP_DIR_TAG_ENV=iface_cpp_trace,JOB_NAME_ENV=${job_name}"
      manifest="${MANIFEST_TRACE}"
    fi

    submit_raw=$(sbatch --parsable \
      "${dependency_args[@]}" \
      --account="${SLURM_ACCOUNT}" \
      --output="${MINI_APP_DIR}/output_%j.txt" \
      --export="${exports}" \
      --exclusive \
      --partition=c23ms \
      --nodes=$((DB_NODES + 1)) \
      --ntasks-per-node=1 \
      --cpus-per-task="${ALLOC_CPUS_PER_NODE}" \
      --mem=0 \
      --time="${TIME_LIMIT}" \
      "${RUNNER}")

    job_id="${submit_raw%%;*}"
    if [[ "${job_id}" != <-> ]]; then
      print -u2 "Failed to submit job: ${submit_raw}"
      exit 1
    fi

    job_count=$((job_count + 1))
    print "[${job_count}/$((REPLICATES * 2))] Submitted rep ${rep} ${condition}: Job ID ${job_id}"
    print "${rep} cpp ${condition} ${job_id}" >> "${manifest}"
  done
done

print "========================================================================"
print " All ${job_count} CPU instrumentation jobs submitted."
print " Manifests: ${MANIFEST_PROF}"
print "            ${MANIFEST_TRACE}"
print "========================================================================"
