#!/usr/bin/env zsh

# 10-replicate CPU benchmark with 216x144 resolution:
# - direct SmartSim water_tile_<rank>_0
# - CMI SmartSim input_<rank>_0
# Both use six 24-core CPU inference nodes and 24 solver ranks (6x4 grid).
# 10 steps per run (steps 4, 6, 8, 10 = 4 steady-state ML steps per run).
# All jobs are chained with dependencies to prevent concurrent Redis port collisions.

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
RUNNER="${SCRIPT_DIR}/run_key_balancing_cpu_debug.sh"
MANIFEST_FILE="${SCRIPT_DIR}/cpu_replicates_jobs.txt"
DB_NODES=6
MPI_RANKS=24
ALLOC_CPUS_PER_NODE=24
DB_CPU_CORES=24
TIME_LIMIT="${TIME_LIMIT_ENV:-00:30:00}"
RANK_GRID_X=6
RANK_GRID_Z=4
REPLICATES=10

typeset -A build_dirs
build_dirs=(direct build_key_balance_direct cpp build_key_balance_cpp)
previous_job_id=""

for interface in direct cpp; do
  build_dir="${build_dirs[${interface}]}"
  if [[ ! -x "${SCRIPT_DIR}/solver_cpp/${build_dir}/terrain_solver" ]]; then
    print -u2 "Missing normal build: solver_cpp/${build_dir}/terrain_solver"
    print -u2 "Run ./build_key_balancing.sh ${interface} first."
    exit 1
  fi
done

rm -f "${MANIFEST_FILE}"

print "Submitting CPU 10-replicate benchmark chain (40 jobs in total)..."
print "Grid: 216x144 | Ranks: 24 (6x4) | DB Nodes: 6 (24 cores/node) | Wall time: ${TIME_LIMIT}"

for rep in $(seq 1 "${REPLICATES}"); do
  for interface in direct cpp; do
    case "${interface}" in
      direct)
        interface_exports="USE_SMARTSIM=1,ML_INTERFACE_ENV=smartsim"
        ;;
      cpp)
        interface_exports="USE_SMARTSIM=0,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM"
        ;;
    esac
    build_dir="${build_dirs[${interface}]}"

    # Alternate variant order per replicate to eliminate order bias
    if (( rep % 2 == 1 )); then
      variants=(balanced_control natural)
    else
      variants=(natural balanced_control)
    fi

    for variant in "${variants[@]}"; do
      if [[ "${variant}" == "balanced_control" ]]; then
        balanced=1
      else
        balanced=0
      fi
      job_name="cpurep${rep}_${interface}_${variant}"
      exports="ALL,USE_SCOREP_ENV=0,USE_SMARTSIM=1,ML_INTERFACE_ENV=smartsim,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${build_dir},USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,TOTAL_STEPS_ENV=10,SAVE_EVERY_ENV=10,MPI_RANKS_ENV=${MPI_RANKS},DB_NODES_ENV=${DB_NODES},ML_INFERENCE_CPU_CORES_ENV=${DB_CPU_CORES},SMARTSIM_DEDICATED_DB_NODES_ENV=1,SMARTSIM_PIN_DB_NODELIST=1,RANK_GRID_X_ENV=${RANK_GRID_X},RANK_GRID_Z_ENV=${RANK_GRID_Z},TARGET_WIDTH_ENV=216,TARGET_HEIGHT_ENV=144,CHUNK_SIZE_ENV=12,ML_BATCH_SIZE_ENV=50000,MODEL_NAME_ENV=benchmark_giant_mlp,MODEL_BACKEND_ENV=TORCH,SMARTSIM_BALANCED_KEYS=${balanced},SMARTSIM_INTRA_OP_THREADS=1,SMARTSIM_INTER_OP_THREADS=1,SMARTSIM_THREADS_PER_QUEUE=${DB_CPU_CORES},JOB_NAME_ENV=${job_name},${interface_exports}"

      dependency_args=()
      if [[ -n "${previous_job_id}" ]]; then
        dependency_args=("--dependency=afterany:${previous_job_id}")
      fi

      previous_job_id=$(sbatch --parsable \
        "${dependency_args[@]}" \
        --export="${exports}" \
        --partition=c23mm \
        --nodes=$((DB_NODES + 1)) \
        --ntasks-per-node="${ALLOC_CPUS_PER_NODE}" \
        --cpus-per-task=1 \
        --mem-per-cpu=5G \
        --time="${TIME_LIMIT}" \
        "${RUNNER}")
      print "Submitted rep ${rep} ${interface} ${variant}: Job ID ${previous_job_id}"
      print "${rep} ${interface} ${variant} ${previous_job_id}" >> "${MANIFEST_FILE}"
    done
  done
done

print "All 40 CPU replicate jobs submitted successfully in dependency chain."
print "Job manifest saved to ${MANIFEST_FILE}."
