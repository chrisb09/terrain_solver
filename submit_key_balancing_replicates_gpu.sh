#!/usr/bin/env zsh

# 10-replicate GPU benchmark with 1920x1080 resolution:
# - direct SmartSim water_tile_<rank>_0
# - CMI SmartSim input_<rank>_0
# Both use six 4-GPU inference nodes and 48 solver ranks (8x6 grid).
# 10 steps per run (steps 4, 6, 8, 10 = 4 steady-state ML steps per run).
# All jobs are chained with dependencies to prevent concurrent Redis port collisions.

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
JOB_SCRIPT="${SCRIPT_DIR}/proper_slurm_job.sh"
MANIFEST_FILE="${SCRIPT_DIR}/gpu_replicates_jobs.txt"
DB_NODES=6
GPUS_PER_NODE=4
MPI_RANKS=48
TIME_LIMIT="${TIME_LIMIT_ENV:-00:30:00}"
RANK_GRID_X=8
RANK_GRID_Z=6
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

print "Submitting GPU 10-replicate benchmark chain (40 jobs in total)..."
print "Grid: 1920x1080 | Ranks: 48 (8x6) | DB Nodes: 6 (4 GPUs/node) | Wall time: ${TIME_LIMIT}"

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
      job_name="gpu_rep${rep}_${interface}_${variant}"
      exports="ALL,USE_SCOREP_ENV=0,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${build_dir},USE_LOCAL_MODEL_CACHE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,TOTAL_STEPS_ENV=10,SAVE_EVERY_ENV=10,MPI_RANKS_ENV=${MPI_RANKS},DB_NODES_ENV=${DB_NODES},RANK_GRID_X_ENV=${RANK_GRID_X},RANK_GRID_Z_ENV=${RANK_GRID_Z},TARGET_WIDTH_ENV=1920,TARGET_HEIGHT_ENV=1080,CHUNK_SIZE_ENV=12,ML_BATCH_SIZE_ENV=50000,MODEL_NAME_ENV=benchmark_giant_mlp,MODEL_BACKEND_ENV=TORCH,SMARTSIM_BALANCED_KEYS=${balanced},JOB_NAME_ENV=${job_name},${interface_exports}"

      dependency_args=()
      if [[ -n "${previous_job_id}" ]]; then
        dependency_args=("--dependency=afterany:${previous_job_id}")
      fi

      previous_job_id=$(sbatch --parsable \
        "${dependency_args[@]}" \
        --export="${exports}" \
        --partition=c23mm \
        --nodes=1 \
        --ntasks-per-node="${MPI_RANKS}" \
        --cpus-per-task=1 \
        --mem-per-cpu=5G \
        --time="${TIME_LIMIT}" \
        : \
        --partition=c23g \
        --nodes="${DB_NODES}" \
        --ntasks-per-node=1 \
        --cpus-per-task=24 \
        --gres="gpu:${GPUS_PER_NODE}" \
        --mem-per-cpu=5G \
        --time="${TIME_LIMIT}" \
        "${JOB_SCRIPT}")
      print "Submitted rep ${rep} ${interface} ${variant}: Job ID ${previous_job_id}"
      print "${rep} ${interface} ${variant} ${previous_job_id}" >> "${MANIFEST_FILE}"
    done
  done
done

print "All 40 replicate jobs submitted successfully in dependency chain."
print "Job manifest saved to ${MANIFEST_FILE}."
