#!/usr/bin/env zsh

# Maximum observed actual imbalance for one request per rank:
# - direct SmartSim water_tile_<rank>_0: 100% GPU bottleneck overhead
# - CMI SmartSim input_<rank>_0: 150% GPU bottleneck overhead
# Both use six 4-GPU inference nodes and 48 solver ranks. Jobs are chained to
# avoid concurrent Redis instances contending for port 6780 on shared nodes.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
JOB_SCRIPT="${SCRIPT_DIR}/proper_slurm_job.sh"
DB_NODES=6
GPUS_PER_NODE=4
MPI_RANKS=48
TIME_LIMIT="${TIME_LIMIT_ENV:-00:15:00}"
RANK_GRID_X=8
RANK_GRID_Z=6

typeset -A build_dirs
build_dirs=(direct build_key_balance_direct cpp build_key_balance_cpp)
previous_job_id=""

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
  if [[ ! -x "${SCRIPT_DIR}/solver_cpp/${build_dir}/terrain_solver" ]]; then
    print -u2 "Missing normal build: solver_cpp/${build_dir}/terrain_solver"
    print -u2 "Run ./build_key_balancing.sh ${interface} first."
    exit 1
  fi

  for variant in balanced_control natural; do
    if [[ "${variant}" == "balanced_control" ]]; then
      balanced=1
    else
      balanced=0
    fi
    job_name="keymax_${interface}_${variant}"
    exports="ALL,USE_SCOREP_ENV=0,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${build_dir},USE_LOCAL_MODEL_CACHE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,TOTAL_STEPS_ENV=10,SAVE_EVERY_ENV=10,MPI_RANKS_ENV=${MPI_RANKS},DB_NODES_ENV=${DB_NODES},RANK_GRID_X_ENV=${RANK_GRID_X},RANK_GRID_Z_ENV=${RANK_GRID_Z},TARGET_WIDTH_ENV=192,TARGET_HEIGHT_ENV=108,CHUNK_SIZE_ENV=12,ML_BATCH_SIZE_ENV=50000,MODEL_NAME_ENV=benchmark_giant_mlp,MODEL_BACKEND_ENV=TORCH,SMARTSIM_BALANCED_KEYS=${balanced},JOB_NAME_ENV=${job_name},${interface_exports}"
    dependency_args=()
    if [[ -n "${previous_job_id}" ]]; then
      dependency_args=("--dependency=afterany:${previous_job_id}")
    fi

    print "Submitting ${job_name}: ${MPI_RANKS} ranks, ${DB_NODES} nodes, ${GPUS_PER_NODE} GPUs/node"
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
    print "Submitted batch job ${previous_job_id}"
  done
done
