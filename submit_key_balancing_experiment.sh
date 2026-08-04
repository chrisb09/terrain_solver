#!/usr/bin/env zsh

# Submit direct SmartSim and CMI SmartSim baseline/balanced pairs. Build the
# matching normal binaries first with ./build_key_balancing.sh.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
JOB_SCRIPT="${SCRIPT_DIR}/proper_slurm_job.sh"
DB_NODES="${DB_NODES_ENV:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE_ENV:-4}"
MPI_RANKS="${MPI_RANKS_ENV:-96}"
TIME_LIMIT="${TIME_LIMIT_ENV:-00:15:00}"

if (( DB_NODES < 1 || GPUS_PER_NODE < 1 || MPI_RANKS < 1 )); then
  print -u2 "DB_NODES_ENV, GPUS_PER_NODE_ENV, and MPI_RANKS_ENV must be positive."
  exit 2
fi

typeset -a interfaces
interfaces=("$@")
if (( ${#interfaces} == 0 )); then
  interfaces=(direct cpp)
fi

for interface in "${interfaces[@]}"; do
  case "${interface}" in
    direct)
      build_dir=build_key_balance_direct
      interface_exports="USE_SMARTSIM=1,ML_INTERFACE_ENV=smartsim"
      ;;
    cpp)
      build_dir=build_key_balance_cpp
      interface_exports="USE_SMARTSIM=0,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM"
      ;;
    *)
      print -u2 "Usage: $0 [direct] [cpp]"
      exit 2
      ;;
  esac

  if [[ ! -x "${SCRIPT_DIR}/solver_cpp/${build_dir}/terrain_solver" ]]; then
    print -u2 "Missing normal build: solver_cpp/${build_dir}/terrain_solver"
    print -u2 "Run ./build_key_balancing.sh ${interface} first."
    exit 1
  fi

  for balanced in 0 1; do
    if (( balanced == 1 )); then
      variant=balanced
    else
      variant=baseline
    fi
    job_name="keybal_${interface}_${variant}"
    exports="ALL,USE_SCOREP_ENV=0,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${build_dir},USE_LOCAL_MODEL_CACHE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,TOTAL_STEPS_ENV=10,SAVE_EVERY_ENV=10,MPI_RANKS_ENV=${MPI_RANKS},DB_NODES_ENV=${DB_NODES},SMARTSIM_BALANCED_KEYS=${balanced},JOB_NAME_ENV=${job_name},${interface_exports}"

    print "Submitting ${job_name}: ${MPI_RANKS} solver ranks, ${DB_NODES} DB nodes, ${GPUS_PER_NODE} GPUs/node"
    sbatch \
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
      "${JOB_SCRIPT}"
  done
done
