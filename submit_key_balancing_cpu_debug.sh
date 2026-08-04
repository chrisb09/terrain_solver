#!/usr/bin/env zsh

# Fast CPU-only validation for Redis hash-tag routing. It submits direct
# SmartSim baseline and balanced runs with one 16-rank solver node and four
# 24-core CPU database nodes. A single five-node allocation is used because
# this Slurm setup rejects two heterogeneous components from c23mm.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
RUNNER="${SCRIPT_DIR}/run_key_balancing_cpu_debug.sh"
BUILD_DIR=build_key_balance_direct
TIME_LIMIT="${TIME_LIMIT_ENV:-00:30:00}"

if [[ ! -x "${SCRIPT_DIR}/solver_cpp/${BUILD_DIR}/terrain_solver" ]]; then
  print -u2 "Missing normal direct build: solver_cpp/${BUILD_DIR}/terrain_solver"
  print -u2 "Run ./build_key_balancing.sh direct first."
  exit 1
fi

previous_job_id=""
for balanced in 0 1; do
  if (( balanced == 1 )); then
    variant=balanced
  else
    variant=baseline
  fi
  job_name="keybal_cpu_${variant}"
  exports="ALL,USE_SCOREP_ENV=0,USE_SMARTSIM=1,ML_INTERFACE_ENV=smartsim,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${BUILD_DIR},USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,TOTAL_STEPS_ENV=10,SAVE_EVERY_ENV=10,MPI_RANKS_ENV=16,DB_NODES_ENV=4,ML_INFERENCE_CPU_CORES_ENV=24,SMARTSIM_DEDICATED_DB_NODES_ENV=1,SMARTSIM_PIN_DB_NODELIST=1,RANK_GRID_X_ENV=4,RANK_GRID_Z_ENV=4,TARGET_WIDTH_ENV=192,TARGET_HEIGHT_ENV=108,CHUNK_SIZE_ENV=12,ML_BATCH_SIZE_ENV=50000,MODEL_NAME_ENV=benchmark_giant_mlp,MODEL_BACKEND_ENV=TORCH,SMARTSIM_BALANCED_KEYS=${balanced},SMARTSIM_INTRA_OP_THREADS=1,SMARTSIM_INTER_OP_THREADS=1,SMARTSIM_THREADS_PER_QUEUE=24,JOB_NAME_ENV=${job_name}"

  print "Submitting ${job_name}: one 16-rank solver node and four 24-core CPU SmartSim nodes"
  dependency_args=()
  if [[ -n "${previous_job_id}" ]]; then
    dependency_args=("--dependency=afterany:${previous_job_id}")
  fi
  previous_job_id=$(sbatch --parsable \
    "${dependency_args[@]}" \
    --export="${exports}" \
    --partition=c23mm \
    --nodes=5 \
    --ntasks-per-node=1 \
    --cpus-per-task=24 \
    --mem-per-cpu=5G \
    --time="${TIME_LIMIT}" \
    "${RUNNER}")
  print "Submitted batch job ${previous_job_id}"
done
