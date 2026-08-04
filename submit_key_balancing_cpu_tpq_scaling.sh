#!/usr/bin/env zsh

# CPU ML Scaling Benchmark with Dynamic TPQ and Intra-Op Threading
# Sweeps DB_NODES in 1, 3, 4, 5, 6 (skipping size 2 unsupported by SmartSim)
# For each db_nodes:
#   TPQ = floor(24 / db_nodes)
#   Intra-Op Threads (I) = ceil(24 / TPQ)
# Uses 24 solver ranks (6x4 grid on 216x144 resolution, 1 request/rank)
# Perfect hash-tag load balancing enabled (SMARTSIM_BALANCED_KEYS=1)
# 10 steps per run (steps 4, 6, 8, 10 = 4 steady-state ML steps per run)
# All jobs chained with dependencies to prevent port collisions.

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
RUNNER="${SCRIPT_DIR}/run_key_balancing_cpu_debug.sh"
MANIFEST_FILE="${SCRIPT_DIR}/cpu_tpq_scaling_jobs.txt"
BUILD_DIR="build_key_balance_direct"
MPI_RANKS=24
ALLOC_CPUS_PER_NODE=24
DB_CPU_CORES=24
TIME_LIMIT="${TIME_LIMIT_ENV:-00:30:00}"
RANK_GRID_X=6
RANK_GRID_Z=4

if [[ ! -x "${SCRIPT_DIR}/solver_cpp/${BUILD_DIR}/terrain_solver" ]]; then
  print -u2 "Missing normal direct build: solver_cpp/${BUILD_DIR}/terrain_solver"
  print -u2 "Run ./build_key_balancing.sh direct first."
  exit 1
fi

rm -f "${MANIFEST_FILE}"

print "Submitting CPU TPQ/Intra-Op ML Scaling Benchmark (1, 3, 4, 5, 6 DB nodes)..."
print "Grid: 216x144 | Ranks: 24 (6x4) | DB Node sweep: 1,3,4,5,6 | Wall time: ${TIME_LIMIT}"

previous_job_id=""

for db_nodes in 1 3 4 5 6; do
  tpq=$(( DB_CPU_CORES / db_nodes ))
  if (( tpq < 1 )); then tpq=1; fi
  
  # Calculate intra = ceil(24 / tpq)
  intra=$(( (DB_CPU_CORES + tpq - 1) / tpq ))
  
  job_name="cpucorescale_n${db_nodes}_tpq${tpq}_i${intra}"
  exports="ALL,USE_SCOREP_ENV=0,USE_SMARTSIM=1,ML_INTERFACE_ENV=smartsim,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${BUILD_DIR},USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,TOTAL_STEPS_ENV=10,SAVE_EVERY_ENV=10,MPI_RANKS_ENV=${MPI_RANKS},DB_NODES_ENV=${db_nodes},ML_INFERENCE_CPU_CORES_ENV=${DB_CPU_CORES},SMARTSIM_DEDICATED_DB_NODES_ENV=1,SMARTSIM_PIN_DB_NODELIST=1,RANK_GRID_X_ENV=${RANK_GRID_X},RANK_GRID_Z_ENV=${RANK_GRID_Z},TARGET_WIDTH_ENV=216,TARGET_HEIGHT_ENV=144,CHUNK_SIZE_ENV=12,ML_BATCH_SIZE_ENV=50000,MODEL_NAME_ENV=benchmark_giant_mlp,MODEL_BACKEND_ENV=TORCH,SMARTSIM_BALANCED_KEYS=1,SMARTSIM_INTRA_OP_THREADS=${intra},SMARTSIM_INTER_OP_THREADS=1,SMARTSIM_THREADS_PER_QUEUE=${tpq},JOB_NAME_ENV=${job_name}"

  dependency_args=()
  if [[ -n "${previous_job_id}" ]]; then
    dependency_args=("--dependency=afterany:${previous_job_id}")
  fi

  previous_job_id=$(sbatch --parsable \
    "${dependency_args[@]}" \
    --export="${exports}" \
    --partition=c23mm \
    --nodes=$((db_nodes + 1)) \
    --ntasks-per-node="${ALLOC_CPUS_PER_NODE}" \
    --cpus-per-task=1 \
    --mem-per-cpu=5G \
    --time="${TIME_LIMIT}" \
    "${RUNNER}")
  print "Submitted ${job_name} (${db_nodes} DB nodes, TPQ=${tpq}, Intra=${intra}): Job ID ${previous_job_id}"
  print "${db_nodes} direct balanced_control ${previous_job_id} tpq=${tpq} intra=${intra}" >> "${MANIFEST_FILE}"
done

print "All 5 CPU TPQ scaling jobs submitted successfully in dependency chain."
print "Job manifest saved to ${MANIFEST_FILE}."
