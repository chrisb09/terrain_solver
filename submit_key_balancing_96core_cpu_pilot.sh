#!/usr/bin/env zsh

# 96-Core Exclusive CPU Pilot Benchmark:
# Verifies 96-core exclusive node allocations, DB pinning, solver task launch (96 ranks),
# and compares Dynamic (TPQ*Intra=96), TPQ-Only (TPQ=96, Intra=1), and Intra-Only (TPQ=1, Intra=96).

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
RUNNER="${SCRIPT_DIR}/run_key_balancing_cpu_debug.sh"
MANIFEST_FILE="${SCRIPT_DIR}/cpu_96core_pilot_jobs.txt"
BUILD_DIR="build_key_balance_direct"
MPI_RANKS=96
ALLOC_CPUS_PER_NODE=96
DB_CPU_CORES=96
TIME_LIMIT="${TIME_LIMIT_ENV:-00:30:00}"
RANK_GRID_X=12
RANK_GRID_Z=8

if [[ ! -x "${SCRIPT_DIR}/solver_cpp/${BUILD_DIR}/terrain_solver" ]]; then
  print -u2 "Missing normal direct build: solver_cpp/${BUILD_DIR}/terrain_solver"
  print -u2 "Run ./build_key_balancing.sh direct first."
  exit 1
fi

rm -f "${MANIFEST_FILE}"

print "Submitting 96-Core Exclusive CPU Pilot Benchmark..."
print "Grid: 1440x960 | Ranks: 96 (12x8) | Node Allocation: Exclusive 96 cores/node | Wall time: ${TIME_LIMIT}"

previous_job_id=""

configs=(
  "1 96 1 dynamic"
  "3 96 1 tpq_only"
  "3 1 96 intra_only"
  "3 32 3 dynamic"
)

for config in "${configs[@]}"; do
  parts=(${=config})
  db_nodes="${parts[1]}"
  tpq="${parts[2]}"
  intra="${parts[3]}"
  mode="${parts[4]}"

  job_name="cpu96pilot_n${db_nodes}_${mode}_tpq${tpq}_i${intra}"
  exports="ALL,USE_SCOREP_ENV=0,USE_SMARTSIM=1,ML_INTERFACE_ENV=smartsim,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${BUILD_DIR},USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,TOTAL_STEPS_ENV=10,SAVE_EVERY_ENV=10,MPI_RANKS_ENV=${MPI_RANKS},DB_NODES_ENV=${db_nodes},ML_INFERENCE_CPU_CORES_ENV=${DB_CPU_CORES},SMARTSIM_DEDICATED_DB_NODES_ENV=1,SMARTSIM_PIN_DB_NODELIST=1,RANK_GRID_X_ENV=${RANK_GRID_X},RANK_GRID_Z_ENV=${RANK_GRID_Z},TARGET_WIDTH_ENV=1440,TARGET_HEIGHT_ENV=960,CHUNK_SIZE_ENV=12,ML_BATCH_SIZE_ENV=50000,MODEL_NAME_ENV=benchmark_giant_mlp,MODEL_BACKEND_ENV=TORCH,SMARTSIM_BALANCED_KEYS=1,SMARTSIM_INTRA_OP_THREADS=${intra},SMARTSIM_INTER_OP_THREADS=1,SMARTSIM_THREADS_PER_QUEUE=${tpq},JOB_NAME_ENV=${job_name}"

  dependency_args=()
  if [[ -n "${previous_job_id}" ]]; then
    dependency_args=("--dependency=afterany:${previous_job_id}")
  fi

  previous_job_id=$(sbatch --parsable \
    "${dependency_args[@]}" \
    --export="${exports}" \
    --partition=c23mm \
    --exclusive \
    --nodes=$((db_nodes + 1)) \
    --ntasks-per-node=1 \
    --cpus-per-task="${ALLOC_CPUS_PER_NODE}" \
    --mem=0 \
    --time="${TIME_LIMIT}" \
    "${RUNNER}")
  print "Submitted ${job_name} (${db_nodes} DB nodes, TPQ=${tpq}, Intra=${intra}): Job ID ${previous_job_id}"
  print "${db_nodes} ${mode} ${previous_job_id} tpq=${tpq} intra=${intra}" >> "${MANIFEST_FILE}"
done

print "All 4 pilot 96-core jobs submitted successfully in dependency chain."
print "Job manifest saved to ${MANIFEST_FILE}."
