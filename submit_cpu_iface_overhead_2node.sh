#!/usr/bin/env zsh

# 2-node Standalone interface-overhead benchmark, CPU inference.
# Compares Direct SmartSim vs CPP-ML-Interface (CMI) at a single DB shard.
# - 1 c23mm solver node (48 ranks, 8x6) + 1 c23mm DB node (96 cores) = 2 nodes.
# - DB_NODES=1, shared layout -> SmartSim Standalone (no clustering/sharding).
# - Settings: TPQ=6, INTRA=16 (6x16 = 96 threads/node).
# - Workload identical to the load-balancing campaign:
#   384x216, CHUNK_SIZE=12, ML_BATCH_SIZE=50000, benchmark_giant_mlp (TORCH).
# - 10 steps per run (steps 4, 6, 8, 10 = 4 steady ML steps).
# - Conditions: direct (water_tile_*) and cpp/CMI (input_*).
# - Account: rwth2150, Partition: c23mm.
# - Overrides: REPLICATES_ENV, TIME_LIMIT_ENV, DEPENDENCY_ENV.

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
RUNNER="${SCRIPT_DIR}/run_key_balancing_cpu_debug.sh"
MANIFEST_LOCAL="${SCRIPT_DIR}/cpu_iface_2node_jobs.txt"
MANIFEST_KEYLB="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/key_loadbalancing/cpu_iface_2node_jobs.txt"

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

if [[ ! -x "${SCRIPT_DIR}/solver_cpp/build_key_balance_direct/terrain_solver" ]]; then
  print -u2 "Missing build: solver_cpp/build_key_balance_direct/terrain_solver"
  print -u2 "Run ./build_key_balancing.sh direct first."
  exit 1
fi
if [[ ! -x "${SCRIPT_DIR}/solver_cpp/build_key_balance_cpp/terrain_solver" ]]; then
  print -u2 "Missing build: solver_cpp/build_key_balance_cpp/terrain_solver"
  print -u2 "Run ./build_key_balancing.sh cpp first."
  exit 1
fi

rm -f "${MANIFEST_LOCAL}" "${MANIFEST_KEYLB}"

print "========================================================================"
print " Submitting CPU 2-node Standalone interface-overhead suite"
print " Account: ${SLURM_ACCOUNT} | Partition: c23mm (exclusive 2 nodes)"
print " Grid: ${TARGET_WIDTH}x${TARGET_HEIGHT} | Ranks: ${MPI_RANKS} (8x6)"
print " DB Nodes: ${DB_NODES} (96 cores, Standalone) | TPQ=${TPQ} | Intra=${INTRA}"
print " Conditions: direct, cpp | Replicates: ${REPLICATES} (4 steady ML steps/run)"
print "========================================================================"

common_exports="USE_SCOREP_ENV=0,SKIP_COMPILE_ENV=1,USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,TOTAL_STEPS_ENV=10,SAVE_EVERY_ENV=10,MPI_RANKS_ENV=${MPI_RANKS},DB_NODES_ENV=${DB_NODES},ML_INFERENCE_CPU_CORES_ENV=${DB_CPU_CORES},SMARTSIM_DEDICATED_DB_NODES_ENV=1,SMARTSIM_PIN_DB_NODELIST=1,RANK_GRID_X_ENV=${RANK_GRID_X},RANK_GRID_Z_ENV=${RANK_GRID_Z},TARGET_WIDTH_ENV=${TARGET_WIDTH},TARGET_HEIGHT_ENV=${TARGET_HEIGHT},CHUNK_SIZE_ENV=12,ML_BATCH_SIZE_ENV=50000,MODEL_NAME_ENV=benchmark_giant_mlp,MODEL_BACKEND_ENV=TORCH,SMARTSIM_BALANCED_KEYS=1,SMARTSIM_INTRA_OP_THREADS=${INTRA},SMARTSIM_INTER_OP_THREADS=1,SMARTSIM_THREADS_PER_QUEUE=${TPQ}"

dependency_args=()
if [[ -n "${DEPENDENCY_ENV:-}" ]]; then
  dependency_args=("--dependency=${DEPENDENCY_ENV}")
fi
job_count=0

for rep in $(seq 1 "${REPLICATES}"); do
  # Alternate submission order; Slurm controls execution order.
  if (( rep % 2 == 1 )); then
    conditions=(direct cpp)
  else
    conditions=(cpp direct)
  fi

  for condition in "${conditions[@]}"; do
    if [[ "${condition}" == "direct" ]]; then
      job_name="cpu_iface_prod_r${rep}_direct"
      exports="ALL,${common_exports},COMPILE_OUTPUT_PATH_ENV=build_key_balance_direct,JOB_NAME_ENV=${job_name},USE_SMARTSIM=1,ML_INTERFACE_ENV=smartsim"
    else
      job_name="cpu_iface_prod_r${rep}_cpp"
      exports="ALL,${common_exports},COMPILE_OUTPUT_PATH_ENV=build_key_balance_cpp,JOB_NAME_ENV=${job_name},USE_SMARTSIM=0,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config_keybalance.toml"
    fi

    submit_raw=$(sbatch --parsable \
      "${dependency_args[@]}" \
      --account="${SLURM_ACCOUNT}" \
      --output="${SCRIPT_DIR}/output_%j.txt" \
      --export="${exports}" \
      --exclusive \
      --partition=c23mm \
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
    print "${rep} ${condition} balanced_control ${job_id}" >> "${MANIFEST_LOCAL}"
    print "${rep} ${condition} balanced_control ${job_id}" >> "${MANIFEST_KEYLB}"
  done
done

print "========================================================================"
print " All ${job_count} CPU 2-node interface-overhead jobs submitted."
print " Manifest: ${MANIFEST_LOCAL}"
print "           ${MANIFEST_KEYLB}"
print "========================================================================"
