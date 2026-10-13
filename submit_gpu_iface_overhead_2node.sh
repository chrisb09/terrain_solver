#!/usr/bin/env zsh

# 2-node Standalone interface-overhead benchmark, GPU inference.
# Compares Direct SmartSim vs CPP-ML-Interface (CMI) at a single DB shard.
# - 1 c23ms solver node (48 ranks, 8x6) + 1 c23g DB node (4 GPUs) = 2 nodes.
#   The solver uses c23ms (same CPUs as c23mm, smaller memory) with
#   --mem-per-cpu=4G (48 x 4G = 192G) to fit a c23ms node; the c23g DB node keeps
#   --mem-per-cpu=5G. Both solver and DB stay exclusive.
# - DB_NODES=1, shared layout -> SmartSim Standalone (no clustering/sharding).
# - Workload identical to the load-balancing campaign:
#   1920x1080, CHUNK_SIZE=12, ML_BATCH_SIZE=50000, benchmark_giant_mlp (TORCH).
# - 10 steps per run (steps 4, 6, 8, 10 = 4 steady ML steps).
# - Conditions: direct (water_tile_*) and cpp/CMI (input_*).
# - Account: thes2181 (matches the original GPU campaign).
# - Overrides: REPLICATES_ENV, TIME_LIMIT_ENV, DEPENDENCY_ENV.

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
JOB_SCRIPT="${SCRIPT_DIR}/proper_slurm_job.sh"
MANIFEST_LOCAL="${SCRIPT_DIR}/gpu_iface_2node_jobs.txt"
MANIFEST_KEYLB="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/key_loadbalancing/gpu_iface_2node_jobs.txt"

SLURM_ACCOUNT="thes2181"
DB_NODES=1
GPUS_PER_NODE=4
MPI_RANKS=48
TIME_LIMIT="${TIME_LIMIT_ENV:-00:20:00}"
RANK_GRID_X=8
RANK_GRID_Z=6
REPLICATES="${REPLICATES_ENV:-5}"

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
print " Submitting GPU 2-node Standalone interface-overhead suite"
print " Account: ${SLURM_ACCOUNT} | 1x c23ms solver (4G/cpu) + 1x c23g DB (4 GPUs) = 2 nodes"
print " Grid: 1920x1080 | Ranks: ${MPI_RANKS} (8x6) | DB_NODES=${DB_NODES} (Standalone)"
print " Conditions: direct, cpp | Replicates: ${REPLICATES} (4 steady ML steps/run)"
print "========================================================================"

common_exports="USE_SCOREP_ENV=0,SKIP_COMPILE_ENV=1,USE_LOCAL_MODEL_CACHE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,TOTAL_STEPS_ENV=10,SAVE_EVERY_ENV=10,MPI_RANKS_ENV=${MPI_RANKS},DB_NODES_ENV=${DB_NODES},RANK_GRID_X_ENV=${RANK_GRID_X},RANK_GRID_Z_ENV=${RANK_GRID_Z},TARGET_WIDTH_ENV=1920,TARGET_HEIGHT_ENV=1080,CHUNK_SIZE_ENV=12,ML_BATCH_SIZE_ENV=50000,MODEL_NAME_ENV=benchmark_giant_mlp,MODEL_BACKEND_ENV=TORCH,SMARTSIM_BALANCED_KEYS=1"

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
      job_name="gpu_iface_prod_r${rep}_direct"
      exports="ALL,${common_exports},COMPILE_OUTPUT_PATH_ENV=build_key_balance_direct,JOB_NAME_ENV=${job_name},USE_SMARTSIM=1,ML_INTERFACE_ENV=smartsim"
    else
      job_name="gpu_iface_prod_r${rep}_cpp"
      exports="ALL,${common_exports},COMPILE_OUTPUT_PATH_ENV=build_key_balance_cpp,JOB_NAME_ENV=${job_name},USE_SMARTSIM=0,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config_keybalance.toml"
    fi

    submit_raw=$(sbatch --parsable \
      "${dependency_args[@]}" \
      --account="${SLURM_ACCOUNT}" \
      --export="${exports}" \
      --partition=c23ms \
      --exclusive \
      --nodes=1 \
      --ntasks-per-node="${MPI_RANKS}" \
      --cpus-per-task=1 \
      --mem-per-cpu=4G \
      --time="${TIME_LIMIT}" \
      : \
      --partition=c23g \
      --exclusive \
      --nodes="${DB_NODES}" \
      --ntasks-per-node=1 \
      --cpus-per-task=24 \
      --gres="gpu:${GPUS_PER_NODE}" \
      --mem-per-cpu=5G \
      --time="${TIME_LIMIT}" \
      "${JOB_SCRIPT}")

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
print " All ${job_count} GPU 2-node interface-overhead jobs submitted."
print " Manifest: ${MANIFEST_LOCAL}"
print "           ${MANIFEST_KEYLB}"
print "========================================================================"
