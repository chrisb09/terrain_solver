#!/usr/bin/env zsh
# 10-replicate CPU CMI SmartSim benchmark:
# - CMI SmartSim: input_<rank>_0
# - 48 solver ranks (8x6 grid), 8 DB nodes (96 cores each)
# - Total nodes: 9 (1 solver + 8 DB nodes), exclusive allocation
# - Settings: TPQ=6, INTRA=16 (6x16 = 96 threads/node)
# - Resolution: 384x216 (4x workload of 192x108)
# - 10 simulation steps (steps 4, 6, 8, 10 = 4 steady ML steps)
# - 10 replicates x 2 variants (balanced_control vs natural) = 20 jobs
# - Chained dependencies to prevent Redis port 6780 collisions
# - Account: rwth2150, Partition: c23mm

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
RUNNER="${SCRIPT_DIR}/run_key_balancing_cpu_debug.sh"
MANIFEST_LOCAL="${SCRIPT_DIR}/cpu_cmi_replicates_jobs.txt"
MANIFEST_KEYLB="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/key_loadbalancing/cpu_cmi_replicates_jobs.txt"

SLURM_ACCOUNT="rwth2150"
DB_NODES=8
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
REPLICATES=10

build_dir="build_key_balance_cpp"
if [[ ! -x "${SCRIPT_DIR}/solver_cpp/${build_dir}/terrain_solver" ]]; then
  print -u2 "Missing build: solver_cpp/${build_dir}/terrain_solver"
  print -u2 "Run ./build_key_balancing.sh cpp first."
  exit 1
fi

rm -f "${MANIFEST_LOCAL}" "${MANIFEST_KEYLB}"

print "========================================================================"
print " Submitting CPU CMI 10-Replicate Benchmark Suite (20 jobs in chain)"
print " Account: ${SLURM_ACCOUNT} | Partition: c23mm (exclusive 9 nodes)"
print " Grid: ${TARGET_WIDTH}x${TARGET_HEIGHT} (4x work) | Ranks: ${MPI_RANKS} (8x6)"
print " DB Nodes: ${DB_NODES} (96 cores/node) | TPQ=${TPQ} | Intra=${INTRA}"
print " Replicates: ${REPLICATES} (4 steady ML steps per run)"
print "========================================================================"

previous_job_id=""
job_count=0

for rep in $(seq 1 "${REPLICATES}"); do
  # Alternate variant order per replicate to eliminate systematic order bias
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
    job_name="cmi_cpu_r${rep}_${variant}"
    exports="ALL,USE_SCOREP_ENV=0,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${build_dir},USE_LOCAL_RUNTIME_STAGE_ENV=0,USE_LOCAL_MODEL_CACHE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,TOTAL_STEPS_ENV=10,SAVE_EVERY_ENV=10,MPI_RANKS_ENV=${MPI_RANKS},DB_NODES_ENV=${DB_NODES},ML_INFERENCE_CPU_CORES_ENV=${DB_CPU_CORES},SMARTSIM_DEDICATED_DB_NODES_ENV=1,SMARTSIM_PIN_DB_NODELIST=1,RANK_GRID_X_ENV=${RANK_GRID_X},RANK_GRID_Z_ENV=${RANK_GRID_Z},TARGET_WIDTH_ENV=${TARGET_WIDTH},TARGET_HEIGHT_ENV=${TARGET_HEIGHT},CHUNK_SIZE_ENV=12,ML_BATCH_SIZE_ENV=50000,MODEL_NAME_ENV=benchmark_giant_mlp,MODEL_BACKEND_ENV=TORCH,SMARTSIM_BALANCED_KEYS=${balanced},SMARTSIM_INTRA_OP_THREADS=${INTRA},SMARTSIM_INTER_OP_THREADS=1,SMARTSIM_THREADS_PER_QUEUE=${TPQ},JOB_NAME_ENV=${job_name},USE_SMARTSIM=0,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config_keybalance.toml"

    dependency_args=()
    if [[ -n "${previous_job_id}" ]]; then
      dependency_args=("--dependency=afterany:${previous_job_id}")
    fi

    submit_raw=$(sbatch --parsable \
      "${dependency_args[@]}" \
      --account="${SLURM_ACCOUNT}" \
      --export="${exports}" \
      --partition=c23mm \
      --exclusive \
      --nodes=$((DB_NODES + 1)) \
      --ntasks-per-node=1 \
      --cpus-per-task="${ALLOC_CPUS_PER_NODE}" \
      --mem=0 \
      --time="${TIME_LIMIT}" \
      "${RUNNER}" 2>&1)

    previous_job_id=$(print -r -- "${submit_raw}" | grep -oE '[0-9]+' | tail -n 1)
    if [[ -z "${previous_job_id}" ]]; then
      print -u2 "Failed to submit job: ${submit_raw}"
      exit 1
    fi

    job_count=$((job_count + 1))
    print "[${job_count}/20] Submitted rep ${rep} cpp ${variant}: Job ID ${previous_job_id}"
    print "${rep} cpp ${variant} ${previous_job_id}" >> "${MANIFEST_LOCAL}"
    print "${rep} cpp ${variant} ${previous_job_id}" >> "${MANIFEST_KEYLB}"
  done
done

chmod +x "${SCRIPT_DIR}/submit_cpu_cmi_10reps.sh"
print "========================================================================"
print " All 20 CPU replicate jobs submitted successfully."
print " Manifest saved to: ${MANIFEST_LOCAL}"
print "                    ${MANIFEST_KEYLB}"
print "========================================================================"
