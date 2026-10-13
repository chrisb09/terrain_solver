#!/usr/bin/env zsh

# Fresh 10-replicate GPU benchmark, CMI SmartSim only (input_<rank>_0):
# - Six 4-GPU inference nodes (c23g) + 48 solver ranks (8x6 grid) on one c23mm node.
# - 1920x1080 resolution, 10 steps per run (steps 4, 6, 8, 10 = 4 steady ML steps).
# - 10 replicates x 2 variants (balanced_control, natural) = 20 jobs, chained.
# - Uses config_keybalance.toml which restores the [normalization] section required
#   by the compiled key-balancing binary (config.toml dropped it on 2026-08-25).
# - Account: thes2181 (matches the original GPU campaign).

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
JOB_SCRIPT="${SCRIPT_DIR}/proper_slurm_job.sh"
MANIFEST_LOCAL="${SCRIPT_DIR}/gpu_cmi_replicates_jobs.txt"
MANIFEST_KEYLB="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/key_loadbalancing/gpu_cmi_replicates_jobs.txt"

SLURM_ACCOUNT="thes2181"
DB_NODES=6
GPUS_PER_NODE=4
MPI_RANKS=48
TIME_LIMIT="${TIME_LIMIT_ENV:-00:30:00}"
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
print " Submitting GPU CMI 10-Replicate Benchmark Suite (20 jobs in chain)"
print " Account: ${SLURM_ACCOUNT} | c23mm solver + c23g DB (6x4 GPUs)"
print " Grid: 1920x1080 | Ranks: ${MPI_RANKS} (8x6) | GPUs: $((DB_NODES * GPUS_PER_NODE))"
print " Replicates: ${REPLICATES} (4 steady ML steps per run)"
print "========================================================================"

previous_job_id=""
job_count=0

for rep in $(seq 1 "${REPLICATES}"); do
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
    job_name="gpu_cmi_r${rep}_${variant}"
    exports="ALL,USE_SCOREP_ENV=0,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${build_dir},USE_LOCAL_MODEL_CACHE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,TOTAL_STEPS_ENV=10,SAVE_EVERY_ENV=10,MPI_RANKS_ENV=${MPI_RANKS},DB_NODES_ENV=${DB_NODES},RANK_GRID_X_ENV=${RANK_GRID_X},RANK_GRID_Z_ENV=${RANK_GRID_Z},TARGET_WIDTH_ENV=1920,TARGET_HEIGHT_ENV=1080,CHUNK_SIZE_ENV=12,ML_BATCH_SIZE_ENV=50000,MODEL_NAME_ENV=benchmark_giant_mlp,MODEL_BACKEND_ENV=TORCH,SMARTSIM_BALANCED_KEYS=${balanced},JOB_NAME_ENV=${job_name},USE_SMARTSIM=0,ML_INTERFACE_ENV=cpp,CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM,CPP_ML_CONFIG_ENV=config_keybalance.toml"

    dependency_args=()
    if [[ -n "${previous_job_id}" ]]; then
      dependency_args=("--dependency=afterany:${previous_job_id}")
    fi

    submit_raw=$(sbatch --parsable \
      "${dependency_args[@]}" \
      --account="${SLURM_ACCOUNT}" \
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
      "${JOB_SCRIPT}" 2>&1)

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

print "========================================================================"
print " All 20 GPU CMI replicate jobs submitted successfully."
print " Manifest: ${MANIFEST_LOCAL}"
print "           ${MANIFEST_KEYLB}"
print "========================================================================"
