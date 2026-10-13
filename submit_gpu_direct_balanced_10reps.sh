#!/usr/bin/env zsh

# 10-replicate GPU benchmark, Direct SmartSim only (water_tile_<rank>_0),
# balanced keys (SMARTSIM_BALANCED_KEYS=1) as the interface-overhead control.
# - Six 4-GPU inference nodes (c23g) + 48 solver ranks (8x6 grid) on one c23mm node.
# - 1920x1080 resolution, 10 steps per run (steps 4, 6, 8, 10 = 4 steady ML steps).
# - 10 replicates, balanced_control only = 10 jobs, chained.
# - Account: thes2181 (matches the original GPU campaign).
# - Uses the Direct path (USE_SMARTSIM=1), so no CMI config is involved.

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
JOB_SCRIPT="${SCRIPT_DIR}/proper_slurm_job.sh"
MANIFEST_LOCAL="${SCRIPT_DIR}/gpu_direct_ctrl_replicates_jobs.txt"
MANIFEST_KEYLB="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/key_loadbalancing/gpu_direct_ctrl_replicates_jobs.txt"

SLURM_ACCOUNT="thes2181"
DB_NODES=6
GPUS_PER_NODE=4
MPI_RANKS=48
TIME_LIMIT="${TIME_LIMIT_ENV:-00:30:00}"
RANK_GRID_X=8
RANK_GRID_Z=6
REPLICATES=10

build_dir="build_key_balance_direct"
if [[ ! -x "${SCRIPT_DIR}/solver_cpp/${build_dir}/terrain_solver" ]]; then
  print -u2 "Missing build: solver_cpp/${build_dir}/terrain_solver"
  print -u2 "Run ./build_key_balancing.sh direct first."
  exit 1
fi

rm -f "${MANIFEST_LOCAL}" "${MANIFEST_KEYLB}"

print "========================================================================"
print " Submitting GPU Direct (balanced control) 10-Replicate Suite (10 jobs)"
print " Account: ${SLURM_ACCOUNT} | c23mm solver + c23g DB (6x4 GPUs)"
print " Grid: 1920x1080 | Ranks: ${MPI_RANKS} (8x6) | GPUs: $((DB_NODES * GPUS_PER_NODE))"
print "========================================================================"

previous_job_id=""

for rep in $(seq 1 "${REPLICATES}"); do
  job_name="gpu_direct_r${rep}_control"
  exports="ALL,USE_SCOREP_ENV=0,SKIP_COMPILE_ENV=1,COMPILE_OUTPUT_PATH_ENV=${build_dir},USE_LOCAL_MODEL_CACHE_ENV=1,FORCE_FRESH_RUN_ENV=1,OVERWRITE_OUTPUT_ENV=1,SKIP_RENDERING_ENV=1,TOTAL_STEPS_ENV=10,SAVE_EVERY_ENV=10,MPI_RANKS_ENV=${MPI_RANKS},DB_NODES_ENV=${DB_NODES},RANK_GRID_X_ENV=${RANK_GRID_X},RANK_GRID_Z_ENV=${RANK_GRID_Z},TARGET_WIDTH_ENV=1920,TARGET_HEIGHT_ENV=1080,CHUNK_SIZE_ENV=12,ML_BATCH_SIZE_ENV=50000,MODEL_NAME_ENV=benchmark_giant_mlp,MODEL_BACKEND_ENV=TORCH,SMARTSIM_BALANCED_KEYS=1,JOB_NAME_ENV=${job_name},USE_SMARTSIM=1,ML_INTERFACE_ENV=smartsim"

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

  print "[${rep}/${REPLICATES}] Submitted GPU direct control rep ${rep}: Job ID ${previous_job_id}"
  print "${rep} direct balanced_control ${previous_job_id}" >> "${MANIFEST_LOCAL}"
  print "${rep} direct balanced_control ${previous_job_id}" >> "${MANIFEST_KEYLB}"
done

print "========================================================================"
print " All ${REPLICATES} GPU Direct control jobs submitted."
print " Manifest: ${MANIFEST_LOCAL}"
print "           ${MANIFEST_KEYLB}"
print "========================================================================"
