#!/bin/zsh
# Run the 12 Score-P sequential-put cases in one exclusive heterogeneous allocation.
# The existing c=0 baselines are intentionally not repeated here.

#SBATCH --job-name=CMI_SmartSim_10step_chain_suite
#SBATCH --account=thes2181
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=logs/scorep_chain_suite_%j.txt

# Solver component: one exclusive CPU node.
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5G

# Database/inference component: one exclusive GPU node.
#SBATCH hetjob
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G

set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${0:A:h}}"
JOB_SCRIPT="${SCRIPT_DIR}/proper_slurm_job.sh"
SUITE_ID="${SLURM_JOB_ID:-$$}"
overall_status=0
case_index=0

mkdir -p "${SCRIPT_DIR}/logs"

for batch_size in 50000 600000; do
  if [[ "${batch_size}" == "50000" ]]; then
    batch_label="50k"
  else
    batch_label="600k"
  fi

  for chains in 1 2 3 4 5 6; do
    (( case_index++ ))
    job_name="CMI_SmartSim_${batch_label}_c${chains}_10step_perfect"
    run_id="${SUITE_ID}_${case_index}_${batch_label}_c${chains}"
    run_log="${SCRIPT_DIR}/logs/scorep_chain_suite_${SUITE_ID}_${case_index}_${batch_label}_c${chains}.log"

    print "Starting ${job_name} (${case_index}/12); log=${run_log}"
    env \
      RUN_ID_ENV="${run_id}" \
      USE_SCOREP_ENV=1 \
      SCOREP_ENABLE_TRACING_ENV=true \
      SKIP_COMPILE_ENV=1 \
      COMPILE_OUTPUT_PATH_ENV=build_scorep \
      USE_SMARTSIM=0 \
      CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM \
      ML_BATCH_SIZE_ENV="${batch_size}" \
      SMARTSIM_MPI_SEQUENTIAL_PUT="${chains}" \
      TOTAL_STEPS_ENV=10 \
      OVERWRITE_OUTPUT_ENV=1 \
      FORCE_FRESH_RUN_ENV=1 \
      MODEL_NAME_ENV=perfect_model \
      MODEL_PATH_ENV=train_models/model_a/perfect_cuda.pt \
      MODEL_ARTIFACT_MANIFEST_ENV=train_models/model_a/artifact_manifest_perfect_flat.json \
      OVERWRITE_JOB_NAME_ENV=1 \
      SKIP_PAPI_METRICS=1 \
      JOB_NAME_ENV="${job_name}" \
      USE_LOCAL_RUNTIME_STAGE_ENV=0 \
      zsh "${JOB_SCRIPT}" >| "${run_log}" 2>&1
    run_status=$?

    if (( run_status != 0 )); then
      print -u2 "FAILED ${job_name}: exit=${run_status}"
      overall_status=1
    else
      print "Completed ${job_name}"
    fi
  done
done

exit "${overall_status}"
