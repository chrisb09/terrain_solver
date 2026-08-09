#!/bin/zsh
# Run 35 Score-P sequential-put cases for giant_mlp at 1/100th scale (840x420 grid, bs=600k)
# Executing 42 simulation steps per run (21 ML steps, 20 warm steady-state ML samples)

#SBATCH --job-name=CMI_SmartSim_42step_giantmlp_suite
#SBATCH --account=thes2181
#SBATCH --time=02:45:00
#SBATCH --exclusive
#SBATCH --output=logs/scorep_giantmlp_suite_%j.txt

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
#SBATCH --exclusive

set -uo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${0:A:h}}"
JOB_SCRIPT="${SCRIPT_DIR}/proper_slurm_job.sh"
SUITE_ID="${SLURM_JOB_ID:-$$}"
overall_status=0
case_index=0

mkdir -p "${SCRIPT_DIR}/logs"

batch_size=600000
batch_label="600k"
model_key="giantmlp"
model_name="benchmark_giant_mlp"
model_path="train_models/model_a/giant_cuda.pt"
manifest_path="train_models/model_a/artifact_manifest_giant_flat.json"
width=840
height=420
scale_label="scale100th"

for chains in 0 1 2 3 4 5 6; do
  for repeat in 1 2 3 4 5; do
    (( case_index++ ))
    job_name="CMI_SmartSim_${model_key}_${scale_label}_c${chains}_rep${repeat}_42step"
    run_id="${SUITE_ID}_${case_index}_${model_key}_${scale_label}_c${chains}_rep${repeat}"
    scorep_tag="smartsim_${model_key}_${scale_label}_bs${batch_label}_c${chains}_run${repeat}"
    run_log="${SCRIPT_DIR}/logs/scorep_giantmlp_suite_${SUITE_ID}_${case_index}_${model_key}_${scale_label}_c${chains}_rep${repeat}.log"

    print "Starting ${job_name} (${case_index}/35); log=${run_log}"
    env \
      RUN_ID_ENV="${run_id}" \
      SCOREP_DIR_TAG_ENV="${scorep_tag}" \
      USE_SCOREP_ENV=1 \
      SCOREP_ENABLE_TRACING_ENV=true \
      SKIP_COMPILE_ENV=1 \
      COMPILE_OUTPUT_PATH_ENV=build_scorep \
      USE_SMARTSIM=0 \
      CPP_ML_INTERFACE_PROVIDER_ENV=SMARTSIM \
      ML_BATCH_SIZE_ENV="${batch_size}" \
      SMARTSIM_MPI_SEQUENTIAL_PUT="${chains}" \
      TOTAL_STEPS_ENV=42 \
      OVERWRITE_OUTPUT_ENV=1 \
      FORCE_FRESH_RUN_ENV=1 \
      MODEL_NAME_ENV="${model_name}" \
      MODEL_PATH_ENV="${model_path}" \
      MODEL_ARTIFACT_MANIFEST_ENV="${manifest_path}" \
      TARGET_WIDTH_ENV="${width}" \
      TARGET_HEIGHT_ENV="${height}" \
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
    sleep 3
  done
done

exit "${overall_status}"
