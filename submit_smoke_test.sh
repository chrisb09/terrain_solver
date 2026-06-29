#!/bin/bash

# Overrides for the smoke test
export TARGET_WIDTH_ENV=${TARGET_WIDTH_ENV:-1920}
export TARGET_HEIGHT_ENV=${TARGET_HEIGHT_ENV:-1080}
export MODEL_NAME_ENV="${MODEL_NAME_ENV:-perfect_model}"
export CPP_ML_INTERFACE_PROVIDER_ENV=${CPP_ML_INTERFACE_PROVIDER_ENV:-AIX}
export MPI_RANKS_ENV=${MPI_RANKS_ENV:-16}
export TOTAL_STEPS_ENV=${TOTAL_STEPS_ENV:-10}
export FORCE_FRESH_RUN_ENV=${FORCE_FRESH_RUN_ENV:-1}
export USE_SCOREP_ENV=${USE_SCOREP_ENV:-0}
export OVERWRITE_OUTPUT=1

# Resource configuration for devel partition
PARTITION="devel"
ACCOUNT="default"
TIME="01:00:00"
MEM_PER_CPU="10000M"
NODES=${NODES_ENV:-1}
if [[ "${CPP_ML_INTERFACE_PROVIDER_ENV}" == "SMARTSIM" ]]; then
  NODES=${NODES_ENV:-2}
fi
TOTAL_TASKS=16

echo "Submitting smoke test to devel partition (Single Allocation Mode)..."
echo "Resolution: ${TARGET_WIDTH_ENV}x${TARGET_HEIGHT_ENV}"
echo "Model: ${MODEL_NAME_ENV}"
echo "Provider: ${CPP_ML_INTERFACE_PROVIDER_ENV}"
echo "Total Steps: ${TOTAL_STEPS_ENV}"
echo "Time Limit: ${TIME}"
echo "Force Fresh Run: ${FORCE_FRESH_RUN_ENV}"

# Ensure logs directory exists
mkdir -p logs

# To prevent sbatch from reading the #SBATCH hetjob directives in the script,
# we create a temporary copy with all #SBATCH lines commented out.
TEMP_JOB_SCRIPT=".smoke_test_$(date +%s).sh"

# We also bake the environment variables directly into the script to ensure 
# they are present even if sbatch environment export is restricted.
{
  echo "#!/bin/zsh"
  echo "export TARGET_WIDTH_ENV=${TARGET_WIDTH_ENV}"
  echo "export TARGET_HEIGHT_ENV=${TARGET_HEIGHT_ENV}"
  echo "export MODEL_NAME_ENV=\"${MODEL_NAME_ENV}\""
  echo "export CPP_ML_INTERFACE_PROVIDER_ENV=\"${CPP_ML_INTERFACE_PROVIDER_ENV}\""
  echo "export MPI_RANKS_ENV=${MPI_RANKS_ENV}"
  echo "export TOTAL_STEPS_ENV=${TOTAL_STEPS_ENV}"
  echo "export FORCE_FRESH_RUN_ENV=${FORCE_FRESH_RUN_ENV}"
  echo "export OVERWRITE_OUTPUT=${OVERWRITE_OUTPUT}"
  if [[ "${USE_SCOREP_ENV}" == "1" ]]; then
    echo "export USE_SCOREP=1"
    echo "export SCOREP_METRIC_PAPI=\"\""
  fi
  echo "export USE_PYTHON_DL_CLIENT=${USE_PYTHON_DL_CLIENT:-1}"
  # Append the original script but skip the first line (shebang) and comment out #SBATCH
  sed '1d; s/^#SBATCH/##SBATCH/g' proper_slurm_job.sh
} > "${TEMP_JOB_SCRIPT}"

# Wrap sbatch with swatch to automatically monitor the job output
# Use --export=ALL to be double-sure
~/scripts/swatch sbatch \
  --partition=${PARTITION} \
  --account=${ACCOUNT} \
  --time=${TIME} \
  --nodes=${NODES} \
  --ntasks=${TOTAL_TASKS} \
  --cpus-per-task=1 \
  --mem-per-cpu=${MEM_PER_CPU} \
  --gres=none \
  --export=ALL \
  --output="logs/mini_app_output_%j.txt" \
  --job-name="smoke_test_${CPP_ML_INTERFACE_PROVIDER_ENV}" \
  "${TEMP_JOB_SCRIPT}"

# Cleanup temp script
rm "${TEMP_JOB_SCRIPT}"
