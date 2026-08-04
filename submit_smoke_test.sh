#!/bin/bash

# Overrides for the smoke test
export TARGET_WIDTH_ENV=${TARGET_WIDTH_ENV:-1920}
export TARGET_HEIGHT_ENV=${TARGET_HEIGHT_ENV:-1080}
export MODEL_NAME_ENV="${MODEL_NAME_ENV:-perfect_model}"
export CPP_ML_INTERFACE_PROVIDER_ENV=${CPP_ML_INTERFACE_PROVIDER_ENV:-AIX}
export MPI_RANKS_ENV=${MPI_RANKS_ENV:-96}
export TOTAL_STEPS_ENV=${TOTAL_STEPS_ENV:-10}
export FORCE_FRESH_RUN_ENV=${FORCE_FRESH_RUN_ENV:-1}
export USE_SCOREP_ENV=${USE_SCOREP_ENV:-0}
export SCOREP_MPP_ENV=${SCOREP_MPP_ENV:-${SCOREP_MPP:-mpi}}
export OVERWRITE_OUTPUT=1

# Resource configuration for devel partition
PARTITION="devel"
ACCOUNT="default"
TIME="01:00:00"
MEM="238G"
NODES=${NODES_ENV:-1}
TOTAL_TASKS=${TOTAL_TASKS_ENV:-96}
NTASKS_PER_NODE=${NTASKS_PER_NODE_ENV:-96}

if [[ "${CPP_ML_INTERFACE_PROVIDER_ENV}" == "PHYDLL" ]]; then
  # Optimize PhyDLL smoke topology to avoid massive single-DL aggregation bottlenecks
  # 7 ranks total: 6 solver ranks, 1 DL client rank
  # 6 solver ranks splits perfectly into 3x2 cartesian grid:
  # chunks_z (18) is divisible by 3, chunks_x (32) is divisible by 2.
  # This guarantees all solver ranks have identical local domain size, avoiding DL client shape mismatch.
  TOTAL_TASKS=7
  NTASKS_PER_NODE=7
  MPI_RANKS_ENV=7
  # Ensure we run at least 2 steps so that step 2 runs ML inference (since ML is on even steps)
  TOTAL_STEPS_ENV=2
elif [[ "${CPP_ML_INTERFACE_PROVIDER_ENV}" == "SMARTSIM" ]]; then
  NODES=${NODES_ENV:-2}
  TOTAL_TASKS=32
  NTASKS_PER_NODE=16
fi

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
HWCTR_ARG=""
if [[ "${USE_SCOREP_ENV}" == "1" ]]; then
  HWCTR_ARG="--hwctr=papi"
fi

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
  echo "export MINI_APP_DIR=\"$(pwd -P)\""
  if [[ "${CPP_ML_INTERFACE_PROVIDER_ENV}" == "SMARTSIM" ]]; then
    echo "export DB_NODES_ENV=1"
    echo "export MODEL_ARTIFACT_MANIFEST_ENV=train_models/model_a/artifact_manifest_perfect_flat.json"
    echo "export USE_CPP_ML_INTERFACE=1"
  fi
  if [[ "${USE_SCOREP_ENV}" == "1" ]]; then
    echo "export USE_SCOREP=1"
    echo "export SCOREP_MPP_ENV=\"${SCOREP_MPP_ENV}\""
    echo "export SCOREP_MPP=\"${SCOREP_MPP_ENV}\""
    echo "export SCOREP_METRIC_PAPI=\"\""
    if [[ "${USE_PYTHON_DL_CLIENT:-1}" == "1" ]]; then
      echo "export PHYDLL_REBUILD_DL_CLIENT=0"
    fi
  fi
  echo "export USE_PYTHON_DL_CLIENT=${USE_PYTHON_DL_CLIENT:-1}"
  # Append the original script but skip the first line (shebang) and comment out #SBATCH
  sed '1d; s/^#SBATCH/##SBATCH/g' proper_slurm_job.sh
} > "${TEMP_JOB_SCRIPT}"

# Wrap sbatch to run the job
# Use --export=ALL to be double-sure
sbatch \
  ${HWCTR_ARG} \
  --partition=${PARTITION} \
  --account=${ACCOUNT} \
  --time=${TIME} \
  --nodes=${NODES} \
  --ntasks=${TOTAL_TASKS} \
  --ntasks-per-node=${NTASKS_PER_NODE} \
  --cpus-per-task=1 \
  --mem=${MEM} \
  --gres=none \
  --export=ALL \
  --output="logs/mini_app_output_%j.txt" \
  --job-name="smoke_test_${CPP_ML_INTERFACE_PROVIDER_ENV}" \
  "${TEMP_JOB_SCRIPT}"

# Cleanup temp script
rm "${TEMP_JOB_SCRIPT}"
