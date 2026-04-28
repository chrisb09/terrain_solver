#!/bin/zsh

# Submit multiple proper_slurm_job.sh runs while varying hetjob component 1:
# - node count
# - gres=gpu:X per node

set -euo pipefail

BASE_SCRIPT="proper_slurm_job.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "Error: ${BASE_SCRIPT} not found in $(pwd)"
  exit 1
fi


COMPILE_OUTPUT_PATH="stable"
MODEL_BACKEND="TF"

DRY_RUN=0 # Set to 1 to print sbatch commands without executing them

EXCLUDE_CONFIGURATIONS=(
  #"1-8:4" # Exclude all configurations with 4 GPUs per node (i.e. 1-8 nodes with 4 GPUs each)
  "1:1"   # Exclude 1 node with 1 GPU (already done for testing)
)

# Sweep values for hetjob component 1 (GPU component).
# Adjust these arrays as needed.
#HET1_NODES=(1 3)
HET1_NODES=(5 6)
#HET1_NODES=(1 3 4 5 6 7 8)
#HET1_GPUS_PER_NODE=(1 2 3 4)
HET1_GPUS_PER_NODE=(1)

# Fixed component definitions (matching proper_slurm_job.sh defaults).
COMP0_PARTITION="c23mm"
COMP0_NODES=1
COMP0_NTASKS_PER_NODE=96
COMP0_CPUS_PER_TASK=1
COMP0_MEM_PER_CPU="5G"

COMP1_PARTITION="c23g"
COMP1_NTASKS_PER_NODE=1
COMP1_CPUS_PER_TASK=24
COMP1_MEM_PER_CPU="5G"

# Optional extra exports for the job script (comma-separated KEY=VALUE entries).
# Example:
# EXTRA_EXPORTS="TOTAL_STEPS_ENV=1000000,SKIP_COMPILE_ENV=1"
EXTRA_EXPORTS="MODEL_BACKEND_ENV=${MODEL_BACKEND},COMPILE_OUTPUT_PATH_ENV=${COMPILE_OUTPUT_PATH},SKIP_COMPILE_ENV=1"

echo "Starting job submission with the following parameters:"
echo "Base script: ${BASE_SCRIPT}"
echo "Component 0 (CPU) - Partition: ${COMP0_PARTITION}, Nodes: ${COMP0_NODES}, Tasks/Node: ${COMP0_NTASKS_PER_NODE}, CPUs/Task: ${COMP0_CPUS_PER_TASK}, Mem/CPU: ${COMP0_MEM_PER_CPU}"
echo "Component 1 (GPU) - Partition: ${COMP1_PARTITION}, Tasks/Node: ${COMP1_NTASKS_PER_NODE}, CPUs/Task: ${COMP1_CPUS_PER_TASK}, Mem/CPU: ${COMP1_MEM_PER_CPU}"
echo "Hetjob Component 1 Sweep - Nodes: ${HET1_NODES[*]}, GPUs/Node: ${HET1_GPUS_PER_NODE[*]}"
echo "Extra exports: ${EXTRA_EXPORTS}"

submitted=0

for het1_nodes in "${HET1_NODES[@]}"; do
  for het1_gpus in "${HET1_GPUS_PER_NODE[@]}"; do
    export_str="ALL,OVERWRITE_JOB_NAME_ENV=1"
    if [[ -n "${EXTRA_EXPORTS}" ]]; then
      export_str="${export_str},${EXTRA_EXPORTS}"
    fi

    # Check if this configuration is in the exclude list
    config_str="${het1_nodes}:${het1_gpus}"
    if [[ " ${EXCLUDE_CONFIGURATIONS[*]} " == *" ${config_str} "* ]]; then
      echo "Skipping excluded configuration: nodes=${het1_nodes}, gpu_per_node=${het1_gpus}"
      continue
    fi

    # Let the time be: 15 minutes + 2 hours / (nodes * gpus per node)
    SLURM_TIME_SECONDS=$((15 * 60 + 2 * 3600 / (het1_nodes * het1_gpus)))
    SLURM_TIME_STR=$(printf "%02d:%02d:%02d" $((SLURM_TIME_SECONDS / 3600)) $(((SLURM_TIME_SECONDS % 3600) / 60)) $((SLURM_TIME_SECONDS % 60)))

    echo "Submitting ${BASE_SCRIPT} with component1: nodes=${het1_nodes}, gres=gpu:${het1_gpus} and time limit ${SLURM_TIME_STR}"

    if [[ ${DRY_RUN} -eq 1 ]]; then
      continue
    fi
    sbatch \
      --export="${export_str}" \
      --partition="${COMP0_PARTITION}" \
      --nodes="${COMP0_NODES}" \
      --ntasks-per-node="${COMP0_NTASKS_PER_NODE}" \
      --cpus-per-task="${COMP0_CPUS_PER_TASK}" \
      --mem-per-cpu="${COMP0_MEM_PER_CPU}" \
      : \
      --partition="${COMP1_PARTITION}" \
      --nodes="${het1_nodes}" \
      --ntasks-per-node="${COMP1_NTASKS_PER_NODE}" \
      --cpus-per-task="${COMP1_CPUS_PER_TASK}" \
      --gres="gpu:${het1_gpus}" \
      --mem-per-cpu="${COMP1_MEM_PER_CPU}" \
      --time="${SLURM_TIME_STR}" \
      "${BASE_SCRIPT}"


    submitted=$((submitted + 1))
  done
done

echo "Submitted ${submitted} jobs in total."
