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

# Sweep values for hetjob component 1 (GPU component).
# Adjust these arrays as needed.
HET1_NODES=(5 6 7 8)
HET1_GPUS_PER_NODE=(1 2 3 4)

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
EXTRA_EXPORTS=""

submitted=0

for het1_nodes in "${HET1_NODES[@]}"; do
  for het1_gpus in "${HET1_GPUS_PER_NODE[@]}"; do
    export_str="ALL,OVERWRITE_JOB_NAME_ENV=1"
    if [[ -n "${EXTRA_EXPORTS}" ]]; then
      export_str="${export_str},${EXTRA_EXPORTS}"
    fi

    echo "Submitting ${BASE_SCRIPT} with component1: nodes=${het1_nodes}, gres=gpu:${het1_gpus}"

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
      "${BASE_SCRIPT}"

    submitted=$((submitted + 1))
  done
done

echo "Submitted ${submitted} jobs in total."
