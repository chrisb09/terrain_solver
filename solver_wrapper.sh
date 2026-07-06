#!/bin/bash
set -euo pipefail

# Score-P: add CUDA stubs only on CPU nodes (GPU nodes have the real driver).
# Detection: check SLURM_JOB_PARTITION or existence of /dev/nvidia* devices.
if [[ -n "${_CUDA_STUBS:-}" ]]; then
    is_gpu_node=0
    if [[ "${SLURM_JOB_PARTITION:-}" == "c23g" ]]; then
        is_gpu_node=1
    elif [[ -e "/dev/nvidia0" ]] || [[ -e "/dev/nvidiactl" ]]; then
        is_gpu_node=1
    fi
    if [[ "${is_gpu_node}" -eq 0 ]]; then
        export LD_LIBRARY_PATH="${_CUDA_STUBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
fi

# Unique Score-P directory per rank to avoid write conflicts in MPMD mode
if [[ -n "${OMPI_COMM_WORLD_RANK:-}" ]]; then
  export SCOREP_EXPERIMENT_DIRECTORY="${SCOREP_EXPERIMENT_DIRECTORY}_rank_${OMPI_COMM_WORLD_RANK}"
elif [[ -n "${PMIX_RANK:-}" ]]; then
  export SCOREP_EXPERIMENT_DIRECTORY="${SCOREP_EXPERIMENT_DIRECTORY}_rank_${PMIX_RANK}"
fi

exec "${_SOLVER_BINARY:-$(dirname "$0")/solver_cpp/build/terrain_solver}" "$@"
