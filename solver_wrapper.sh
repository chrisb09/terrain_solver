#!/bin/bash
set -euo pipefail

# Score-P: add CUDA stubs only on CPU nodes (GPU nodes have the real driver).
# Detection: check SLURM_JOB_PARTITION, existence of /dev/nvidia* devices, or availability of the real libnvidia-ml.so.1 library in library paths.
echo "[WRAPPER] _CUDA_STUBS=${_CUDA_STUBS:-} SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-} host=$(hostname)"
if [[ -n "${_CUDA_STUBS:-}" ]]; then
    is_gpu_node=0
    # Check if the real driver library is actually available
    if [[ -f "/usr/lib64/libnvidia-ml.so.1" ]] || [[ -f "/lib64/libnvidia-ml.so.1" ]]; then
        is_gpu_node=1
    elif [[ -e "/dev/nvidia0" ]] || [[ -e "/dev/nvidiactl" ]]; then
        # Check if we actually have a GPU allocated in this job to avoid loader errors on CPU-only runs
        if [[ "${CUDA_VISIBLE_DEVICES:-}" != "NoDevFiles" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            is_gpu_node=1
        fi
    fi
    echo "[WRAPPER] is_gpu_node=${is_gpu_node}"
    if [[ "${is_gpu_node}" -eq 0 ]]; then
        export LD_LIBRARY_PATH="${_CUDA_STUBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "[WRAPPER] Prepended stubs. LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    fi
fi

# Profiles are independent per rank in MPMD mode. An OTF2 trace, however,
# requires all MPI ranks to write one shared Score-P experiment directory.
if [[ -n "${SCOREP_EXPERIMENT_DIRECTORY:-}" && "${SCOREP_ENABLE_TRACING:-false}" != "true" && "${SCOREP_ENABLE_TRACING:-false}" != "1" ]]; then
  if [[ -n "${OMPI_COMM_WORLD_RANK:-}" ]]; then
    export SCOREP_EXPERIMENT_DIRECTORY="${SCOREP_EXPERIMENT_DIRECTORY}_rank_${OMPI_COMM_WORLD_RANK}"
  elif [[ -n "${PMIX_RANK:-}" ]]; then
    export SCOREP_EXPERIMENT_DIRECTORY="${SCOREP_EXPERIMENT_DIRECTORY}_rank_${PMIX_RANK}"
  fi
  mkdir -p "${SCOREP_EXPERIMENT_DIRECTORY}"
fi

exec "${_SOLVER_BINARY:-$(dirname "$0")/solver_cpp/build/terrain_solver}" "$@"
