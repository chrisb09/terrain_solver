#!/bin/bash
set -euo pipefail

# HET surrogate: this task runs on a c23g node that stands in for a c23mm
# (CPU-only) node; hide the reserved GPUs from the CUDA runtime. Slurm's GPU
# binding sets CUDA_VISIBLE_DEVICES and would override an srun --export value,
# so the marker is applied here, at process start.
if [[ "${HET_SURROGATE_HIDE_GPU:-0}" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES="NoDevFiles"
    echo "[WRAPPER] HET_SURROGATE_HIDE_GPU: forced CUDA_VISIBLE_DEVICES=NoDevFiles"
fi

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

# Historical profiles use per-rank dirs; shared profiles and OTF2 traces keep
# the same experiment directory across the instrumented MPI world.
if [[ -n "${SCOREP_EXPERIMENT_DIRECTORY:-}" && "${SCOREP_SHARED_PROFILE_ENV:-0}" != "1" && "${SCOREP_ENABLE_TRACING:-false}" != "true" && "${SCOREP_ENABLE_TRACING:-false}" != "1" ]]; then
  if [[ -n "${OMPI_COMM_WORLD_RANK:-}" ]]; then
    export SCOREP_EXPERIMENT_DIRECTORY="${SCOREP_EXPERIMENT_DIRECTORY}_rank_${OMPI_COMM_WORLD_RANK}"
  elif [[ -n "${PMIX_RANK:-}" ]]; then
    export SCOREP_EXPERIMENT_DIRECTORY="${SCOREP_EXPERIMENT_DIRECTORY}_rank_${PMIX_RANK}"
  elif [[ -n "${SLURM_PROCID:-}" ]]; then
    export SCOREP_EXPERIMENT_DIRECTORY="${SCOREP_EXPERIMENT_DIRECTORY}_rank_${SLURM_PROCID}"
  fi
  mkdir -p "${SCOREP_EXPERIMENT_DIRECTORY}"
fi

exec "${_SOLVER_BINARY:-$(dirname "$0")/solver_cpp/build/terrain_solver}" "$@"
