#!/bin/bash
set -euo pipefail

# Campaign-local copy of solver_wrapper.sh with robust GPU-node detection.
#
# The stock wrapper decides "GPU node" from /usr/lib64/libnvidia-ml.so.1 or
# /dev/nvidia* + non-empty CUDA_VISIBLE_DEVICES. On some c23g nodes neither
# check fires, so it prepends the CUDA *stub*, shadowing the real driver and
# breaking cudaGetDeviceCount (error 34 "CUDA driver is a stub library").
#
# In Slurm hetjobs SLURM_JOB_PARTITION is the *batch* partition for every task
# (c23g here, even on the c23mm ranks), so it must NOT be used to decide. The
# reliable per-task signal is the allocated-GPU list: SLURM_STEP_GPUS /
# SLURM_JOB_GPUS is "0" (or a list) on GPU ranks and empty on CPU-only ranks.
echo "[WRAPPER] _CUDA_STUBS=${_CUDA_STUBS:-} SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-} SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} host=$(hostname)"
if [[ -n "${_CUDA_STUBS:-}" ]]; then
    is_gpu_node=0
    for _g in "${SLURM_STEP_GPUS:-}" "${SLURM_JOB_GPUS:-}"; do
        if [[ -n "${_g}" && "${_g}" != "NoDevFiles" ]]; then
            is_gpu_node=1
        fi
    done
    if [[ "${is_gpu_node}" -eq 0 && -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
        is_gpu_node=1
    fi
    if [[ "${is_gpu_node}" -eq 0 ]] && { [[ -f "/usr/lib64/libnvidia-ml.so.1" ]] || [[ -f "/lib64/libnvidia-ml.so.1" ]]; }; then
        is_gpu_node=1
    fi
    if [[ "${is_gpu_node}" -eq 0 ]] && { [[ -e "/dev/nvidia0" ]] || [[ -e "/dev/nvidiactl" ]]; }; then
        is_gpu_node=1
    fi
    echo "[WRAPPER] is_gpu_node=${is_gpu_node}"
    if [[ "${is_gpu_node}" -eq 0 ]]; then
        export LD_LIBRARY_PATH="${_CUDA_STUBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "[WRAPPER] Prepended stubs. LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    fi
fi

# Het surrogate: the CPU-side group lives on a c23g node and therefore has GPUs
# assigned by Slurm; hide them from AIx for this group only. Slurm sets
# CUDA_VISIBLE_DEVICES per task after --export, so this must happen here.
if [[ "${HET_SURROGATE_HIDE_GPU:-0}" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES="NoDevFiles"
    echo "[WRAPPER] HET_SURROGATE_HIDE_GPU: forced CUDA_VISIBLE_DEVICES=NoDevFiles"
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
