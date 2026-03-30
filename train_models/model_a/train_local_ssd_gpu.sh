#!/bin/zsh
#SBATCH --job-name=train_local_ssd_gpu
#SBATCH --partition=c23g
#SBATCH --account=thes2181
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --output=train_local_ssd_gpu.out
#SBATCH --error=train_local_ssd_gpu.err

# GPU training with local SSD/BeeOND stage-in.
# Submit with 1-4 GPUs, e.g.:
#   sbatch --gpus=1 --cpus-per-task=24 train_local_ssd_gpu.sh
#   sbatch --gpus=4 --cpus-per-task=96 train_local_ssd_gpu.sh
#
# Notes:
# - Data is staged to local storage.
# - train.py uses DDP only when launched via torchrun.
# - Prepared-data cache mode preloads to CPU RAM (fast I/O), while batches are trained on GPU.

set -euo pipefail

SCRIPT_DIR="/hpcwork/ro092286/smartsim/mini_app/train_models/model_a"
RUN_DIR="${SCRIPT_DIR}"

# Choose one mode by setting MODE to "h5" or "prepared"
MODE="prepared"

# HDF5 mode input
H5_DATA_PATH="/hpcwork/ro092286/smartsim/mini_app/external_data/circle_r300_d300_s10000_periodic_2160x1080_devel_1n_96t_1c__rank0_gather_none/world_trajectory.h5"

# Prepared mode input: either a directory with metadata.json + .bin files or a single .bin file
PREPARED_PATH="${SCRIPT_DIR}/prepare_mpi"

# Local fast storage base directory (override via LOCAL_SSD_DIR if needed)
LOCAL_STAGE_BASE="${LOCAL_SSD_DIR:-${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}}"
LOCAL_WORK_DIR="${LOCAL_STAGE_BASE}/train_local_gpu_${SLURM_JOB_ID}"
LOCAL_INPUT_DIR="${LOCAL_WORK_DIR}/input"
LOCAL_OUTPUT_DIR="${LOCAL_WORK_DIR}/output"

# Resolve number of GPUs allocated by Slurm
GPU_RAW="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}"
if [[ "${GPU_RAW}" =~ '^[0-9]+$' ]]; then
  GPU_COUNT="${GPU_RAW}"
else
  GPU_COUNT="$(echo "${GPU_RAW}" | grep -oE '[0-9]+' | head -n1)"
  GPU_COUNT="${GPU_COUNT:-1}"
fi

if (( GPU_COUNT < 1 )); then
  echo "ERROR: Could not determine GPU count from Slurm env (SLURM_GPUS_ON_NODE='${SLURM_GPUS_ON_NODE:-}', SLURM_GPUS='${SLURM_GPUS:-}')." >&2
  exit 1
fi
if (( GPU_COUNT > 4 )); then
  echo "WARNING: GPU_COUNT=${GPU_COUNT} (>4). Script is tuned for 1-4 GPUs, continuing anyway."
fi

THREADS_PER_RANK=$(( ${SLURM_CPUS_PER_TASK:-1} / GPU_COUNT ))
if (( THREADS_PER_RANK < 1 )); then
  THREADS_PER_RANK=1
fi

cd /hpcwork/ro092286/smartsim/ || exit
source ./install.sh cuda-12 || exit

cd "${RUN_DIR}" || exit
mkdir -p "${LOCAL_INPUT_DIR}" "${LOCAL_OUTPUT_DIR}"

echo "[config] GPUs=${GPU_COUNT}  cpus-per-task=${SLURM_CPUS_PER_TASK:-1}  threads-per-rank=${THREADS_PER_RANK}"

TRAIN_ARGS=(
  --model watercnn
  --epochs 100
  --batch-size 1024
  --num-threads "${THREADS_PER_RANK}"
)

if [[ "${MODE}" == "h5" ]]; then
  echo "[stage-in] copying HDF5 to local SSD/BeeOND ..."
  LOCAL_H5_PATH="${LOCAL_INPUT_DIR}/$(basename "${H5_DATA_PATH}")"
  rsync -ah --info=progress2 "${H5_DATA_PATH}" "${LOCAL_H5_PATH}"
  TRAIN_ARGS+=(--data-path "${LOCAL_H5_PATH}")
  # For HDF5 on GPU, stream mode is usually safest.
  TRAIN_ARGS+=(--cache-mode stream)
elif [[ "${MODE}" == "prepared" ]]; then
  echo "[stage-in] copying prepared input to local SSD/BeeOND ..."
  if [[ -d "${PREPARED_PATH}" ]]; then
    LOCAL_PREPARED_DIR="${LOCAL_INPUT_DIR}/prepared"
    mkdir -p "${LOCAL_PREPARED_DIR}"
    rsync -ah --info=progress2 "${PREPARED_PATH}/" "${LOCAL_PREPARED_DIR}/"
    TRAIN_ARGS+=(--prepared-data-path "${LOCAL_PREPARED_DIR}")
  elif [[ -f "${PREPARED_PATH}" ]]; then
    LOCAL_PREPARED_FILE="${LOCAL_INPUT_DIR}/$(basename "${PREPARED_PATH}")"
    rsync -ah --info=progress2 "${PREPARED_PATH}" "${LOCAL_PREPARED_FILE}"
    META_SRC="$(dirname "${PREPARED_PATH}")/metadata.json"
    if [[ -f "${META_SRC}" ]]; then
      rsync -ah "${META_SRC}" "${LOCAL_INPUT_DIR}/metadata.json"
    fi
    TRAIN_ARGS+=(--prepared-data-path "${LOCAL_PREPARED_FILE}")
  else
    echo "ERROR: PREPARED_PATH does not exist: ${PREPARED_PATH}" >&2
    exit 1
  fi

  # Fast path for prepared data: preload rank-local subset into RAM.
  TRAIN_ARGS+=(--cache-mode cache --num-workers 0)
else
  echo "ERROR: MODE must be either 'h5' or 'prepared'" >&2
  exit 1
fi

echo "[run] starting distributed GPU training with torchrun"
# --standalone is valid for single-node training
# RANK/LOCAL_RANK/WORLD_SIZE are injected by torchrun and consumed by train.py
torchrun --standalone --nnodes=1 --nproc_per_node="${GPU_COUNT}" train.py "${TRAIN_ARGS[@]}"

echo "[stage-out] syncing model artifacts back to shared filesystem ..."
rsync -ah "${LOCAL_OUTPUT_DIR}/" "${RUN_DIR}/"
echo "[stage-out] done"

echo "[cleanup] removing local staged data"
rm -rf "${LOCAL_WORK_DIR}"
