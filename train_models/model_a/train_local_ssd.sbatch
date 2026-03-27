#!/bin/zsh
#SBATCH --job-name=train_local_ssd
#SBATCH --partition=c23ml
#SBATCH --account=thes2181
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --output=train_local_ssd.out
#SBATCH --error=train_local_ssd.err

# Train model_a while staging input data to node-local SSD/BeeOND-like storage.
# Supports either:
#   1) HDF5 input via --data-path
#   2) Prepared binary input via --prepared-data-path

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
LOCAL_WORK_DIR="${LOCAL_STAGE_BASE}/train_local_${SLURM_JOB_ID}"
LOCAL_INPUT_DIR="${LOCAL_WORK_DIR}/input"
LOCAL_OUTPUT_DIR="${LOCAL_WORK_DIR}/output"

cd /hpcwork/ro092286/smartsim/ || exit
source ./install.sh cpu || exit

cd "${RUN_DIR}" || exit
mkdir -p "${LOCAL_INPUT_DIR}" "${LOCAL_OUTPUT_DIR}"

TRAIN_ARGS=(
  --model watercnn
  --epochs 50
  --batch-size 4096
  --num-threads 8
  --output "${LOCAL_OUTPUT_DIR}/best_model.pt"
  --inference-output "${LOCAL_OUTPUT_DIR}/best_model_jit.pt"
)

if [[ "${MODE}" == "h5" ]]; then
  echo "[stage-in] copying HDF5 to local SSD/BeeOND ..."
  LOCAL_H5_PATH="${LOCAL_INPUT_DIR}/$(basename "${H5_DATA_PATH}")"
  rsync -ah --info=progress2 "${H5_DATA_PATH}" "${LOCAL_H5_PATH}"
  TRAIN_ARGS+=(--data-path "${LOCAL_H5_PATH}")
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
else
  echo "ERROR: MODE must be either 'h5' or 'prepared'" >&2
  exit 1
fi

echo "[run] starting training"
python3 train.py "${TRAIN_ARGS[@]}"

echo "[stage-out] syncing model artifacts back to shared filesystem ..."
rsync -ah "${LOCAL_OUTPUT_DIR}/" "${RUN_DIR}/"
echo "[stage-out] done"

echo "[cleanup] removing local staged data"
rm -rf "${LOCAL_WORK_DIR}"
