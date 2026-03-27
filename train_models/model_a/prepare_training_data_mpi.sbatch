#!/bin/zsh
#SBATCH --job-name=prep_pairs_mpi
##SBATCH --partition=devel
#SBATCH --partition=c23mm
#SBATCH --account=thes2181
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
##SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --output=prepare_training_data.out
#SBATCH --error=prepare_training_data.err

# Simple MPI launcher for prepare_training_data.py
# - Splits step-pairs across MPI ranks
# - Merges rank-local dedup tables in a tree reduction
# - Writes final binary output only on rank 0
# - Stages input to node-local SSD/BeeOND-like storage first

set -euo pipefail

# Adjust these paths for your setup
SCRIPT_DIR="/hpcwork/ro092286/smartsim/mini_app/train_models/model_a"
DATA_PATH="/hpcwork/ro092286/smartsim/mini_app/external_data/circle_r300_d300_s10000_periodic_2160x1080_devel_1n_96t_1c__rank0_gather_none/world_trajectory.h5"
OUTPUT_DIR="${SCRIPT_DIR}/prepare_mpi"

# Local fast storage base directory (override via LOCAL_SSD_DIR if needed).
# Priority: LOCAL_SSD_DIR -> BEEOND -> SLURM_TMPDIR -> /tmp.
LOCAL_STAGE_BASE="${LOCAL_SSD_DIR:-${BEEOND:-${SLURM_TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID}}}}"
LOCAL_WORK_DIR="${LOCAL_STAGE_BASE}/prep_pairs_${SLURM_JOB_ID}"
LOCAL_DATA_PATH="${LOCAL_WORK_DIR}/$(basename "${DATA_PATH}")"
LOCAL_OUTPUT_DIR="${LOCAL_WORK_DIR}/prepared_out"

# Optional: activate your environment (must include numpy + h5py + mpi4py)
# source /path/to/venv/bin/activate
# or: conda activate smartsim_cpu

cd /hpcwork/ro092286/smartsim/ || exit
source ./install.sh cpu || exit

cd "${SCRIPT_DIR}" || exit

mkdir -p "${LOCAL_WORK_DIR}"

echo "[stage-in] local work dir: ${LOCAL_WORK_DIR}"
echo "[stage-in] copying input HDF5 to local SSD/BeeOND ..."
rsync -ah --info=progress2 "${DATA_PATH}" "${LOCAL_DATA_PATH}"
echo "[stage-in] done"

echo "[preflight] checking mpi4py in active environment"
python3 -c "import mpi4py; print('mpi4py ok')"

# Notes:
# - Positional args: <data_path> <output_dir>
# - Example below uses 100 steps and writes 4 shuffled output batches
srun --mpi=pmix_v3 python3 -m mpi4py prepare_training_data.py \
  "${LOCAL_DATA_PATH}" \
  "${LOCAL_OUTPUT_DIR}" \
  --max-steps 10000 \
  --num-batches 16 \
  --shuffle-output \
  --topk 20 \
  --progress-chunks 20 \
  --log-rss

echo "[stage-out] syncing prepared output back to shared filesystem ..."
mkdir -p "${OUTPUT_DIR}"
rsync -ah --delete "${LOCAL_OUTPUT_DIR}/" "${OUTPUT_DIR}/"
echo "[stage-out] done"

echo "[cleanup] removing local staged data"
rm -rf "${LOCAL_WORK_DIR}"
