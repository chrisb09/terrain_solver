#!/bin/zsh
#SBATCH --job-name=terrain_appendix_figures
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --account=rwth2150
#SBATCH --partition=c23ms
#SBATCH --output=logs/terrain_appendix_figures_%j.out

set -euo pipefail

cd /hpcwork/ro092286/smartsim/ || exit

source /hpcwork/ro092286/smartsim/install.sh
source /hpcwork/ro092286/smartsim/set_env_claix23_cuda12.3.sh

MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
PY_ENV="/hpcwork/ro092286/smartsim/python/smartsim_cpu"
EXTERNAL_DIR="/hpcwork/thes2181/mini_app"
JOB_NAME="circle_r300_d300_s10000000_triangular_2160x1080_c23ms_1n_24t_1c__rank0_gather_none_ts2"
TRAJ_H5="${EXTERNAL_DIR}/${JOB_NAME}/world_trajectory.h5"
OUT_DIR="${EXTERNAL_DIR}/${JOB_NAME}/edited/appendix_figures"

source "${PY_ENV}/bin/activate"
cd "${MINI_APP_DIR}" || exit
mkdir -p "${OUT_DIR}"

for entry in "0:000000" "358:128522" "3162:10000000"; do
  STEP_IDX="${entry%%:*}"
  STEP_LABEL="${entry##*:}"

  echo "Rendering top view at trajectory index ${STEP_IDX} (step ${STEP_LABEL})..."
  python3 render.py \
    --input-hdf5 "${TRAJ_H5}" \
    --step "${STEP_IDX}" \
    --output-png "${OUT_DIR}/terrain_top_step_${STEP_LABEL}.png" \
    --no-overlay-text

  echo "Rendering slice at trajectory index ${STEP_IDX} (step ${STEP_LABEL})..."
  python3 render_slice.py \
    --input-hdf5 "${TRAJ_H5}" \
    --step "${STEP_IDX}" \
    --slice-z 540 \
    --height-min 0 \
    --image-height 1080 \
    --output-image "${OUT_DIR}/terrain_slice_step_${STEP_LABEL}.png" \
    --no-overlay-text
done

echo "Output files:"
ls -l "${OUT_DIR}"
