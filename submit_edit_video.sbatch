#!/bin/zsh
#SBATCH --job-name=edit_video
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --account=rwth2150
#SBATCH --partition=c23ms
#SBATCH --output=logs/edit_video_%j.out

MODE="${MODE:-preview}"

MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
OUT_DIR="/rwthfs/rz/cluster/hpcwork/thes2181/mini_app/circle_r300_d300_s10000000_triangular_2160x1080_c23ms_1n_24t_1c__rank0_gather_none_ts2/edited"

set -euo pipefail

cd "${MINI_APP_DIR}"
mkdir -p logs

module purge
module load GCCcore/14.2.0
module load FFmpeg/7.1.1

PYBIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin"
if [[ -x "${PYBIN}/python3" ]]; then
  export PATH="${PYBIN}:${PATH}"
fi

echo "MODE=${MODE} host=$(hostname) date=$(date)"
ffmpeg -version | head -1
python3 -c "import numpy, PIL, h5py; print('python deps OK')"

START_TIME=$(date +%s)

if [[ "${MODE}" == "preview" ]]; then
  mkdir -p "${OUT_DIR}/frames_preview"
  find "${OUT_DIR}/frames_preview" -maxdepth 1 -name '*.jpg' -delete
  python3 render_edit.py --mode preview --preview-seconds 48 --jobs 24
  python3 make_audio.py
  ffmpeg -y \
    -framerate 30 \
    -i "${OUT_DIR}/frames_preview/%05d.jpg" \
    -i "${OUT_DIR}/audio.wav" \
    -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p \
    -movflags +faststart \
    -c:a aac -b:a 192k \
    -shortest \
    "${OUT_DIR}/terrain_solver_demo_preview.mp4"
else
  mkdir -p "${OUT_DIR}/frames"
  find "${OUT_DIR}/frames" -maxdepth 1 -name '*.jpg' -delete
  python3 render_edit.py --mode full --jobs 24
  python3 make_audio.py
  ffmpeg -y \
    -framerate 30 \
    -i "${OUT_DIR}/frames/%05d.jpg" \
    -i "${OUT_DIR}/audio.wav" \
    -c:v libx264 -preset slow -crf 16 -pix_fmt yuv420p \
    -movflags +faststart \
    -c:a aac -b:a 192k \
    -shortest \
    "${OUT_DIR}/terrain_solver_demo_v3.mp4"
fi

END_TIME=$(date +%s)
echo "MODE=${MODE} finished in $((END_TIME - START_TIME)) seconds"
ls -lh "${OUT_DIR}"/*.mp4
