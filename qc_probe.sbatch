#!/bin/zsh
#SBATCH --job-name=qc_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account=rwth2150
#SBATCH --partition=c23ms
#SBATCH --output=logs/qc_probe_%j.out

module purge
module load GCCcore/14.2.0
module load FFmpeg/7.1.1

EDITED="/rwthfs/rz/cluster/hpcwork/thes2181/mini_app/circle_r300_d300_s10000000_triangular_2160x1080_c23ms_1n_24t_1c__rank0_gather_none_ts2/edited"

for f in "${EDITED}/terrain_solver_demo_preview.mp4" "${EDITED}/terrain_solver_demo_v3.mp4"; do
  if [[ -f "$f" ]]; then
    echo "=== $f"
    ffprobe -v error -show_entries format=duration,size \
      -show_entries stream=codec_name,width,height,r_frame_rate,sample_rate,channels \
      -of default=noprint_wrappers=1 "$f"
  fi
done

mkdir -p "${EDITED}/qc"
V2="${EDITED}/terrain_solver_demo_v3.mp4"
if [[ -f "$V2" ]]; then
  for t in 2 11.5 35 55 72 82 92 97.5 100 104 105.4; do
    ffmpeg -hide_banner -loglevel error -y -ss "$t" -i "$V2" \
      -frames:v 1 -q:v 2 "${EDITED}/qc/mp4_t${t}.jpg"
  done
  ls -la "${EDITED}/qc/"
fi
