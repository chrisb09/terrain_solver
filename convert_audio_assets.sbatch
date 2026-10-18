#!/bin/zsh
#SBATCH --job-name=audio_conv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --account=rwth2150
#SBATCH --partition=c23ms
#SBATCH --output=logs/audio_conv_%j.out

MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"

set -euo pipefail
cd "${MINI_APP_DIR}"
mkdir -p logs assets/audio/wav

module purge
module load GCCcore/14.2.0
module load FFmpeg/7.1.1

for f in assets/audio/*.mp3; do
  base=$(basename "$f" .mp3)
  out="assets/audio/wav/${base}.wav"
  if [[ ! -f "${out}" ]]; then
    echo "convert ${f}"
    ffmpeg -hide_banner -loglevel error -y -i "$f" \
      -ac 2 -ar 48000 -c:a pcm_s16le "${out}"
  fi
done

ls -lh assets/audio/wav/
