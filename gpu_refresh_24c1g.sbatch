#!/bin/bash
#SBATCH --job-name=gpu-refresh-24c1g
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
exec python3 "${GPU_REFRESH_MINI_ENV:?}/gpu_refresh_campaign.py" --run "$@"
