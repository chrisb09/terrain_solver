#!/bin/zsh
# AIx retest -- single c23g node, 96 ranks / 4 GPUs (96c4g), 3x3 AIx matrix.
#SBATCH --job-name=retest96c4g
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=logs/retest_aix_96c4g_%j.txt
set -uo pipefail
cd /hpcwork/ro092286/smartsim/mini_app || exit 1
export CFG_ENV=96c4g
export BUILD_SET_ENV=matrix
export DRIVER_ENV=/hpcwork/ro092286/smartsim/mini_app/5r_driver_single.zsh
export RANKS_ENV=96 GRID_X_ENV=12 GRID_Z_ENV=8 GPUS_ENV=4
zsh "${RUNNER_ENV:-/hpcwork/ro092286/smartsim/mini_app/retest_aix_cells.zsh}"
