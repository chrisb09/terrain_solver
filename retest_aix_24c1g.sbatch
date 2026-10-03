#!/bin/zsh
# AIx retest -- single c23g node, 24 ranks / 1 GPU (24c1g), 3x3 AIx matrix.
#SBATCH --job-name=retest24c1g
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=logs/retest_aix_24c1g_%j.txt
set -uo pipefail
cd /hpcwork/ro092286/smartsim/mini_app || exit 1
export CFG_ENV=24c1g
export BUILD_SET_ENV=matrix
export DRIVER_ENV=/hpcwork/ro092286/smartsim/mini_app/5r_driver_single.zsh
export RANKS_ENV=24 GRID_X_ENV=6 GRID_Z_ENV=4 GPUS_ENV=1
zsh "${RUNNER_ENV:-/hpcwork/ro092286/smartsim/mini_app/retest_aix_cells.zsh}"
