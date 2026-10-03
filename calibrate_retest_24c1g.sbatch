#!/bin/zsh
############################
# Calibration smoke -- single c23g node, 24 ranks / 1 GPU (24c1g).
# Cells: WaterCNN 8k + Giant-MLP 270p, AIx default/coll + SmartSim shared_c0.
############################
#SBATCH --job-name=cal_retest_24c1g
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/cal_retest_24c1g_%j.txt

set -uo pipefail
cd /hpcwork/ro092286/smartsim/mini_app || exit 1

export CFG_ENV=24c1g
export MPI_RANKS_ENV=24
export RANK_GRID_X_ENV=6
export RANK_GRID_Z_ENV=4
export GPUS_PER_NODE_ENV=1

zsh /hpcwork/ro092286/smartsim/mini_app/calibrate_retest_cells.zsh
