#!/bin/zsh
############################
# Calibration smoke -- single c23g node, 96 ranks / 4 GPUs (96c4g).
# Cells: WaterCNN 8k + Giant-MLP 270p, AIx default/coll, SmartSim shared_c0,
# and SmartSim per-GPU DB c0 (4 DBs pinned to the 4 GPUs).
############################
#SBATCH --job-name=cal_retest_96c4g
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/cal_retest_96c4g_%j.txt

set -uo pipefail
cd /hpcwork/ro092286/smartsim/mini_app || exit 1

export CFG_ENV=96c4g
export MPI_RANKS_ENV=96
export RANK_GRID_X_ENV=12
export RANK_GRID_Z_ENV=8
export GPUS_PER_NODE_ENV=4

zsh /hpcwork/ro092286/smartsim/mini_app/calibrate_retest_cells.zsh
