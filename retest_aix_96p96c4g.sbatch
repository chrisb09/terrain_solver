#!/bin/zsh
# AIx retest -- heterogeneous surrogate 96 (CPU) + 96 (GPU) ranks / 4 GPUs
# (96p96c4g), 3x3 AIx matrix. Both het groups are c23g nodes.
#SBATCH --job-name=retest96p96c4g
#SBATCH --account=rwth0792
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --output=logs/retest_aix_96p96c4g_%j.txt

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=5G

#SBATCH hetjob

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=5G

set -uo pipefail
cd /hpcwork/ro092286/smartsim/mini_app || exit 1
export CFG_ENV=96p96c4g
export BUILD_SET_ENV=het_matrix
export DRIVER_ENV=/hpcwork/ro092286/smartsim/mini_app/5r_driver_hetsurrogate.zsh
export APPLY_CUDA_STUBS_ENV=1
zsh "${RUNNER_ENV:-/hpcwork/ro092286/smartsim/mini_app/retest_aix_cells.zsh}"
