#!/bin/zsh
# SmartSim retest -- heterogeneous surrogate 24 (CPU) + 24 (GPU) ranks / 1 GPU
# (24p24c1g). Uses --mem=0 + --exclusive (all node memory) because the SmartSim
# DB process is an extra step on the GPU node and the per-cpu limit caused
# "Memory required by task is not available" at solver-step creation.
#SBATCH --job-name=retestS24p24c1g
#SBATCH --account=rwth0792
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --output=logs/retest_aix_24p24c1g_%j.txt

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0

#SBATCH hetjob

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0

set -uo pipefail
cd /hpcwork/ro092286/smartsim/mini_app || exit 1
export CFG_ENV=24p24c1g
export BUILD_SET_ENV=het_matrix
export DRIVER_ENV=/hpcwork/ro092286/smartsim/mini_app/5r_driver_hetsurrogate.zsh
export APPLY_CUDA_STUBS_ENV=1
zsh "${RUNNER_ENV:-/hpcwork/ro092286/smartsim/mini_app/retest_smartsim_cells.zsh}"
