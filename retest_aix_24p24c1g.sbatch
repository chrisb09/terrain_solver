#!/bin/zsh
# AIx retest -- heterogeneous surrogate 24 (CPU) + 24 (GPU) ranks / 1 GPU
# (24p24c1g), 3x3 AIx matrix. Both het groups are c23g nodes; the CPU-side group
# hides its GPU via the campaign solver wrapper.
#SBATCH --job-name=retest24p24c1g
#SBATCH --account=rwth0792
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --output=logs/retest_aix_24p24c1g_%j.txt

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G

#SBATCH hetjob

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5G

set -uo pipefail
cd /hpcwork/ro092286/smartsim/mini_app || exit 1
export CFG_ENV=24p24c1g
export BUILD_SET_ENV=het_matrix
export DRIVER_ENV=/hpcwork/ro092286/smartsim/mini_app/5r_driver_hetsurrogate.zsh
# HET_SURROGATE_HIDE_GPU must NOT be set job-wide: the driver applies it only to
# the CPU-side het group. Setting it here hides the GPU-side group's GPUs too.
export APPLY_CUDA_STUBS_ENV=1
zsh "${RUNNER_ENV:-/hpcwork/ro092286/smartsim/mini_app/retest_aix_cells.zsh}"
