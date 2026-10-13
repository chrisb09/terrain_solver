#!/bin/zsh
# PhyDLL retest -- heterogeneous surrogate 24 (CPU) + 24 (GPU) solver ranks / 1 GPU
# (24p24c1g). 1 C++ DL client runs on the GPU-side node via mpirun MPMD alongside
# the 24 GPU-side solver ranks. The GPU node is allocated 24 ntasks-per-node;
# the driver declares 25 OpenMPI slots (24 solver + 1 DL) in the -host string so
# OpenMPI's internal accounting allows both components within the exclusive node.
# Both het groups are c23g nodes; the CPU-side node hides its GPU.
#SBATCH --job-name=retestP24p24c1g
#SBATCH --account=rwth0792
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --output=logs/retest_phydll_24p24c1g_%j.txt

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
export APPLY_CUDA_STUBS_ENV=1
zsh "${RUNNER_ENV:-/hpcwork/ro092286/smartsim/mini_app/retest_phydll_cells.zsh}"
