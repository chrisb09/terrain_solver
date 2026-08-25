#!/bin/zsh
#SBATCH --account=thes2181
#SBATCH --time=00:05:00

# Component 0: CPU (c23mm)
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G

#SBATCH hetjob

# Component 1a: GPU (c23g)
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-/hpcwork/ro092286/smartsim/mini_app}"
cd "${SCRIPT_DIR}"
exec /bin/zsh "${SCRIPT_DIR}/proper_slurm_job.sh"
