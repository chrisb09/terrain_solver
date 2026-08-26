#!/bin/zsh
#SBATCH --account=default
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=00:10:00
#SBATCH --gres=none

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-/hpcwork/ro092286/smartsim/mini_app}"
cd "${SCRIPT_DIR}"
exec /bin/zsh "${SCRIPT_DIR}/proper_slurm_job.sh"
