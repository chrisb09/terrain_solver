#!/bin/zsh
#SBATCH --account=thes2181
#SBATCH --time=00:10:00

# Component 0: CPU solver
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G

#SBATCH hetjob

# Component 1: CPU ML inference / DL client
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-/hpcwork/ro092286/smartsim/mini_app}"
cd "${SCRIPT_DIR}"
exec /bin/zsh "${SCRIPT_DIR}/proper_slurm_job.sh"
