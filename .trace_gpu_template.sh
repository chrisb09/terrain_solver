#!/bin/zsh
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:10:00

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-/hpcwork/ro092286/smartsim/mini_app}"
cd "${SCRIPT_DIR}"
exec /bin/zsh "${SCRIPT_DIR}/proper_slurm_job.sh"
