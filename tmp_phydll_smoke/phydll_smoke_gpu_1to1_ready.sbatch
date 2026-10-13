#!/bin/zsh
#SBATCH --job-name=phydll_smoke_gpu_1to1_ready
#SBATCH --account=rwth0792
#SBATCH --time=00:05:00
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/phydll_smoke_gpu_1to1_ready_%j.log
# Component 0: Solver on c23g (GPU hidden)
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G

#SBATCH hetjob

# Component 1: DL client on c23g (GPU active)
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G

SCRIPT_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app"
cd "${SCRIPT_DIR}"
exec /bin/zsh "${SCRIPT_DIR}/proper_slurm_job.sh"
