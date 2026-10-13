#!/bin/zsh
#SBATCH --job-name=phydll_smoke_cpu_4to1_ready
#SBATCH --account=rwth2150
#SBATCH --time=00:05:00
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/phydll_smoke_cpu_4to1_ready_%j.log
#SBATCH --partition=c23ms
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2400M

#SBATCH hetjob

#SBATCH --partition=c23ms
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=2400M

SCRIPT_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app"
cd "${SCRIPT_DIR}"
exec /bin/zsh "${SCRIPT_DIR}/proper_slurm_job.sh"
