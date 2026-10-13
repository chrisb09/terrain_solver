#!/bin/zsh
#SBATCH --job-name=scorep-pilot-96p96c4g
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --chdir=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/scorep_pilot_96p96c4g_%j.out
#SBATCH --error=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/scorep_pilot_96p96c4g_%j.err
#SBATCH hetjob
#SBATCH --account=rwth0792
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:20:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL APPLY_CUDA_STUBS_ENV=1
exec zsh /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/retest_scorep_pilot.zsh 96p96c4g "$@"
