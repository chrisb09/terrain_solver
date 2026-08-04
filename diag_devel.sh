#!/bin/zsh
#SBATCH --job-name=diag_devel
#SBATCH --account=thes2181
#SBATCH --partition=devel
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/diag_devel_%j.txt

module load GCCcore/11.3.0 OpenMPI/4.1.4
echo "=== Running Devel diagnostic ==="
srun ./mpi_test
