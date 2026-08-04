#!/bin/zsh
#SBATCH --job-name=diag_c23ms_c23mm
#SBATCH --account=thes2181
#SBATCH --partition=c23ms
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH hetjob
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/diag_c23ms_c23mm_%j.txt

module load GCCcore/11.3.0 OpenMPI/4.1.4
echo "=== Running C23MS + C23MM diagnostic ==="
srun --het-group=0 -n 1 ./mpi_test : --het-group=1 -n 1 ./mpi_test
