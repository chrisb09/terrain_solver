#!/bin/zsh

# Ensure logs dir exists
mkdir -p logs

cat << 'EOF' > diag_devel.sh
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
EOF

cat << 'EOF' > diag_c23mm.sh
#!/bin/zsh
#SBATCH --job-name=diag_c23mm
#SBATCH --account=thes2181
#SBATCH --partition=c23mm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/diag_c23mm_%j.txt

module load GCCcore/11.3.0 OpenMPI/4.1.4
echo "=== Running C23MM diagnostic ==="
srun ./mpi_test
EOF

cat << 'EOF' > diag_c23ms_c23mm.sh
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
EOF

cat << 'EOF' > diag_c23mm_c23g.sh
#!/bin/zsh
#SBATCH --job-name=diag_c23mm_c23g
#SBATCH --account=thes2181
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH hetjob
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/diag_c23mm_c23g_%j.txt

module load GCCcore/11.3.0 OpenMPI/4.1.4
echo "=== Running C23MM + C23G diagnostic ==="
srun --het-group=0 -n 1 ./mpi_test : --het-group=1 -n 1 ./mpi_test
EOF

chmod +x diag_*.sh

echo "Submitting DIAG_DEVEL..."
sbatch diag_devel.sh
echo "Submitting DIAG_C23MM..."
sbatch diag_c23mm.sh
echo "Submitting DIAG_C23MS_C23MM..."
sbatch diag_c23ms_c23mm.sh
echo "Submitting DIAG_C23MM_C23G..."
sbatch diag_c23mm_c23g.sh
