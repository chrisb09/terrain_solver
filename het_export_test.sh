#!/bin/bash
#SBATCH --job-name=het_export_test
#SBATCH --account=thes2181
#SBATCH --time=00:02:00
#SBATCH --output=logs/het_export_test_%j.txt
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500M
#SBATCH hetjob

#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=500M

set -u

echo "=== Job env diagnostics ==="
echo "SLURM_HET_SIZE=${SLURM_HET_SIZE:-<unset>}"
echo "SLURM_JOB_NUM_NODES_HET_GROUP_0=${SLURM_JOB_NUM_NODES_HET_GROUP_0:-<unset>}"
echo "SLURM_JOB_NUM_NODES_HET_GROUP_1=${SLURM_JOB_NUM_NODES_HET_GROUP_1:-<unset>}"

# Set a污染 global LD_LIBRARY_PATH (simulating the Score-P stubs prepend)
export LD_LIBRARY_PATH="GLOBAL_STUB_VALUE${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "Global LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

echo
echo "=== Test 1: baseline --export=ALL on both components (no override) ==="
srun --export=ALL bash -c 'echo comp0 LDLP=$LD_LIBRARY_PATH' \
  : --export=ALL bash -c 'echo comp1 LDLP=$LD_LIBRARY_PATH'

echo
echo "=== Test 2: override LD_LIBRARY_PATH on comp1 only ==="
srun --export=ALL bash -c 'echo comp0 LDLP=$LD_LIBRARY_PATH' \
  : --export=ALL,LD_LIBRARY_PATH=CLEAN_OVERRIDE bash -c 'echo comp1 LDLP=$LD_LIBRARY_PATH'

echo
echo "=== Test 3: override LD_LIBRARY_PATH on both components (different values) ==="
srun --export=ALL,LD_LIBRARY_PATH=COMP0_OVERRIDE bash -c 'echo comp0 LDLP=$LD_LIBRARY_PATH' \
  : --export=ALL,LD_LIBRARY_PATH=COMP1_OVERRIDE bash -c 'echo comp1 LDLP=$LD_LIBRARY_PATH'

echo
echo "=== Test 4: explicit het-group flags on both components ==="
srun --het-group=0 --export=ALL bash -c 'echo comp0 LDLP=$LD_LIBRARY_PATH' \
  : --het-group=1 --export=ALL,LD_LIBRARY_PATH=CLEAN_OVERRIDE bash -c 'echo comp1 LDLP=$LD_LIBRARY_PATH'

echo
echo "=== Test 5: --export=ALL on comp0, override on comp1 with explicit het-group on comp1 only ==="
srun --export=ALL bash -c 'echo comp0 LDLP=$LD_LIBRARY_PATH' \
  : --het-group=1 --export=ALL,LD_LIBRARY_PATH=CLEAN_OVERRIDE bash -c 'echo comp1 LDLP=$LD_LIBRARY_PATH'

echo
echo "=== Test 6: full env export check (PATH override) ==="
srun --export=ALL,PATH=/special/path0 bash -c 'echo comp0 PATH=$PATH' \
  : --export=ALL,PATH=/special/path1 bash -c 'echo comp1 PATH=$PATH'

echo
echo "=== Done ==="