#!/bin/bash
#SBATCH --job-name=het_srun_dbg
#SBATCH --account=thes2181
#SBATCH --time=00:05:00
#SBATCH --output=logs/het_srun_debug_%j.txt
#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500M

#SBATCH hetjob

#SBATCH --partition=c23mm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500M

set -uo pipefail

echo "=== Allocation ==="
echo "SLURM_HET_SIZE=${SLURM_HET_SIZE}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_JOB_NUM_NODES_HET_GROUP_0=${SLURM_JOB_NUM_NODES_HET_GROUP_0}"
echo "SLURM_JOB_NUM_NODES_HET_GROUP_1=${SLURM_JOB_NUM_NODES_HET_GROUP_1}"
echo "SLURM_TRES_PER_TASK_HET_GROUP_0=${SLURM_TRES_PER_TASK_HET_GROUP_0}"
echo "SLURM_TRES_PER_TASK_HET_GROUP_1=${SLURM_TRES_PER_TASK_HET_GROUP_1}"

# Load a minimal environment (no Score-P) to isolate the issue
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

echo "=== Test 1: srun with --het-group=1 (simulating SmartSim DB launch) ==="
srun --output=/tmp/srun_het1.out --error=/tmp/srun_het1.err --job-name=orchestrator_0 \
  --export=ALL --het-group=1 --cpus-per-task=1 hostname 2>&1
echo "srun exited with code $?"
echo "stdout file: $(cat /tmp/srun_het1.out 2>/dev/null || echo 'EMPTY')"
echo "stderr file: $(cat /tmp/srun_het1.err 2>/dev/null || echo 'EMPTY')"

echo "=== Test 2: srun with --het-group=0 (control) ==="
srun --output=/tmp/srun_het0.out --error=/tmp/srun_het0.err --job-name=control_0 \
  --export=ALL --het-group=0 --cpus-per-task=1 hostname 2>&1
echo "srun exited with code $?"
echo "stdout file: $(cat /tmp/srun_het0.out 2>/dev/null || echo 'EMPTY')"
echo "stderr file: $(cat /tmp/srun_het0.err 2>/dev/null || echo 'EMPTY')"

echo "=== Test 3: srun with --jobid (SmartSim style) on het-group=1 ==="
srun --output=/tmp/srun_jobid1.out --error=/tmp/srun_jobid1.err --job-name=jobid_test_1 \
  --jobid="${SLURM_JOB_ID}" --export=ALL --het-group=1 --cpus-per-task=1 hostname 2>&1
echo "srun exited with code $?"
echo "stdout file: $(cat /tmp/srun_jobid1.out 2>/dev/null || echo 'EMPTY')"
echo "stderr file: $(cat /tmp/srun_jobid1.err 2>/dev/null || echo 'EMPTY')"

echo "=== Test 4: sacct check for the steps ==="
sacct --noheader -p --format=jobname,jobid,state,exitcode 2>/dev/null

echo "=== Test 5: sacct -j SLURM_JOB_ID ==="
sacct -j "${SLURM_JOB_ID}" --format=jobname,jobid,state,exitcode 2>/dev/null

echo "=== Done ==="