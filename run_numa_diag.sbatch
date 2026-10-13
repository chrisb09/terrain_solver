#!/bin/bash
#SBATCH -J numa_diag
#SBATCH --account=thes2181
#SBATCH --partition=c23mm
#SBATCH -N 1
#SBATCH --ntasks-per-node=96
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH -o logs/numa_diag_%j.out

# Decide whether the anomalous NUMA0->controller-0 incast follows rank 0 or
# NUMA 0 itself: measure with controllers on the GPU NUMAs but shifted off
# rank 0 (1,25,49,73 vs 0,24,48,72), rotate order, many repeats.
set -euo pipefail

ROOT="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim"
cd "${ROOT}"
source ./set_env_claix23_cuda12.4.sh

PY="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
OUT="mini_app/results/aix_layout_matrix/numa_diag_${SLURM_JOB_ID}"
export TMPDIR=/tmp

echo "host=$(hostname) job=${SLURM_JOB_ID}"

for cset in "0,24,48,72" "1,25,49,73"; do
  tag=$(echo "${cset}" | tr ',' '_')
  echo "=== controllers=${cset} ==="
  mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
    --map-by core --bind-to core -np 96 \
    "${PY}" mini_app/numa_measure.py \
    --controllers "${cset}" --payload 6220800 --reps 30 --warmup 5 --numas 8 \
    --order rotate --out "${OUT}/measure_${tag}.json"
done
echo "done"
