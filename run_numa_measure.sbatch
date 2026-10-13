#!/bin/bash
#SBATCH -J numa_measure
#SBATCH --account=thes2181
#SBATCH --partition=c23mm
#SBATCH -N 1
#SBATCH --ntasks-per-node=96
#SBATCH --mem=238G
#SBATCH --time=00:30:00
#SBATCH -o logs/numa_measure_%j.out

# Measure tiny-message rank->controller latency and payload NUMA->controller
# incast on one node, then solve the NUMA-level workgroup assignment.
set -euo pipefail

ROOT="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim"
cd "${ROOT}"
source ./set_env_claix23_cuda12.4.sh

PY="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
OUT="mini_app/results/aix_layout_matrix/numa_measure_${SLURM_JOB_ID}"
export TMPDIR=/tmp

echo "host=$(hostname) job=${SLURM_JOB_ID}"
for order in fixed rotate shuffle; do
  echo "=== order=${order} ==="
  mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
    --map-by core --bind-to core -np 96 \
    "${PY}" mini_app/numa_measure.py \
    --controllers 0,24,48,72 --payload 6220800 --reps 20 --warmup 3 --numas 8 \
    --order "${order}" --out "${OUT}/measure_${order}.json"
done

"${PY}" mini_app/numa_assign_numa.py --measure-json "${OUT}/measure_rotate.json" \
  --out-dir "${OUT}/assign_rotate"
echo "done"
