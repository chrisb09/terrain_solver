#!/bin/bash
#SBATCH -J numa_proto
#SBATCH -N 1
#SBATCH --ntasks-per-node=96
#SBATCH --mem=238G
#SBATCH --time=00:30:00
#SBATCH -p devel
#SBATCH -o logs/numa_proto_%j.out

# Measure rank-to-rank transfer cost on a devel node (same Sapphire Rapids
# architecture as c23, no GPUs) and run the per-rank min-cost-flow assignment
# prototype against the measured matrix.
set -euo pipefail

ROOT="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim"
cd "${ROOT}"
source ./set_env_claix23_cuda12.4.sh

PYTHON_BIN="/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3"
PAYLOAD=6220800
CSV="affinity_test/numa_proto_96ranks.csv"
OUT_DIR="mini_app/results/aix_layout_matrix/numa_prototype_devel_${SLURM_JOB_ID}"

echo "host=$(hostname) job=${SLURM_JOB_ID}"
echo "=== measuring rank-to-rank transfer cost (payload=${PAYLOAD} B) ==="
cd affinity_test
mpirun -mca shmem_mmap_enable_nfs_warning 0 -mca orte_tmpdir_base /tmp \
  --map-by core --bind-to core \
  -np 96 ./build/mpi_pair_latency \
  -o numa_proto_96ranks.csv -s 64 -l "${PAYLOAD}" -w 5 -i 10
cd "${ROOT}"

echo "=== solving per-rank assignment (controllers on cores 0,24,48,72) ==="
"${PYTHON_BIN}" mini_app/numa_assign_prototype.py \
  --csv "${CSV}" \
  --controllers 0,24,48,72 \
  --payload "${PAYLOAD}" \
  --out-dir "${OUT_DIR}"

numactl --hardware | head -20 || true
echo "done"
