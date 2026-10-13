#!/bin/bash
# Two-node PhyDLL launch-mechanism probe.
#
# Mirrors the 24p24c1g topology at small scale:
#   het-group 0: CPU-side node, NO gres, 4 solver tasks
#   het-group 1: GPU-side node, gres=gpu:1, 4 slots for solver + 1 extra DL task
#
# Runs several launch mechanisms in sequence and logs which ones produce a
# single shared MPI world across both nodes. Nothing here touches the solver.
#SBATCH --job-name=mpi_probe
#SBATCH --account=rwth0792
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --output=logs/mpi_probe_%j.txt

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0

#SBATCH hetjob

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=0

set -uo pipefail
MINI_APP_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app"
cd "${MINI_APP_DIR}/mpi_launch_test" || exit 1
mkdir -p ../logs

PROBE="${MINI_APP_DIR}/mpi_launch_test/mpi_world_probe"

CPU_NODES="$(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_0}" | paste -sd, -)"
GPU_NODES="$(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_1}" | paste -sd, -)"
CPU_HOST="$(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_0}" | head -1)"
GPU_HOST="$(scontrol show hostnames "${SLURM_JOB_NODELIST_HET_GROUP_1}" | head -1)"

echo "=== mpi_probe job ${SLURM_JOB_ID} ==="
echo "CPU_NODES=${CPU_NODES}  GPU_NODES=${GPU_NODES}"
echo "CPU_HOST=${CPU_HOST}  GPU_HOST=${GPU_HOST}"
echo "SLURM_HET_SIZE=${SLURM_HET_SIZE:-unset}"
echo "SLURM_NTASKS_PER_NODE_HET_GROUP_0=${SLURM_NTASKS_PER_NODE_HET_GROUP_0:-unset}"
echo "SLURM_NTASKS_PER_NODE_HET_GROUP_1=${SLURM_NTASKS_PER_NODE_HET_GROUP_1:-unset}"
echo "plain SLURM_NODELIST=${SLURM_NODELIST:-unset}"
echo "plain SLURM_NTASKS=${SLURM_NTASKS:-unset}"

banner() { echo; echo "########## $* ##########"; }

# ---------------------------------------------------------------------------
banner "TEST A: mpirun MPMD with hostfile (current failing approach)"
HF="$(mktemp /tmp/mpi_probe_hostfile.XXXXXX)"
printf '%s slots=%d\n' "${CPU_HOST}" 4 >  "${HF}"
printf '%s slots=%d\n' "${GPU_HOST}" 5 >> "${HF}"
echo "hostfile:"; cat "${HF}"
timeout 120 mpirun --bind-to none --oversubscribe --hostfile "${HF}" \
  -np 4 "${PROBE}" solver : \
  -np 5 -host "${GPU_HOST}:5" "${PROBE}" dl
echo "TEST_A_RC=$?"
rm -f "${HF}"

# ---------------------------------------------------------------------------
banner "TEST B: srun MPMD, het-group 1 referenced twice (solver + DL)"
timeout 120 srun --wait=30 --het-group=0 --nodes=1 --ntasks=4 --cpus-per-task=1 \
    "${PROBE}" solver : \
  --het-group=1 --nodes=1 --ntasks=4 --cpus-per-task=1 --oversubscribe \
    "${PROBE}" gpu_solver : \
  --het-group=1 --nodes=1 --ntasks=1 --cpus-per-task=1 --oversubscribe \
    "${PROBE}" dl
echo "TEST_B_RC=$?"

# ---------------------------------------------------------------------------
banner "TEST C: srun MPMD with --multi-prog on het-group 1 (single component)"
MP="$(mktemp /tmp/mpi_probe_multiprog.XXXXXX)"
cat > "${MP}" <<EOF
0-3 ${PROBE} gpu_solver
4   ${PROBE} dl
EOF
echo "multi-prog config:"; cat "${MP}"
timeout 120 srun --wait=30 --het-group=0 --nodes=1 --ntasks=4 --cpus-per-task=1 \
    "${PROBE}" cpu_solver : \
  --het-group=1 --nodes=1 --ntasks=5 --cpus-per-task=1 --oversubscribe \
    --multi-prog "${MP}"
echo "TEST_C_RC=$?"
rm -f "${MP}"

# ---------------------------------------------------------------------------
banner "TEST D: srun MPMD, each het-group once (no DL oversubscription)"
timeout 120 srun --wait=30 --het-group=0 --nodes=1 --ntasks=4 --cpus-per-task=1 \
    "${PROBE}" cpu_solver : \
  --het-group=1 --nodes=1 --ntasks=4 --cpus-per-task=1 \
    "${PROBE}" gpu_solver
echo "TEST_D_RC=$?"

banner "DONE"
echo "=== mpi_probe job ${SLURM_JOB_ID} finished ==="
