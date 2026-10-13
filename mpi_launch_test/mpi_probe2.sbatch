#!/bin/bash
# Second launch probe: verify that a single srun component on het-group 1 can
# host both the GPU-side solver ranks and the DL-client rank via a
# rank-dispatching wrapper, while sharing ONE MPI world with the CPU-side
# solver ranks on het-group 0.
#
# Topology (small-scale 24p24c1g analogue):
#   het-group 0: 4 tasks            -> probe (cpu_solver)
#   het-group 1: 5 tasks (4+1 over) -> wrapper: local 0-3 solver, local 4 dl
#   expected world size = 9, one communicator
#SBATCH --job-name=mpi_probe2
#SBATCH --account=rwth0792
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --output=logs/mpi_probe2_%j.txt

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
TEST_DIR="${MINI_APP_DIR}/mpi_launch_test"
cd "${TEST_DIR}" || exit 1
mkdir -p logs

PROBE="${TEST_DIR}/mpi_world_probe"
WRAPPER="${TEST_DIR}/dispatch_wrapper.sh"
chmod +x "${WRAPPER}"

echo "=== mpi_probe2 job ${SLURM_JOB_ID} ==="
echo "het-group 0 nodes: ${SLURM_JOB_NODELIST_HET_GROUP_0}"
echo "het-group 1 nodes: ${SLURM_JOB_NODELIST_HET_GROUP_1}"
echo "ntasks-per-node grp0=${SLURM_NTASKS_PER_NODE_HET_GROUP_0} grp1=${SLURM_NTASKS_PER_NODE_HET_GROUP_1}"

export DISPATCH_N_SOLVER=4
export DISPATCH_SOLVER_CMD="${PROBE} gpu_solver"
export DISPATCH_DL_CMD="${PROBE} dl"

echo
echo "########## TEST E: srun MPMD, het-group1 = wrapper(4 solver + 1 DL) ##########"
srun --wait=30 --het-group=0 --nodes=1 --ntasks=4 --cpus-per-task=1 \
    "${PROBE}" cpu_solver : \
  --het-group=1 --nodes=1 --ntasks=5 --cpus-per-task=1 --overcommit \
    "${WRAPPER}" 4 "${PROBE} gpu_solver" "${PROBE} dl"
echo "TEST_E_RC=$?"

echo
echo "########## TEST F: same but 2 DL ranks (local 0-3 solver, 4-5 dl) ##########"
export DISPATCH_N_SOLVER=4
srun --wait=30 --het-group=0 --nodes=1 --ntasks=4 --cpus-per-task=1 \
    "${PROBE}" cpu_solver : \
  --het-group=1 --nodes=1 --ntasks=6 --cpus-per-task=1 --overcommit \
    "${WRAPPER}" 4 "${PROBE} gpu_solver" "${PROBE} dl"
echo "TEST_F_RC=$?"

echo
echo "=== mpi_probe2 job ${SLURM_JOB_ID} finished ==="
