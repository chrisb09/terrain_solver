#!/bin/bash
# Third launch probe: validate the EXACT interface the driver will use --
# generated command-scripts + dispatch_wrapper.sh with script paths, including
# quoted arguments (to confirm quoting survives).
#
# Topology:
#   het-group 0: 4 tasks  -> solver script (quoted arg)
#   het-group 1: 5 tasks  -> wrapper: local 0-3 solver, local 4 dl
#   expected world size = 9
#SBATCH --job-name=mpi_probe3
#SBATCH --account=rwth0792
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --output=logs/mpi_probe3_%j.txt

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

# Generate the exact per-rank command scripts, with a quoted argument.
SOLVER_SH="${TEST_DIR}/logs/_probe_solver.sh"
DL_SH="${TEST_DIR}/logs/_probe_dl.sh"
cat > "${SOLVER_SH}" <<EOF
#!/bin/bash
exec ${PROBE} "solver role" --tag "a b c"
EOF
cat > "${DL_SH}" <<EOF
#!/bin/bash
exec ${PROBE} "dl role"
EOF
chmod +x "${SOLVER_SH}" "${DL_SH}"

echo "=== mpi_probe3 job ${SLURM_JOB_ID} ==="
echo "het-group 0 nodes: ${SLURM_JOB_NODELIST_HET_GROUP_0}"
echo "het-group 1 nodes: ${SLURM_JOB_NODELIST_HET_GROUP_1}"

echo
echo "########## TEST G: srun MPMD + wrapper with script paths + quoted args ##########"
srun --wait=30 --het-group=0 --nodes=1 --ntasks=4 --cpus-per-task=1 \
    "${SOLVER_SH}" : \
  --het-group=1 --nodes=1 --ntasks=5 --cpus-per-task=1 --overcommit \
    "${WRAPPER}" 4 "${SOLVER_SH}" "${DL_SH}"
echo "TEST_G_RC=$?"

echo
echo "=== mpi_probe3 job ${SLURM_JOB_ID} finished ==="
