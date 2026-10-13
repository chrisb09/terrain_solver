#!/bin/bash
# Rank-dispatching wrapper for a single srun component that must host BOTH
# GPU-side solver ranks and PhyDLL DL-client ranks on the same het group.
#
# Slurm cannot reference one het group twice in a single srun MPMD, and
# --multi-prog is not allowed with multiple het groups. This wrapper is the
# executable of the GPU-side component and dispatches each task to the solver
# or the DL client by its LOCAL rank within the component.
#
# Usage:
#   dispatch_wrapper.sh N_SOLVER SOLVER_SCRIPT DL_SCRIPT
#
# SOLVER_SCRIPT / DL_SCRIPT are executable scripts holding the exact command
# (generated per run by the driver, so quoting is preserved verbatim).
set -uo pipefail

LOCAL_ID="${SLURM_LOCALID:-0}"
N_SOLVER="${1:?N_SOLVER required}"
SOLVER_SCRIPT="${2:?SOLVER_SCRIPT required}"
DL_SCRIPT="${3:?DL_SCRIPT required}"

if [[ "${LOCAL_ID}" -lt "${N_SOLVER}" ]]; then
    echo "[DISPATCH] local=${LOCAL_ID} -> solver (${SOLVER_SCRIPT})"
    exec "${SOLVER_SCRIPT}"
else
    echo "[DISPATCH] local=${LOCAL_ID} -> dl (${DL_SCRIPT})"
    exec "${DL_SCRIPT}"
fi
