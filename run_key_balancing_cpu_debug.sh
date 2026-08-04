#!/usr/bin/env zsh

# Execute the regular launcher without exposing its embedded heterogeneous
# #SBATCH directives to a single-allocation CPU debug submission.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${0:A:h}}"
exec /bin/zsh "${SCRIPT_DIR}/proper_slurm_job.sh"
