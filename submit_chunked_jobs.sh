#!/bin/zsh
# Script to submit chunked mini_app jobs with increasing total steps
# Breaks 1,000,000 steps into 20 jobs of 50,000 steps each
# Rendering is disabled for the first 19 jobs and enabled for the last job

set -euo pipefail

# Configuration
TOTAL_FINAL_STEPS=1000000
CHUNK_SIZE=50000
NUM_JOBS=$((TOTAL_FINAL_STEPS / CHUNK_SIZE))
SBATCH_SCRIPT="/hpcwork/ro092286/smartsim/mini_app/slurm_sbatch_mini_app.sh"

# Validate that SBATCH script exists
if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "Error: SBATCH script not found at ${SBATCH_SCRIPT}"
  exit 1
fi

echo "Submitting ${NUM_JOBS} chunked jobs..."
echo "Each job runs ${CHUNK_SIZE} additional steps"
echo "Final total: ${TOTAL_FINAL_STEPS} steps"
echo ""

PREVIOUS_JOB_ID=""

for i in $(seq 1 ${NUM_JOBS}); do
  CURRENT_STEPS=$((i * CHUNK_SIZE))
  JOB_NUMBER=$i
  
  # For the last job, enable rendering; for others, disable it
  if [[ $i -eq ${NUM_JOBS} ]]; then
    SKIP_RENDERING=0
    RENDER_STATUS="enabled"
    SKIP_COMPILE=1
    COMPILE_STATUS="skipped"
  elif [[ $i -eq 1 ]]; then
    SKIP_RENDERING=1
    RENDER_STATUS="disabled"
    SKIP_COMPILE=0
    COMPILE_STATUS="normal"
  else
    SKIP_RENDERING=1
    RENDER_STATUS="disabled"
    SKIP_COMPILE=1
    COMPILE_STATUS="skipped"
  fi
  
  echo "Job ${JOB_NUMBER}/${NUM_JOBS}: total_steps=${CURRENT_STEPS}, rendering=${RENDER_STATUS}, compile=${COMPILE_STATUS}"
  
  # Build sbatch command with environment variable overrides
  SBATCH_CMD="sbatch --export=TOTAL_STEPS_ENV=${CURRENT_STEPS},SKIP_RENDERING_ENV=${SKIP_RENDERING},SKIP_COMPILE_ENV=${SKIP_COMPILE}"
  
  # Add job dependency if not the first job
  if [[ -n "${PREVIOUS_JOB_ID}" ]]; then
    SBATCH_CMD="${SBATCH_CMD} --dependency=afterok:${PREVIOUS_JOB_ID}"
  fi
  
  # Submit with environment variables
  JOB_ID=$(${SBATCH_CMD} "${SBATCH_SCRIPT}" 2>&1 | grep -oP 'Submitted batch job \K\d+')
  
  if [[ -z "${JOB_ID}" ]]; then
    echo "Error: Failed to submit job ${JOB_NUMBER}"
    exit 1
  fi
  
  echo "  → Submitted with job ID: ${JOB_ID}"
  PREVIOUS_JOB_ID="${JOB_ID}"
done

echo ""
echo "All ${NUM_JOBS} jobs submitted successfully!"
echo "First job ID: (see above)"
echo "Last job ID: ${PREVIOUS_JOB_ID}"
echo ""
echo "Jobs are configured to run sequentially with dependencies."
echo "Use 'squeue' to monitor job status."
