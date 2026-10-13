#!/bin/zsh
# HH2 G-class wrapper: c23g-only single allocation (96 x 1 CPU + 4 GPUs,
# exclusive). Runs the A=0 scenarios sequentially for ONE library config:
#   s1 = 0+4  (1 GPU)   s3 = 0+24 (1 GPU)   s6 = 0+96 (4 GPUs)
# The library config (provider exports) is supplied at submission via
# --export; HH2_CONFIG_ENV names it (used for tags and log paths).
# Each scenario invokes single_node_96c_4gpu.sh as a child process with
# per-scenario env (unique RUN_ID/JOB_NAME/Score-P tag, port offset).
#SBATCH --job-name=HH2_G
#SBATCH --account=thes2181
#SBATCH --time=02:00:00
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --output=logs/hh2_%j.txt

set -uo pipefail

CONFIG_TAG="${HH2_CONFIG_ENV:?HH2_CONFIG_ENV required}"
MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
cd "${MINI_APP_DIR}"
mkdir -p logs

SUMMARY="logs/hh2_summary_${SLURM_JOB_ID}.txt"
echo "HH2 G-class config=${CONFIG_TAG} scenarios=s1,s3,s6 start=$(date)" > "${SUMMARY}"

run_scenario() {
    local sid="$1" ranks="$2" ngpus="$3"
    export RUN_ID_ENV="hethires2_s${sid}_${CONFIG_TAG}"
    export JOB_NAME_ENV="hethires2_s${sid}_${CONFIG_TAG}"
    export SCOREP_DIR_TAG_ENV="hethires2_s${sid}_${CONFIG_TAG}"
    export MPI_RANKS_ENV="${ranks}"
    export GPUS_PER_NODE_ENV="${ngpus}"
    export SMARTSIM_PORT_OFFSET_ENV="${sid}"
    if [[ "${ngpus}" -le 1 ]]; then
        export CUDA_VISIBLE_DEVICES="0"
        export SOLVER_STEP_GPUS_ENV="1"
    else
        unset CUDA_VISIBLE_DEVICES
        unset SOLVER_STEP_GPUS_ENV
    fi
    echo "=== HH2 s${sid} (ranks=${ranks}, gpus=${ngpus}) start $(date) ===" >> "${SUMMARY}"
    zsh "${MINI_APP_DIR}/single_node_96c_4gpu.sh" >> "${SUMMARY}" 2>&1
    local rc=$?
    echo "=== HH2 s${sid}_${CONFIG_TAG} finished rc=${rc} $(date) ===" >> "${SUMMARY}"
    return 0   # continue with the next scenario regardless
}

# Optional scenario filter: HH2_SCENARIOS_ENV="1,3,6" runs only those sX ids.
want_scenario() {
    [[ -z "${HH2_SCENARIOS_ENV:-}" ]] || [[ ",${HH2_SCENARIOS_ENV}," == *",${1},"* ]]
}

want_scenario 1 && run_scenario 1 4 1
want_scenario 3 && run_scenario 3 24 1
want_scenario 6 && run_scenario 6 96 4

echo "HH2 G wrapper done $(date): $(grep 'finished' "${SUMMARY}" | tr '\n' ' ')" >> "${SUMMARY}"
