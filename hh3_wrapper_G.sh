#!/bin/zsh
# HH3 G-class wrapper: c23g-only single allocation (96 x 1 CPU + 4 GPUs,
# exclusive). Runs the A=0 scenarios sequentially for ONE library config:
#   s1 = 0+4  (1 GPU)   s3 = 0+24 (1 GPU)   s6 = 0+96 (4 GPUs)
# The library config (provider exports) is supplied at submission via
# --export; HH3_CONFIG_ENV names it, HH3_LAYOUT_ENV names the layout.
# Each scenario invokes single_node_96c_4gpu.sh as a child process with
# per-scenario env (unique RUN_ID/JOB_NAME/Score-P tag, port offset).
#SBATCH --job-name=HH3_G
#SBATCH --account=rwth0792
#SBATCH --time=00:12:00
#SBATCH --output=logs/hh3_%j.txt
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive

set -uo pipefail

CONFIG_TAG="${HH3_CONFIG_ENV:?HH3_CONFIG_ENV required}"
LAYOUT_TAG="${HH3_LAYOUT_ENV:-default}"
SUITE="${HH3_SUITE_ENV:-hethires3}"
MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
cd "${MINI_APP_DIR}"
mkdir -p logs

SUMMARY="logs/hh3_summary_${SLURM_JOB_ID}.txt"
echo "HH3 G-class config=${CONFIG_TAG} layout=${LAYOUT_TAG} scenarios=s1,s3,s6 start=$(date)" > "${SUMMARY}"

want_scenario() {
    [[ -z "${HH3_SCENARIOS_ENV:-}" ]] || [[ ",${HH3_SCENARIOS_ENV}," == *",${1},"* ]]
}

run_scenario() {
    local sid="$1" ranks="$2" ngpus="$3"
    export RUN_ID_ENV="${SUITE}_s${sid}_${CONFIG_TAG}_${LAYOUT_TAG}"
    export JOB_NAME_ENV="${SUITE}_s${sid}_${CONFIG_TAG}_${LAYOUT_TAG}"
    export SCOREP_DIR_TAG_ENV="${SUITE}_s${sid}_${CONFIG_TAG}_${LAYOUT_TAG}"
    export MPI_RANKS_ENV="${ranks}"
    export GPUS_PER_NODE_ENV="${ngpus}"
    export SMARTSIM_PORT_OFFSET_ENV="${sid}"
    
    # Select solver variant: AIX uses layout-specific build; others use default
    if [[ "${CONFIG_TAG}" == aix* ]]; then
        export COMPILE_OUTPUT_PATH_ENV="build_scorep_cmi_${LAYOUT_TAG}"
    else
        export COMPILE_OUTPUT_PATH_ENV="build_scorep_cmi_default"
    fi

    if [[ "${ngpus}" -le 1 ]]; then
        export CUDA_VISIBLE_DEVICES="0"
        export SOLVER_STEP_GPUS_ENV="1"
    else
        unset CUDA_VISIBLE_DEVICES
        unset SOLVER_STEP_GPUS_ENV
    fi
    echo "=== HH3 s${sid}_${CONFIG_TAG}_${LAYOUT_TAG} (ranks=${ranks}, gpus=${ngpus}, build=${COMPILE_OUTPUT_PATH_ENV}) start $(date) ===" >> "${SUMMARY}"
    zsh "${MINI_APP_DIR}/single_node_96c_4gpu.sh" >> "${SUMMARY}" 2>&1
    local rc=$?
    echo "=== HH3 s${sid}_${CONFIG_TAG}_${LAYOUT_TAG} finished rc=${rc} $(date) ===" >> "${SUMMARY}"
    return 0   # continue with the next scenario regardless
}

want_scenario 1 && run_scenario 1 4 1
want_scenario 3 && run_scenario 3 24 1
want_scenario 6 && run_scenario 6 96 4

echo "HH3 G wrapper done $(date): $(grep 'finished' "${SUMMARY}" | tr '\n' ' ')" >> "${SUMMARY}"
