#!/bin/zsh
# HH3 H-class SURROGATE wrapper: het allocation of TWO c23g nodes under
# rwth0792 (c23mm stand-in while thes2181 is out of CPU quota). Topology is
# identical to c23mm (96 cores); the CPU-side component's GPUs are hidden from
# the solver via CUDA_VISIBLE_DEVICES=NoDevFiles so AIX/smartredis discovery
# behaves exactly like on a c23mm node.
#   s5 = 24+24 (1 GPU)   s7 = 96+96 (4 GPUs)
#SBATCH --job-name=HH3_H_sur
#SBATCH --account=rwth0792
#SBATCH --time=00:15:00
#SBATCH --output=logs/hh3_%j.txt
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=5G
#SBATCH --exclusive

#SBATCH hetjob

#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=5G
#SBATCH --exclusive

set -uo pipefail

CONFIG_TAG="${HH3_CONFIG_ENV:?HH3_CONFIG_ENV required}"
LAYOUT_TAG="${HH3_LAYOUT_ENV:-default}"
SUITE="${HH3_SUITE_ENV:-hethires3}"

MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
cd "${MINI_APP_DIR}"
mkdir -p logs

SUMMARY="logs/hh3_summary_${SLURM_JOB_ID}.txt"
echo "HH3 H-class SURROGATE config=${CONFIG_TAG} layout=${LAYOUT_TAG} scenarios=s5,s7 start=$(date)" > "${SUMMARY}"

want_scenario() {
    [[ -z "${HH3_SCENARIOS_ENV:-}" ]] || [[ ",${HH3_SCENARIOS_ENV}," == *",${1},"* ]]
}

run_scenario() {
    local sid="$1" mm="$2" gside_b="$3" ngpus="$4"
    local gside
    if [[ "${CONFIG_TAG}" == aix* ]]; then
        gside=$(( gside_b > 0 ? gside_b : 1 ))
    else
        gside="${gside_b}"
    fi
    export RUN_ID_ENV="${SUITE}_s${sid}_${CONFIG_TAG}_${LAYOUT_TAG}"
    export JOB_NAME_ENV="${SUITE}_s${sid}_${CONFIG_TAG}_${LAYOUT_TAG}"
    export SCOREP_DIR_TAG_ENV="${SUITE}_s${sid}_${CONFIG_TAG}_${LAYOUT_TAG}"
    export MPI_RANKS_ENV="${mm}"
    export HET_G_SIDE_TASKS_ENV="${gside}"
    export GPUS_PER_NODE_ENV="${ngpus}"
    export SMARTSIM_PORT_OFFSET_ENV="${sid}"
    export PHYDLL_NP_DL_ENV="${ngpus}"
    # Signal to the engine (NOT the solver env): hide GPUs on the CPU-side
    # het-group. The engine injects HET_SURROGATE_HIDE_GPU into that group's
    # srun --export only, and solver_wrapper.sh applies NoDevFiles there.
    export HH3_SURROGATE=1
    unset HET_SURROGATE_HIDE_GPU

    if [[ "${CONFIG_TAG}" == aix* ]]; then
        export COMPILE_OUTPUT_PATH_ENV="build_scorep_cmi_${LAYOUT_TAG}"
    else
        export COMPILE_OUTPUT_PATH_ENV="build_scorep_cmi_default"
    fi

    if [[ "${ngpus}" -le 1 ]]; then
        export SOLVER_STEP_GPUS_ENV="1"
        export CUDA_VISIBLE_DEVICES="0"
    else
        unset SOLVER_STEP_GPUS_ENV
        unset CUDA_VISIBLE_DEVICES
    fi

    if [[ "${CONFIG_TAG}" == phydll* && "${gside}" -gt 0 ]]; then
        export PHYDLL_DL_OVERSUBSCRIBE_ENV="1"
    else
        unset PHYDLL_DL_OVERSUBSCRIBE_ENV
    fi

    echo "=== HH3 s${sid}_${CONFIG_TAG}_${LAYOUT_TAG} (mm=${mm}, gside=${gside}, gpus=${ngpus}, build=${COMPILE_OUTPUT_PATH_ENV}, surrogate=1) start $(date) ===" >> "${SUMMARY}"
    zsh "${MINI_APP_DIR}/het_hires_job.sh" >> "${SUMMARY}" 2>&1
    local rc=$?
    echo "=== HH3 s${sid}_${CONFIG_TAG}_${LAYOUT_TAG} finished rc=${rc} $(date) ===" >> "${SUMMARY}"
    return 0
}

want_scenario 5 && run_scenario 5 24 24 1
want_scenario 7 && run_scenario 7 96 96 4

echo "HH3 H surrogate wrapper done $(date): $(grep 'finished' "${SUMMARY}" | tr '\n' ' ')" >> "${SUMMARY}"
