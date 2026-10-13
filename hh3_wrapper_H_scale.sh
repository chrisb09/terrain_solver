#!/bin/zsh
# HH3 H-class Scale wrapper: bundled multi-node het allocation
# (16 c23mm nodes [1536 cores] + 1 c23g node [96 cores, 4 GPUs], exclusive).
# Runs the scaling scenarios sequentially inside ONE 17-node allocation:
#   s8  =  192 mm ( 2 nodes) + 96 g (1 node, 4 GPUs) =  288 ranks ( 2:1)
#   s9  =  384 mm ( 4 nodes) + 96 g (1 node, 4 GPUs) =  480 ranks ( 4:1)
#   s10 =  768 mm ( 8 nodes) + 96 g (1 node, 4 GPUs) =  864 ranks ( 8:1)
#   s11 = 1536 mm (16 nodes) + 96 g (1 node, 4 GPUs) = 1632 ranks (16:1)
#
# Config and layout are passed at submission via --export.
#SBATCH --job-name=HH3_H_scale
#SBATCH --account=thes2181
#SBATCH --output=logs/hh3_%j.txt
#SBATCH --partition=c23mm
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
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
MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
cd "${MINI_APP_DIR}"
mkdir -p logs

SUMMARY="logs/hh3_summary_${SLURM_JOB_ID}.txt"
echo "HH3 H-scale config=${CONFIG_TAG} layout=${LAYOUT_TAG} scenarios=s8,s9,s10,s11 start=$(date)" > "${SUMMARY}"

want_scenario() {
    [[ -z "${HH3_SCENARIOS_ENV:-}" ]] || [[ ",${HH3_SCENARIOS_ENV}," == *",${1},"* ]]
}

run_scenario() {
    local sid="$1" mm="$2" mm_nodes="$3" gside_b="$4" ngpus="$5"
    local gside
    if [[ "${CONFIG_TAG}" == aix* ]]; then
        gside=$(( gside_b > 0 ? gside_b : 1 ))
    else
        gside="${gside_b}"
    fi

    export HET_MM_NODES_ENV="${mm_nodes}"
    export MPI_RANKS_ENV="${mm}"
    export HET_G_SIDE_TASKS_ENV="${gside}"
    export GPUS_PER_NODE_ENV="${ngpus}"
    export RUN_ID_ENV="hethires3_s${sid}_${CONFIG_TAG}_${LAYOUT_TAG}"
    export JOB_NAME_ENV="hethires3_s${sid}_${CONFIG_TAG}_${LAYOUT_TAG}"
    export SCOREP_DIR_TAG_ENV="hethires3_s${sid}_${CONFIG_TAG}_${LAYOUT_TAG}"
    export SMARTSIM_PORT_OFFSET_ENV="${sid}"
    export PHYDLL_NP_DL_ENV="${ngpus}"

    # Solver variant: AIX uses layout-specific build; others use default
    if [[ "${CONFIG_TAG}" == aix* ]]; then
        export COMPILE_OUTPUT_PATH_ENV="build_scorep_cmi_${LAYOUT_TAG}"
    else
        export COMPILE_OUTPUT_PATH_ENV="build_scorep_cmi_default"
    fi

    # All scaling scenarios use all 4 GPUs on the c23g node
    unset SOLVER_STEP_GPUS_ENV
    unset CUDA_VISIBLE_DEVICES

    # PhyDLL oversubscription for large het runs
    if [[ "${CONFIG_TAG}" == phydll* && "${gside}" -gt 0 ]]; then
        export PHYDLL_DL_OVERSUBSCRIBE_ENV="1"
    else
        unset PHYDLL_DL_OVERSUBSCRIBE_ENV
    fi

    # SmartSim per-GPU database settings
    if [[ "${CONFIG_TAG}" == *"per_gpu_db"* ]]; then
        export DB_NODES_ENV="4"
        export DB_LAYOUT_ENV="per-ml-node"
        export SMARTSIM_DB_GPU_COUNT_ENV="4"
    fi

    echo "=== HH3 s${sid}_${CONFIG_TAG}_${LAYOUT_TAG} (mm=${mm}, mm_nodes=${mm_nodes}, gside=${gside}, gpus=${ngpus}, build=${COMPILE_OUTPUT_PATH_ENV}) start $(date) ===" >> "${SUMMARY}"
    zsh "${MINI_APP_DIR}/het_hires_job.sh" >> "${SUMMARY}" 2>&1
    local rc=$?
    echo "=== HH3 s${sid}_${CONFIG_TAG}_${LAYOUT_TAG} finished rc=${rc} $(date) ===" >> "${SUMMARY}"
    return 0
}

want_scenario 8 && run_scenario 8 192 2 96 4
want_scenario 9 && run_scenario 9 384 4 96 4
want_scenario 10 && run_scenario 10 768 8 96 4
want_scenario 11 && run_scenario 11 1536 16 96 4

echo "HH3 H-scale done $(date): $(grep 'finished' "${SUMMARY}" | tr '\n' ' ')" >> "${SUMMARY}"
