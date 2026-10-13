#!/bin/zsh
# HH2 H-class wrapper: het allocation (96 c23mm + 96 c23g tasks, 4 GPUs,
# exclusive). Runs the mixed-mm/g scenarios sequentially for ONE library
# config:
#   s2 = 4+0   (1 GPU)   s4 = 24+0   (1 GPU)
#   s5 = 24+24 (1 GPU)   s7 = 96+96  (4 GPUs)
# The library config (provider exports) is supplied at submission via
# --export; HH2_CONFIG_ENV names it. c23g-side SOLVER tasks depend on the
# provider: AIX -> max(B,1) (auto-selected GPU controllers), smartsim/phydll
# -> B (0 = the c23g group only hosts DB / DL workers). Each scenario invokes
# het_hires_job.sh as a child process with per-scenario env.
#SBATCH --job-name=HH2_H
#SBATCH --account=thes2181
#SBATCH --time=02:30:00
#SBATCH --output=logs/hh2_%j.txt
#SBATCH --partition=c23mm
#SBATCH --nodes=1
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

CONFIG_TAG="${HH2_CONFIG_ENV:?HH2_CONFIG_ENV required}"
MINI_APP_DIR="/hpcwork/ro092286/smartsim/mini_app"
cd "${MINI_APP_DIR}"
mkdir -p logs

SUMMARY="logs/hh2_summary_${SLURM_JOB_ID}.txt"
echo "HH2 H-class config=${CONFIG_TAG} scenarios=s2,s4,s5,s7 start=$(date)" > "${SUMMARY}"

run_scenario() {
    local sid="$1" mm="$2" gside_b="$3" ngpus="$4"
    local gside
    if [[ "${CONFIG_TAG}" == aix* ]]; then
        gside=$(( gside_b > 0 ? gside_b : 1 ))   # AIX: >= 1 controller task
    else
        gside="${gside_b}"                        # smartsim/phydll: B (may be 0)
    fi
    export RUN_ID_ENV="hethires2_s${sid}_${CONFIG_TAG}"
    export JOB_NAME_ENV="hethires2_s${sid}_${CONFIG_TAG}"
    export SCOREP_DIR_TAG_ENV="hethires2_s${sid}_${CONFIG_TAG}"
    export MPI_RANKS_ENV="${mm}"
    export HET_G_SIDE_TASKS_ENV="${gside}"
    export GPUS_PER_NODE_ENV="${ngpus}"
    export SMARTSIM_PORT_OFFSET_ENV="${sid}"
    export PHYDLL_NP_DL_ENV="${ngpus}"
    # PhyDLL scenarios where the c23g group hosts BOTH solver ranks and the
    # DL workers need a globally oversubscribed MPMD step.
    if [[ "${CONFIG_TAG}" == phydll* && "${gside}" -gt 0 ]]; then
        export PHYDLL_DL_OVERSUBSCRIBE_ENV="1"
    else
        unset PHYDLL_DL_OVERSUBSCRIBE_ENV
    fi
    if [[ "${ngpus}" -le 1 ]]; then
        export SOLVER_STEP_GPUS_ENV="1"
    else
        unset SOLVER_STEP_GPUS_ENV
    fi
    echo "=== HH2 s${sid} (mm=${mm}, gside=${gside}, gpus=${ngpus}) start $(date) ===" >> "${SUMMARY}"
    zsh "${MINI_APP_DIR}/het_hires_job.sh" >> "${SUMMARY}" 2>&1
    local rc=$?
    echo "=== HH2 s${sid}_${CONFIG_TAG} finished rc=${rc} $(date) ===" >> "${SUMMARY}"
    return 0   # continue with the next scenario regardless
}

# Optional scenario filter: HH2_SCENARIOS_ENV="5,7" runs only those sX ids.
want_scenario() {
    [[ -z "${HH2_SCENARIOS_ENV:-}" ]] || [[ ",${HH2_SCENARIOS_ENV}," == *",${1},"* ]]
}

want_scenario 2 && run_scenario 2 4 0 1
want_scenario 4 && run_scenario 4 24 0 1
want_scenario 5 && run_scenario 5 24 24 1
want_scenario 7 && run_scenario 7 96 96 4

echo "HH2 H wrapper done $(date): $(grep 'finished' "${SUMMARY}" | tr '\n' ' ')" >> "${SUMMARY}"
