#!/bin/bash
#SBATCH --job-name=phydll_dbg
#SBATCH --account=default
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --time=01:00:00
#SBATCH --gres=none
#SBATCH --output=logs/phydll_debug_%x_%j.log

# PhyDLL MPMD sync debug harness (devel, partial node).
# SUITE=off -> non-profiled solver + non-profiled DL clients
# SUITE=on  -> Score-P solver + Score-P DL clients
# Controlled by PHYDLL_SAFE_MPI per sub-condition (default 0 = no MCA overrides).

set -uo pipefail

SUITE="${SUITE:?SUITE must be off|on}"
MINI_APP_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app"
BASE_DIR="/rwthfs/rz/cluster/hpcwork/ro092286/smartsim"
CPP_DIR="${BASE_DIR}/CPP-ML-Interface"
cd "${MINI_APP_DIR}" || exit 1

source "${BASE_DIR}/set_env_claix23_cuda12.4.sh"
export CUDA_STUBS_DIR="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/stubs/lib64"
export LD_LIBRARY_PATH="${CUDA_STUBS_DIR}:${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HDF5_USE_FILE_LOCKING=FALSE

# The PhyDLL MPMD split; module_test passes these consistently.
export MLCOUPLING_INTRA_OP_THREADS=1
export MLCOUPLING_INTER_OP_THREADS=1
export PHYDLL_DL_COUNT=1
export PHYDLL_DL_FIELD_COUNT=1
# NOTE: deliberately NOT setting PHYDLL_MPMD_SHUTDOWN_BARRIER (module_test does not).

MODEL="${MINI_APP_DIR}/train_models/model_a/watercnn_cpu.pt"
PREP_H5="${MINI_APP_DIR}/input/prep_watercnn_216x144.h5"
SMARTSIM_PYTHON="${CPP_DIR}/extern/python/smartsim_cuda-12/bin/python"
PY_CLIENT="${CPP_DIR}/dl_clients/phydll_dl_client.py"
DL_CPP_PLAIN="${CPP_DIR}/dl_clients/build/phydll_dl_client"
DL_CPP_SCOREP="${CPP_DIR}/dl_clients/build-scorep/phydll_dl_client"

if [[ "${SUITE}" == "on" ]]; then
    SOLVER_BIN="${MINI_APP_DIR}/solver_cpp/build_scorep_cmi_fresh/terrain_solver"
    export SMARTSIM_PAPI_ROOT="${CPP_DIR}/tmp/opencode/papi-7.2.0-install"
    export SMARTSIM_SCOREP_ROOT="${CPP_DIR}/tmp/opencode/scorep-8.4-papi72-install"
    source "${CPP_DIR}/env_scorep.sh"
    export SCOREP_ENABLE_PROFILING=true
    export SCOREP_ENABLE_TRACING=false
    export SCOREP_METRIC_PAPI=
    export SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY=true
else
    SOLVER_BIN="${MINI_APP_DIR}/solver_cpp/build_matrix_default/terrain_solver"
fi

run_root="/hpcwork/thes2181/mini_app/phydll_debug_${SUITE}_${SLURM_JOB_ID:-$$}"
mkdir -p "${run_root}" logs

SOLVER_ARGS=(
    --device CPU --gpus-per-node 0 --ml-nodes 1
    --model-backend TORCH --model-path "${MODEL}" --model-io-layout flat_contiguous
    --cpp-ml-config config_phydll.toml
    --input-hdf5 "${PREP_H5}" --steps 2 --save-every 2 --save-mode periodic
    --triangular-scale 1 --chunk-size 12 --io-mode parallel_hdf5
    --mpi-sync-mode none --hdf5-xfer-mode independent
    --rank-grid-x 3 --rank-grid-z 2
    --ml-interface cpp --ml-batch-size 100000
)

declare -A STATUS

classify() {
    local log="$1" rc="$2"
    if [[ "${rc}" -eq 124 || "${rc}" -eq 137 ]]; then echo "TIMEOUT(rc=${rc})"; return; fi
    if [[ "${rc}" -ne 0 ]]; then echo "FAIL(rc=${rc})"; return; fi
    if grep -q "kernel_init split" "${log}" 2>/dev/null; then echo "PASS"; else echo "PASS(no-split-log)"; fi
}

run_case() {
    local name="$1" dl_kind="$2" safe_mpi="$3"
    local log="logs/phydll_dbg_${SUITE}_${name}_${SLURM_JOB_ID:-$$}.log"
    local outdir="${run_root}/${name}"
    mkdir -p "${outdir}"

    if [[ "${safe_mpi}" == "1" ]]; then
        export OMPI_MCA_pml=ob1
        export OMPI_MCA_btl=self,vader,tcp
        export PMIX_MCA_gds=hash
        local mca_env=(-x OMPI_MCA_pml -x OMPI_MCA_btl -x PMIX_MCA_gds)
    else
        unset OMPI_MCA_pml OMPI_MCA_btl PMIX_MCA_gds
        local mca_env=()
    fi

    local dl_cmd
    if [[ "${dl_kind}" == "cpp" ]]; then
        if [[ "${SUITE}" == "on" ]]; then dl_cmd=("${DL_CPP_SCOREP}"); else dl_cmd=("${DL_CPP_PLAIN}"); fi
    else
        if [[ "${SUITE}" == "on" ]] && "${SMARTSIM_PYTHON}" -c 'import importlib.util,sys; sys.exit(0 if importlib.util.find_spec("scorep.__main__") else 1)' 2>/dev/null; then
            dl_cmd=("${SMARTSIM_PYTHON}" "-m" "scorep" "--keep-files" "--instrumenter-type=dummy" "--noinstrumenter" "--mpp=mpi" "${PY_CLIENT}")
        else
            dl_cmd=("${SMARTSIM_PYTHON}" "${PY_CLIENT}")
        fi
    fi

    echo ""
    echo ">>> [${SUITE}] ${name}: dl=${dl_kind} safe_mpi=${safe_mpi}"
    local scorep_env=()
    if [[ "${SUITE}" == "on" ]]; then
        export SCOREP_EXPERIMENT_DIRECTORY="${outdir}/scorep"; rm -rf "${SCOREP_EXPERIMENT_DIRECTORY}"
        export ENABLE_SCOREP_USER=1
        scorep_env=(-x SCOREP_EXPERIMENT_DIRECTORY -x SCOREP_ENABLE_PROFILING -x SCOREP_ENABLE_TRACING -x SCOREP_METRIC_PAPI -x SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY -x ENABLE_SCOREP_USER)
    fi

    timeout -k 10 150 mpirun --bind-to none --oversubscribe \
        -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
        -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
        "${mca_env[@]}" "${scorep_env[@]}" \
        -n 6 "${SOLVER_BIN}" "${SOLVER_ARGS[@]}" --output-hdf5 "${outdir}/traj.h5" \
        : -x LD_LIBRARY_PATH -x PHYDLL_DL_COUNT -x PHYDLL_DL_FIELD_COUNT \
        -x MLCOUPLING_INTRA_OP_THREADS -x MLCOUPLING_INTER_OP_THREADS \
        "${mca_env[@]}" "${scorep_env[@]}" \
        -n 1 "${dl_cmd[@]}" > "${log}" 2>&1
    local rc=$?
    STATUS["${name}"]="$(classify "${log}" "${rc}")"
    echo "    -> ${STATUS[${name}]}"
}

if [[ "${SUITE}" == "off" ]]; then
    run_case "cpp_nomca"    cpp    0
    run_case "py_nomca"     python 0
    run_case "cpp_safempi"  cpp    1
    run_case "py_safempi"   python 1
else
    run_case "cpp_nomca"    cpp    0
    run_case "py_nomca"     python 0
    run_case "py_safempi"   python 1
    run_case "cpp_safempi"  cpp    1
fi

echo ""
echo "==================== SUMMARY (SUITE=${SUITE}) ===================="
for k in "${!STATUS[@]}"; do printf "  %-16s %s\n" "${k}" "${STATUS[$k]}"; done
echo "run_root: ${run_root}"
echo "================================================================"
