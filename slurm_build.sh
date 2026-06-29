#!/usr/bin/env bash
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=01:00:00
#SBATCH --job-name=build-mini-app
#SBATCH --output=build-mini-app.%j.out
#SBATCH --error=build-mini-app.%j.err

# Exit on any error
set -euo pipefail

# Get directories
export USE_SCOREP=1
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    MINI_APP_DIR="${SLURM_SUBMIT_DIR}"
else
    MINI_APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
CPP_ML_DIR="$(realpath "${MINI_APP_DIR}/../CPP-ML-Interface")"
ABS_SCRIPT="${MINI_APP_DIR}/$(basename "${BASH_SOURCE[0]}")"

# If not already running inside a Slurm job, self-submit via srun
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Not inside a Slurm job. Re-executing via srun on devel partition with 96 cores..."
    exec srun --partition=devel --cpus-per-task=96 --time=01:00:00 "${ABS_SCRIPT}" "$@"
fi

echo "=== Slurm Build Job Started ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "CPUs allocated: ${SLURM_CPUS_ON_NODE:-96}"

# 1. cd into CPP-ML-Interface folder
cd "${CPP_ML_DIR}"

# 2. source install.sh cuda-12
echo "Sourcing environment and runtime from CPP-ML-Interface/install.sh cuda-12..."
source ./install.sh cuda-12

# 3. Handle Score-P gating
if [[ -n "${USE_SCOREP:-}" ]]; then
    echo "USE_SCOREP is set. Loading Score-P and PAPI modules..."
    module load Score-P/8.4
    module load PAPI/7.0.0
    export CXX=scorep-mpicxx
    export CC=scorep-mpicc
    export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--nocompiler --user --mpp=mpi --io=none --memory=malloc --thread=none --nocuda"
fi

# 4. go back to the original dir and run the build.sh
cd "${MINI_APP_DIR}"
echo "Running build.sh in ${MINI_APP_DIR}..."
./build.sh "$@"

echo "=== Slurm Build Job Completed Successfully ==="
