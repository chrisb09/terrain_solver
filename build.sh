#!/usr/bin/env bash

# Resolve mutually exclusive ML mode
# Modes: cmi (default), direct_aix (or aix), direct_smartsim (or smartsim), none
BUILD_ML_MODE="${BUILD_ML_MODE:-${MODE:-}}"

if [[ -z "${BUILD_ML_MODE}" ]]; then
    if [[ "${WITH_DIRECT_AIX:-0}" == "1" || "${WITH_DIRECT_AIX:-}" == "ON" ]]; then
        BUILD_ML_MODE="direct_aix"
    elif [[ "${WITH_DIRECT_SMARTSIM:-0}" == "1" || "${WITH_DIRECT_SMARTSIM:-}" == "ON" ]]; then
        BUILD_ML_MODE="direct_smartsim"
    elif [[ "${USE_CPP_ML_INTERFACE:-ON}" == "OFF" || "${USE_CPP_ML_INTERFACE:-ON}" == "0" ]]; then
        BUILD_ML_MODE="direct_smartsim"
    else
        BUILD_ML_MODE="cmi"
    fi
fi

case "${BUILD_ML_MODE}" in
    cmi|cpp)
        CMAKE_ML_FLAGS="-DUSE_CPP_ML_INTERFACE=ON -DWITH_DIRECT_SMARTSIM=OFF -DWITH_DIRECT_AIX=OFF"
        ;;
    direct_aix|aix)
        CMAKE_ML_FLAGS="-DUSE_CPP_ML_INTERFACE=OFF -DWITH_DIRECT_SMARTSIM=OFF -DWITH_DIRECT_AIX=ON"
        ;;
    direct_smartsim|smartsim)
        CMAKE_ML_FLAGS="-DUSE_CPP_ML_INTERFACE=OFF -DWITH_DIRECT_SMARTSIM=ON -DWITH_DIRECT_AIX=OFF"
        ;;
    none|no_ml)
        CMAKE_ML_FLAGS="-DUSE_CPP_ML_INTERFACE=OFF -DWITH_DIRECT_SMARTSIM=OFF -DWITH_DIRECT_AIX=OFF"
        ;;
    *)
        echo "Unknown BUILD_ML_MODE: '${BUILD_ML_MODE}' (expected cmi, direct_aix, direct_smartsim, none)" >&2
        exit 1
        ;;
esac

echo "Building terrain solver with mode: ${BUILD_ML_MODE} (${CMAKE_ML_FLAGS})"

# Build the terrain solver

current_dir=$(pwd)

custom_build_dir="${1:-build}"
if [ -n "$custom_build_dir" ]; then
  echo "Using custom build directory: $custom_build_dir"
else
  echo "Using default build directory: build"
fi

# delete the old build directory if it exists to ensure a clean build
if [ -d "solver_cpp/${custom_build_dir}" ]; then
  echo "Removing old build directory: solver_cpp/${custom_build_dir}"
  rm -rf "solver_cpp/${custom_build_dir}"
fi

# create build directory if it doesn't exist
if [ ! -d "solver_cpp/${custom_build_dir}" ]; then
  mkdir -p "solver_cpp/${custom_build_dir}"
  echo "Created build directory: solver_cpp/${custom_build_dir}"
fi

script_dir=$(dirname "$0")
# get absolute path to the script directory
script_dir=$(cd "$script_dir" && pwd)

cd "$script_dir" || exit 1

# Record timestamp before build
build_start_seconds=$(date +%s)
build_start_date=$(date)
echo "Build started at: $build_start_date"

cd "solver_cpp/${custom_build_dir}" || { echo "Failed to change directory to solver_cpp/${custom_build_dir}"; cd "$current_dir"; exit 1; }


pushd /hpcwork/ro092286/smartsim > /dev/null
source ./install.sh cuda-12
popd > /dev/null

CUDA_ROOT="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0"
export LD_LIBRARY_PATH="$CUDA_ROOT/extras/CUPTI/lib64:$CUDA_ROOT/lib64:$CUDA_ROOT/targets/x86_64-linux/lib/stubs:${script_dir}/../CPP-ML-Interface/extern/phydll/build/lib:${script_dir}/../CPP-ML-Interface/extern/SmartRedis/install/lib64:${script_dir}/../CPP-ML-Interface/extern/SmartRedis/install/lib:${script_dir}/../CPP-ML-Interface/extern/AIxeleratorService/INSTALL/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_ROOT/extras/CUPTI/lib64:$CUDA_ROOT/lib64:$CUDA_ROOT/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH"
export PATH="$CUDA_ROOT/bin:$PATH"

AIX_SERVICE_NAME="${AIX_SERVICE_NAME_ENV:-AIxeleratorService}"
AIX_SERVICE_DIR="$(realpath "${script_dir}/../CPP-ML-Interface/extern/${AIX_SERVICE_NAME}")"
TORCH_VERSION="${TORCH_VERSION:-2.4.0}"

if [[ "${USE_SCOREP:-}" == "1" ]]; then
    CMI_DIR="$(realpath "${script_dir}/../CPP-ML-Interface")"
    export SMARTSIM_PAPI_ROOT="${CMI_DIR}/tmp/opencode/papi-7.2.0-install"
    export SMARTSIM_SCOREP_ROOT="${CMI_DIR}/tmp/opencode/scorep-8.4-papi72-install"
    CMI_SCOREP_ENV="${CMI_DIR}/env_scorep.sh"
    if [[ -f "${CMI_SCOREP_ENV}" ]]; then
        source "${CMI_SCOREP_ENV}"
    fi
    if [[ "${AIX_SERVICE_NAME}" == *"pipelined"* || "${AIX_SERVICE_NAME}" == *"P2P"* || "${AIX_SERVICE_NAME}" == *"p2p"* ]]; then
        AIX_INSTALL_PREFIX="${AIX_SERVICE_DIR}/INSTALL-PIPELINED-SCOREP-MPI"
        if [[ ! -d "${AIX_INSTALL_PREFIX}" ]]; then
            AIX_INSTALL_PREFIX="${AIX_SERVICE_DIR}/INSTALL-PIPELINED-SCOREP"
        fi
    else
        AIX_INSTALL_PREFIX="${AIX_SERVICE_DIR}/INSTALL-SCOREP"
    fi
    export LD_LIBRARY_PATH="${AIX_INSTALL_PREFIX}/lib:${SMARTSIM_SCOREP_ROOT}/lib:${SMARTSIM_PAPI_ROOT}/lib:${LD_LIBRARY_PATH}"

    if ! command -v scorep-config >/dev/null 2>&1 || ! command -v scorep-mpicxx >/dev/null 2>&1; then
        echo "USE_SCOREP=1 but local Score-P tools are unavailable on PATH." >&2
        exit 1
    fi

    echo "USE_SCOREP=1 detected, using local scorep-mpicxx compiler for CMake"
    SCOREP_BIN_DIR="$(dirname "$(command -v scorep-config)")"
    export SCOREP_WRAPPER_INSTRUMENTER_FLAGS="--nocompiler --user --mpp=${SCOREP_MPP:-none} --io=none --memory=none --thread=none --nocuda"
    SCOREP_FLAGS="-DCMAKE_CXX_COMPILER=${SCOREP_BIN_DIR}/scorep-mpicxx -DCMAKE_C_COMPILER=${SCOREP_BIN_DIR}/scorep-mpicc -DWITH_SCOREP=ON -DSCOREP_ROOT_DIR=${SMARTSIM_SCOREP_ROOT} -DSCOREP_CONFIG_EXECUTABLE=${SCOREP_BIN_DIR}/scorep-config -DSCOREP_INFO_EXECUTABLE=${SCOREP_BIN_DIR}/scorep-info -DAIX_USE_PREBUILT=ON -DFORCE_AIX_REBUILD=${FORCE_AIX_REBUILD_ENV:-OFF} -DAIXELERATOR_DIR=${AIX_SERVICE_DIR} -DAIXELERATOR_PREBUILT_INSTALL_PREFIX=${AIX_INSTALL_PREFIX} -DTORCH_VERSION=${TORCH_VERSION}"
    unset CC
    unset CXX
else
    if [[ "${AIX_SERVICE_NAME}" == *"pipelined"* || "${AIX_SERVICE_NAME}" == *"P2P"* || "${AIX_SERVICE_NAME}" == *"p2p"* ]]; then
        AIX_INSTALL_PREFIX="${AIX_SERVICE_DIR}/INSTALL-PIPELINED"
    else
        AIX_INSTALL_PREFIX="${AIX_SERVICE_DIR}/INSTALL"
    fi
    SCOREP_FLAGS="-DWITH_SCOREP=OFF -DAIX_USE_PREBUILT=ON -DAIXELERATOR_DIR=${AIX_SERVICE_DIR} -DAIXELERATOR_PREBUILT_INSTALL_PREFIX=${AIX_INSTALL_PREFIX}"
fi

cmake -S .. -DCMAKE_BUILD_TYPE=Release ${CMAKE_ML_FLAGS} $SCOREP_FLAGS || { echo "Build failed"; cd "$current_dir"; exit 1; }
build_jobs="${SLURM_CPUS_ON_NODE:-4}"
echo "Building with -j${build_jobs} parallel jobs..."
cmake --build . -j ${build_jobs} || { echo "Build failed"; cd "$current_dir"; exit 1; }
echo "Build completed successfully"

# Get the build timestamp from the binary
if [[ "${SKIP_TIMESTAMP_CHECK:-0}" == "1" ]]; then
    echo "Skipping binary timestamp verification (SKIP_TIMESTAMP_CHECK=1)"
else
    echo ""
    echo "Verifying build timestamp..."
    if [[ "${USE_SCOREP:-}" == "1" ]]; then
        # Running MPI instrumented binaries directly under Slurm can hang due to PMIx initialization; run via mpirun
        binary_output=$(mpirun -n 1 ./terrain_solver --print-build-timestamp 2>&1)
    else
        binary_output=$(./terrain_solver --print-build-timestamp 2>&1)
    fi
    build_timestamp_line=$(echo "$binary_output" | grep "Build timestamp:")

    if [ -z "$build_timestamp_line" ]; then
        echo "ERROR: Could not extract build timestamp from binary."
        echo "Binary output was:"
        echo "${binary_output}"
        cd "$current_dir"
        exit 1
    fi

    echo "Binary reports: $build_timestamp_line"

    # Extract the timestamp string (everything after "Build timestamp: ")
    binary_timestamp_str=$(echo "$build_timestamp_line" | sed 's/Build timestamp: //')

    # Convert binary timestamp to seconds since epoch for comparison
    # Handle format like "Mar 15 2026 14:23:45"
    binary_timestamp_seconds=$(date -d "$binary_timestamp_str" +%s 2>/dev/null)

    if [ -z "$binary_timestamp_seconds" ]; then
        echo "ERROR: Could not parse binary timestamp"
        cd "$current_dir"
        exit 1
    fi

    # Compare timestamps
    if [ "$binary_timestamp_seconds" -ge "$build_start_seconds" ]; then
        echo "✓ Binary timestamp verification PASSED"
        echo "  Build started at:  $(date -d @$build_start_seconds)"
        echo "  Binary built at:   $(date -d @$binary_timestamp_seconds)"
    else
        echo "✗ Binary timestamp verification FAILED"
        echo "  Build started at:  $(date -d @$build_start_seconds)"
        echo "  Binary built at:   $(date -d @$binary_timestamp_seconds)"
        echo "  Binary timestamp is older than build start time!"
        cd "$current_dir"
        exit 1
    fi
fi

cd "$current_dir"
