#!/usr/bin/env bash
# CPU-only production CUDA build. --preflight is read-only; --resume is for failed builds.
#SBATCH --job-name=build-gpu-refresh
#SBATCH --account=default
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/build_gpu_refresh_%j.out
#SBATCH --error=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/build_gpu_refresh_%j.err
set -Eeuo pipefail
root=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim
mini="$root/mini_app"
cmi="$root/CPP-ML-Interface"
aix="$cmi/extern/AIxeleratorService"
phy_src="$cmi/extern/phydll"
phy="$phy_src/build-gpu-refresh"
torch="$cmi/extern/libtorch"
scorep="$cmi/tmp/opencode/scorep-8.4-papi72-install"
papi="$cmi/tmp/opencode/papi-7.2.0-install"
cuda=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0
hdf5=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/HDF5/1.12.2-gompi-2022a
stubs="$cuda/lib/stubs"
python="$root/python/smartsim_cuda-12/bin/python"
toml="$mini/solver_cpp/build_scorep_cmi_fresh/_deps/tomlplusplus-src"
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
trap 'printf "Build failed at line %s (status %s, pipeline %s): %s; no readiness published for unfinished mode.\n" "$LINENO" "$?" "${PIPESTATUS[*]}" "$BASH_COMMAND" >&2' ERR
exec 3>&1 4>&2
preflight=0; resume=0
for arg in "$@"; do
    case "$arg" in --preflight) preflight=1;; --resume) resume=1;; *) fail "Unknown argument: $arg";; esac
done
git -C "$cmi" merge-base --is-ancestor 7952128 HEAD
git -C "$phy_src" merge-base --is-ancestor 05e0e29 HEAD
provider="$cmi/include/provider/ml_coupling_provider_phydll.hpp"
runtime="$cmi/dl_clients/phydll_dl_runtime.cpp"
client="$cmi/dl_clients/dl_client.cpp"
grep -Fq 'kBcastMetaVersion = 4;' "$provider"
grep -Fq 'kBcastMetaVersion = 4;' "$client"
grep -Fq 'phydll_get_field_view(&buffer_ptr, label)' "$runtime"
grep -Fq 'frame.data.swap(combined_data_)' "$runtime"
grep -Fq 'void DlRuntime::receive_frame(Frame& frame)' "$runtime"
grep -Fq 'void receive_frame(Frame& frame)' "$cmi/dl_clients/phydll_dl_runtime.hpp"
! grep -Eq '\.clone\(|float_codec|wire_dtype' "$client" "$provider" "$runtime" || fail 'Experimental codec or old clone present.'
for dir in "$mini/logs" "$mini/solver_cpp" "$cmi/dl_clients" "$phy_src" "$toml"; do
    [[ -d "$dir" && ! -L "$dir" ]] || fail "Unsafe/missing parent: $dir"
done
for file in "$torch/lib/libtorch_cuda.so" "$torch/share/cmake/Torch/TorchConfig.cmake" \
    "$stubs/libcuda.so.1" "$stubs/libnvidia-ml.so.1" "$aix/INSTALL/lib/libAIxeleratorService.so" \
    "$aix/INSTALL-SCOREP/lib/libAIxeleratorService.so" "$aix/build_aix/CMakeCache.txt" \
    "$aix/build-scorep/CMakeCache.txt" "$scorep/bin/scorep-mpicxx" "$python"; do
    [[ -r "$file" ]] || fail "Missing dependency: $file"
done
grep -Fxq 'WITH_SCOREP:BOOL=OFF' "$aix/build_aix/CMakeCache.txt"
grep -Fxq 'WITH_TORCH:BOOL=ON' "$aix/build_aix/CMakeCache.txt"
grep -q 'MPI_CXX_COMPILER:FILEPATH=.*OpenMPI/4.1.4-GCC-11.3.0' "$aix/build_aix/CMakeCache.txt"
for dir in "$phy" "$mini/solver_cpp/build_gpu_refresh_plain" "$mini/solver_cpp/build_gpu_refresh_scorep" \
    "$cmi/dl_clients/build-gpu-refresh-plain" "$cmi/dl_clients/build-gpu-refresh-scorep"; do
    if [[ -e "$dir" || -L "$dir" ]]; then
        (( resume )) && [[ -d "$dir" && ! -L "$dir" && -f "$dir/GPU_REFRESH_BUILD" ]] || fail "Target exists: $dir"
    fi
done
if (( preflight )); then
    printf 'Preflight passed: CMI=%s PhyDLL=%s\n' "$(git -C "$cmi" rev-parse HEAD)" "$(git -C "$phy_src" rev-parse HEAD)"
    printf 'Submit: sbatch --export=NONE %q' "$mini/build_gpu_refresh.sbatch"
    (( ! resume )) || printf ' --resume'
    printf '\n'; exit 0
fi
[[ "${SLURM_JOB_PARTITION:-}" == devel && "${SLURM_CPUS_PER_TASK:-0}" -ge 8 ]] || fail 'Requires devel CPU allocation.'
[[ -z "${SLURM_JOB_GPUS:-}" ]] || fail 'GPU reservation forbidden.'
export CMI_DIR="$cmi" USE_SCOREP=0
set +u
source "$cmi/set_env_claix23_cuda12.4.sh"
set -u
[[ "$(command -v mpicxx)" == *OpenMPI/4.1.4* && "$(command -v gcc)" == *11.3.0* ]] || fail 'Compiler/MPI mismatch.'
export CUDA_ROOT="$cuda" CUDA_HOME="$cuda" LIBCLANG_PATH="${EBROOTCLANG:?}/lib"
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
base_ld="${LD_LIBRARY_PATH:-}"
base_lib="${LIBRARY_PATH:-}"
base_cpath="${CPATH:-}"
export LD_LIBRARY_PATH="$stubs:$torch/lib:$cuda/lib64:$cuda/extras/CUPTI/lib64:$base_ld"
"$python" -c 'import clang.cindex; clang.cindex.Index.create()'
"$cuda/bin/nvcc" --version
"$hdf5/bin/h5pcc" -showconfig | grep -E 'Parallel HDF5: *yes'
if [[ ! -d "$phy" ]]; then
    ls -d "$phy_src"
    mkdir "$phy"
    touch "$phy/GPU_REFRESH_BUILD"
fi
ls -d "$phy"
mkdir -p "$phy/lib" "$phy/include"
# The shared transport core is plain MPI in both modes, with identical production flags.
if [[ ! -e "$mini/solver_cpp/build_gpu_refresh_plain/BUILD_READY" && ! -e "$mini/solver_cpp/build_gpu_refresh_scorep/BUILD_READY" ]]; then
    make -C "$phy_src" compile "BUILD=$phy" "CC=$(command -v mpicc)" 'CFLAGS=-O2 -g -Wall -Wextra -std=c99'
fi
cmp "$phy_src/src/core/phydll.h" "$phy/include/phydll.h"
nm -D --defined-only "$phy/lib/libphydll.so" > "$phy/exported-symbols.txt"
grep ' phydll_get_field_view$' "$phy/exported-symbols.txt"
for mode in plain scorep; do
    solver="$mini/solver_cpp/build_gpu_refresh_$mode"
    dl="$cmi/dl_clients/build-gpu-refresh-$mode"
    [[ ! -e "$solver/BUILD_READY" && ! -e "$dl/BUILD_READY" ]] || { printf 'Preserving ready mode: %s\n' "$mode"; continue; }
    for dir in "$solver" "$dl"; do
        if [[ ! -d "$dir" ]]; then ls -d "$(dirname "$dir")"; mkdir "$dir"; touch "$dir/GPU_REFRESH_BUILD"; fi
    done
    provenance="$solver/provenance-$SLURM_JOB_ID"
    ls -d "$solver"
    mkdir "$provenance"
    exec > >(tee -a "$provenance/build.log" >&3) 2>&1
    on=OFF; use=0; prefix="$aix/INSTALL"; cc="$(command -v mpicc)"; cxx="$(command -v mpicxx)"
    if [[ "$mode" == scorep ]]; then on=ON; use=1; prefix="$solver/aix-refresh-install"; cc="$scorep/bin/scorep-mpicc"; cxx="$scorep/bin/scorep-mpicxx"; fi
    export USE_SCOREP="$use"
    export SCOREP_WRAPPER_INSTRUMENTER_FLAGS='--mpp=mpi --nocompiler --user --io=none --memory=none --thread=none --nocuda'
    export LD_LIBRARY_PATH="$stubs:$prefix/lib:$phy/lib:$dl:$solver/cpp-ml-interface-build:$torch/lib:$cmi/extern/SmartRedis/install/lib64:$cuda/lib64:$cuda/extras/CUPTI/lib64:$scorep/lib:$papi/lib:$base_ld"
    export LIBRARY_PATH="$cuda/lib64:$stubs:$scorep/lib:$papi/lib:$base_lib"
    export CPATH="$scorep/include:$papi/include:$base_cpath"
    module list 2> "$provenance/modules.txt"
    printf 'mode=%s job=%s flags=%s\nBUILD_ONLY_LD_LIBRARY_PATH=%s\n' "$mode" "$SLURM_JOB_ID" "$SCOREP_WRAPPER_INSTRUMENTER_FLAGS" "$LD_LIBRARY_PATH" > "$provenance/environment.txt"
    if [[ "$mode" == scorep ]]; then
        # The shared production install has compiler instrumentation; leave it untouched.
        cmake -S "$aix" -B "$solver/aix-refresh-build" -DCMAKE_BUILD_TYPE=Release \
            "-DCMAKE_C_COMPILER=$(command -v mpicc)" "-DCMAKE_CXX_COMPILER=$(command -v mpicxx)" \
            "-DCMAKE_Fortran_COMPILER=$(command -v mpifort)" -DBUILD_TESTS=OFF -DWITH_TORCH=ON -DWITH_SCOREP=ON \
            -DWITH_FORTRAN_INTERFACE=OFF -DAIX_SKIP_VENV_CREATION=ON "-DAIX_VENV_DIR=$root/python/smartsim_cuda-12" \
            "-DLIBTORCH_DIR=$torch" "-DTorch_DIR=$torch/share/cmake/Torch" -DTORCH_VERSION=2.4.0 \
            "-DCMAKE_CUDA_COMPILER=$cuda/bin/nvcc" "-DCUDAToolkit_ROOT=$cuda" "-DCUDA_TOOLKIT_ROOT_DIR=$cuda" \
            "-DCUDA_CUDA_LIBRARY=$stubs/libcuda.so" "-DCUDA_cuda_driver_LIBRARY=$stubs/libcuda.so" \
            "-DSCOREP_ROOT_DIR=$scorep" "-DSCOREP_CONFIG_EXECUTABLE=$solver/scorep-config-refresh" \
            -DSCOREP_ENABLE_COMPILER=OFF -DSCOREP_ENABLE_USER=ON -DSCOREP_MPP_SYSTEM=mpi \
            "-DCMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES=/lib;/lib64;/usr/lib;/usr/lib64;$stubs;$cuda/targets/x86_64-linux/lib/stubs" \
            "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,$stubs" -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF
        cmake --build "$solver/aix-refresh-build" --target AIxeleratorServiceLib --parallel "$SLURM_CPUS_PER_TASK" --verbose
        ls -d "$solver"
        mkdir -p "$prefix/lib" "$prefix/include"
        cp "$solver/aix-refresh-build/src/libAIxeleratorService.so" "$prefix/lib/"
        cp -a "$aix/src/." "$prefix/include/"
        cp "$solver/aix-refresh-build/CMakeCache.txt" "$provenance/aix-refresh-cache.txt"
        sha256sum "$solver/scorep-config-refresh" > "$provenance/scorep-config.sha256"
    fi
    printf 'PhyDLL core flags: -O2 -g -Wall -Wextra -std=c99 (older build used -O0)\n' >> "$provenance/environment.txt"
    for repo in "$mini" "$cmi" "$phy_src" "$aix"; do
        git -C "$repo" rev-parse HEAD; git -C "$repo" status --short; git -C "$repo" diff --binary
    done > "$provenance/source-git.txt"
    for dir in "$mini/solver_cpp/include" "$cmi/include" "$cmi/src" "$cmi/scripts" "$phy_src/src/core" "$aix/src" "$aix/cmake"; do
        git -C "$dir" ls-files --full-name -z | while IFS= read -r -d '' file; do
            repo_root="$(git -C "$dir" rev-parse --show-toplevel)"
            [[ ! -f "$repo_root/$file" ]] || sha256sum "$repo_root/$file"
        done
    done > "$provenance/source.sha256"
    [[ "$mode" != scorep ]] || sha256sum "$solver/scorep-config-refresh" >> "$provenance/source.sha256"
    sha256sum "$mini/build_gpu_refresh.sbatch" "$mini/solver_cpp/CMakeLists.txt" "$mini/solver_cpp/terrain_solver.cpp" \
        "$cmi/CMakeLists.txt" "$client" "$runtime" "$cmi/dl_clients/phydll_dl_runtime.hpp" "$cmi/dl_clients/CMakeLists.txt" \
        "$aix/CMakeLists.txt" "$prefix/lib/libAIxeleratorService.so" "$phy/lib/libphydll.so" "$phy/include/phydll.h" "$torch/lib/libtorch_cuda.so" >> "$provenance/source.sha256"
    cp "$aix/build_aix/CMakeCache.txt" "$provenance/aix-plain-reference-cache.txt"
    cp "$aix/build-scorep/CMakeCache.txt" "$provenance/aix-scorep-reference-cache.txt"
    [[ ! -r "$prefix/.git_commit" ]] || cp "$prefix/.git_commit" "$provenance/aix-original-stamp.txt"
    # Do not relabel/rebuild installed AIx when its older Git stamp differs.
    facade="$solver/aix-prebuilt-source"
    if [[ ! -d "$facade" ]]; then
        ls -d "$solver"; mkdir "$facade"; ln -s "$aix/src" "$facade/src"; ln -s "$aix/cmake" "$facade/cmake"
    fi
    ! GIT_CEILING_DIRECTORIES="$solver" git -C "$facade" rev-parse HEAD >/dev/null 2>&1 || fail 'Unsafe AIx facade.'
    common=( -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        "-DCMAKE_CXX_COMPILER=$cxx" "-DCMAKE_C_COMPILER=$cc" "-DWITH_SCOREP=$on"
        -DCPPML_SCOREP_MPP=mpi -DSCOREP_MPP_SYSTEM=mpi -DSCOREP_ENABLE_COMPILER=OFF -DSCOREP_ENABLE_USER=ON
        "-DSCOREP_ROOT_DIR=$scorep" "-DSCOREP_CONFIG_EXECUTABLE=$scorep/bin/scorep-config" "-DSCOREP_INFO_EXECUTABLE=$scorep/bin/scorep-info"
        -DWITH_TORCH=ON -DPHYDLL_DL_WITH_TORCH=ON -DTORCH_VERSION=2.4.0
        "-DLIBTORCH_DIR=$torch" "-DTorch_DIR=$torch/share/cmake/Torch" "-DPHYDLL_BUILD_DIR=$phy"
        "-DCUDA_TOOLKIT_ROOT_DIR=$cuda" "-DCUDAToolkit_ROOT=$cuda" "-DCMAKE_CUDA_COMPILER=$cuda/bin/nvcc"
        "-DCUDA_HOST_COMPILER=$(command -v gcc)" "-DCUDA_CUDA_LIBRARY=$stubs/libcuda.so" "-DCUDA_cuda_driver_LIBRARY=$stubs/libcuda.so"
        "-DCMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES=/lib;/lib64;/usr/lib;/usr/lib64;$stubs;$cuda/targets/x86_64-linux/lib/stubs"
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,$stubs" "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,$stubs" )
    printf '%q ' cmake "${common[@]}" > "$provenance/configure-flags.txt"
    GIT_CEILING_DIRECTORIES="$solver" cmake -S "$mini/solver_cpp" -B "$solver" "${common[@]}" \
        -DUSE_CPP_ML_INTERFACE=ON -DWITH_DIRECT_SMARTSIM=OFF -DWITH_DIRECT_AIX=OFF \
        -DWITH_SMARTSIM=ON -DWITH_AIX=ON -DWITH_PHYDLL=ON -DWITH_FORTRAN=OFF -DBUILD_TESTING=OFF \
        -DAIX_USE_PREBUILT=ON -DFORCE_AIX_REBUILD=OFF "-DAIXELERATOR_DIR=$facade" \
        "-DAIXELERATOR_PREBUILT_INSTALL_PREFIX=$prefix" "-DAIXELERATOR_PREBUILT_LIB_DIR=$prefix/lib" \
        "-DSMARTSIM_PYTHON=$python" "-DHDF5_ROOT=$hdf5" "-DHDF5_C_COMPILER_EXECUTABLE=$hdf5/bin/h5pcc" \
        "-DFETCHCONTENT_SOURCE_DIR_TOMLPLUSPLUS=$toml" -DFETCHCONTENT_FULLY_DISCONNECTED=ON
    cmake -S "$cmi/dl_clients" -B "$dl" "${common[@]}"
    for cache in "$solver/CMakeCache.txt" "$dl/CMakeCache.txt"; do
        for entry in "WITH_SCOREP=$on" 'CMAKE_BUILD_TYPE=Release' "PHYDLL_BUILD_DIR=$phy" "CMAKE_CXX_COMPILER=$cxx" 'PHYDLL_DL_WITH_TORCH=ON' 'CPPML_SCOREP_MPP=mpi'; do
            grep -Eq "^${entry%%=*}:[^=]+=${entry#*=}$" "$cache" || fail "Cache mismatch: $entry"
        done
    done
    for key in USE_CPP_ML_INTERFACE WITH_SMARTSIM WITH_AIX WITH_PHYDLL WITH_TORCH; do grep -Fxq "$key:BOOL=ON" "$solver/CMakeCache.txt"; done
    grep -q 'PHYDLL_DL_USE_TORCH' "$dl/compile_commands.json"
    cmake --build "$solver" --target terrain_solver --parallel "$SLURM_CPUS_PER_TASK" --verbose
    cmake --build "$dl" --target phydll_dl_client --parallel "$SLURM_CPUS_PER_TASK" --verbose
    cmp "$solver/generated_registry.hpp" "$solver/cpp-ml-interface-build/generated_registry.hpp"
    grep -Fq 'solver_readiness_wait' "$solver/generated_registry.hpp"
    ! grep -Eq 'float_codec|wire_dtype' "$solver/generated_registry.hpp" || fail 'Experimental generated registry.'
    nm -D -C "$dl/phydll_dl_client" > "$provenance/dl-symbols.txt"
    grep -Fq 'c10::cuda::getCurrentCUDAStream' "$provenance/dl-symbols.txt" || fail 'CUDA stream support silently disabled.'
    nm -D -C "$dl/libphydll_dl_runtime.so" > "$provenance/runtime-symbols.txt"
    grep -Fq 'phydll_get_field_view' "$provenance/runtime-symbols.txt"
    for binary in "$solver/terrain_solver" "$dl/phydll_dl_client" "$solver/cpp-ml-interface-build/libcpp_ml_interface_library.so" \
        "$dl/libphydll_dl_runtime.so" "$prefix/lib/libAIxeleratorService.so" "$phy/lib/libphydll.so"; do
        name="$(basename "$binary")"
        ldd -r "$binary" > "$provenance/$name.ldd.txt" 2>&1
        ! grep -q 'not found' "$provenance/$name.ldd.txt" || fail "Missing dependency: $binary"
        if [[ "$binary" == *.so && "$mode" == scorep ]]; then
            ! grep -E 'undefined symbol' "$provenance/$name.ldd.txt" | grep -Ev '^undefined symbol: scorep_(subsystems|number_of_subsystems)[[:space:]]+\([^)]*/libscorep_measurement\.so[^)]*\)$' || fail "Undefined DSO symbol: $binary"
        else
            ! grep -q 'undefined symbol' "$provenance/$name.ldd.txt" || fail "Undefined symbol: $binary"
        fi
        if [[ "$mode" == plain ]]; then ! grep -q 'libscorep' "$provenance/$name.ldd.txt" || fail 'Plain links Score-P.'; fi
        ! grep -Eq 'libscorep_mpp_mockup|libscorep_adapter_cuda|libscorep_adapter_compiler' "$provenance/$name.ldd.txt" || fail "Forbidden Score-P subsystem: $binary"
        readelf -d "$binary" > "$provenance/$name.dynamic.txt"
        ! grep -E '(RPATH|RUNPATH|NEEDED).*/stubs' "$provenance/$name.dynamic.txt" || fail 'Embedded CUDA stubs.'
        sha256sum "$binary" >> "$provenance/binaries.sha256"
    done
    for name in terrain_solver phydll_dl_client; do
        grep -Fq "$phy/lib/libphydll.so" "$provenance/$name.ldd.txt"
        grep -q 'libtorch_cuda' "$provenance/$name.ldd.txt"
        if [[ "$mode" == scorep ]]; then
            grep -q 'libscorep_mpp_mpi' "$provenance/$name.ldd.txt"
            grep -q 'libscorep_adapter_user' "$provenance/$name.ldd.txt"
            ! grep -Eq 'libscorep_mpp_mockup|libscorep_adapter_cuda|libscorep_adapter_compiler' "$provenance/$name.ldd.txt" || fail "Forbidden Score-P subsystem: $name"
        fi
    done
    sha256sum --check "$provenance/source.sha256" > "$provenance/source-check.txt"
    sha256sum --check "$provenance/binaries.sha256" > "$provenance/binaries-check.txt"
    cp "$solver/CMakeCache.txt" "$provenance/solver-cache.txt"
    cp "$dl/CMakeCache.txt" "$provenance/dl-cache.txt"
    runtime_libs="$phy/lib:$dl:$solver/cpp-ml-interface-build:$prefix/lib:$torch/lib:$cmi/extern/SmartRedis/install/lib64:$cuda/lib64:$cuda/extras/CUPTI/lib64"
    [[ "$mode" != scorep ]] || runtime_libs="$runtime_libs:$scorep/lib:$papi/lib"
    printf 'SOLVER_BIN=%q\nPHYDLL_DL_CLIENT=%q\nPHYDLL_BUILD_DIR=%q\nUSE_SCOREP=%q\nSCOREP_MPP=mpi\nPROVENANCE=%q\nGPU_REFRESH_LIBRARY_PATH=%q\nexport LD_LIBRARY_PATH="$GPU_REFRESH_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"\n' \
        "$solver/terrain_solver" "$dl/phydll_dl_client" "$phy" "$use" "$provenance" "$runtime_libs" > "$solver/build.env"
    cp "$solver/build.env" "$dl/build.env"
    printf 'BUILD_READY job=%s mode=%s completed=%s validation=cpu-build-only runtime_validation=pending\n' "$SLURM_JOB_ID" "$mode" "$(date -u +%FT%TZ)" > "$solver/BUILD_READY.tmp"
    sha256sum "$provenance/binaries.sha256" "$provenance/source.sha256" >> "$solver/BUILD_READY.tmp"
    cp "$solver/BUILD_READY.tmp" "$dl/BUILD_READY.tmp"
    cmp "$solver/BUILD_READY.tmp" "$dl/BUILD_READY.tmp"
    mv "$dl/BUILD_READY.tmp" "$dl/BUILD_READY"
    mv "$solver/BUILD_READY.tmp" "$solver/BUILD_READY"
    printf 'BUILD_READY: %s\nMatched environment: %s\n' "$solver/BUILD_READY" "$solver/build.env"
done
