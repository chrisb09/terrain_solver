#!/usr/bin/env bash
# Run with --preflight [--resume] on login for read-only checks/submission command.
# One CMI solver serves both single-node and heterogeneous runtime layouts.
#SBATCH --job-name=build-scorep-pilots
#SBATCH --account=default
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/build_scorep_pilots_%j.out
#SBATCH --error=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/build_scorep_pilots_%j.err

set -Eeuo pipefail
root=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim
mini="$root/mini_app"
cmi="$root/CPP-ML-Interface"
solver="$mini/solver_cpp/build_scorep_pilots"
dl="$cmi/dl_clients/build-scorep-pilots"
aix="$cmi/extern/AIxeleratorService"
prefix="$aix/INSTALL-SCOREP"
torch="$cmi/extern/libtorch"
phy="$cmi/extern/phydll/build"
scorep="$cmi/tmp/opencode/scorep-8.4-papi72-install"
papi="$cmi/tmp/opencode/papi-7.2.0-install"
cuda=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0
stubs="$cuda/lib/stubs"
hdf5=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/HDF5/1.12.2-gompi-2022a
python="$root/python/smartsim_cuda-12/bin/python"
toml="$mini/solver_cpp/build_scorep_cmi_fresh/_deps/tomlplusplus-src"
fail() { printf 'ERROR (line %s): %s\n' "${BASH_LINENO[0]}" "$*" >&2; exit 1; }
build_error() {
    local status="$1" line="$2" command="${3%% *}"
    # Report only a safe command name, never expanded arguments or assignments.
    [[ "$command" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*$ ]] || command='[redacted compound command]'
    printf 'Build check failed (status=%s, line=%s, command=%s); inspect %s\n' \
        "$status" "$line" "$command" "${provenance:-Slurm stderr}" >&2
    exit "$status"
}
trap 'build_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
resume=0
preflight=0
for arg in "$@"; do
    case "$arg" in
        --resume) resume=1 ;;
        --preflight) preflight=1 ;;
        --resume-preflight) resume=1; preflight=1 ;;
        *) fail 'Supported arguments: --resume, --preflight, --resume-preflight.' ;;
    esac
done
check_pilot_caches() {
    local cache entry key value actual
    for cache in "$solver/CMakeCache.txt" "$dl/CMakeCache.txt"; do
        [[ -r "$cache" && ! -L "$cache" ]] || fail "Missing or symlinked resume cache: $cache"
        for entry in WITH_SCOREP=ON CPPML_SCOREP_MPP=mpi SCOREP_MPP_SYSTEM=mpi \
            WITH_TORCH=ON PHYDLL_DL_WITH_TORCH=ON CMAKE_BUILD_TYPE=Release \
            SCOREP_ENABLE_COMPILER=OFF SCOREP_ENABLE_USER=ON \
            "CMAKE_CXX_COMPILER=$scorep/bin/scorep-mpicxx" \
            "CMAKE_C_COMPILER=$scorep/bin/scorep-mpicc" \
            "LIBTORCH_DIR=$torch" "Torch_DIR=$torch/share/cmake/Torch" "PHYDLL_BUILD_DIR=$phy"; do
            key="${entry%%=*}"; value="${entry#*=}"
            actual="$(grep -E "^${key}:[^=]+=" "$cache")" || fail "Missing cache setting $key: $cache"
            [[ "${actual#*=}" == "$value" ]] || fail "Unexpected cache setting $key: $cache"
        done
    done
    grep -Fxq "CMAKE_HOME_DIRECTORY:INTERNAL=$mini/solver_cpp" "$solver/CMakeCache.txt" || fail 'Wrong solver source path.'
    grep -Fxq "CMAKE_HOME_DIRECTORY:INTERNAL=$cmi/dl_clients" "$dl/CMakeCache.txt" || fail 'Wrong DL source path.'
    for entry in USE_CPP_ML_INTERFACE WITH_PHYDLL WITH_AIX; do
        grep -Fxq "$entry:BOOL=ON" "$solver/CMakeCache.txt" || fail "Solver requires $entry=ON."
    done
}

for dir in "$mini/logs" "$mini/solver_cpp" "$cmi/dl_clients" "$toml"; do
    [[ -d "$dir" && ! -L "$dir" ]] || fail "Missing or symlinked directory: $dir"
done
for file in "$prefix/lib/libAIxeleratorService.so" "$phy/lib/libphydll.so" \
    "$phy/include/phydll.h" "$torch/share/cmake/Torch/TorchConfig.cmake" \
    "$torch/lib/libtorch.so" "$torch/lib/libtorch_cuda.so" \
    "$cmi/extern/SmartRedis/install/lib64/libsmartredis.so" \
    "$toml/CMakeLists.txt" "$mini/solver_cpp/build_scorep_cmi_fresh/CMakeCache.txt" \
    "$cmi/dl_clients/build-scorep/CMakeCache.txt" \
    "$stubs/libcuda.so" "$stubs/libcuda.so.1" "$stubs/libnvidia-ml.so.1"; do
    [[ -r "$file" ]] || fail "Missing dependency/reference: $file"
done
for tool in "$scorep/bin/scorep" "$scorep/bin/scorep-mpicxx" \
    "$scorep/bin/scorep-mpicc" "$scorep/bin/scorep-config" \
    "$scorep/bin/scorep-info" "$cuda/bin/nvcc" "$hdf5/bin/h5pcc" "$python"; do
    [[ -x "$tool" ]] || fail "Missing executable: $tool"
done
if (( resume )); then
    for dir in "$solver" "$dl"; do
        [[ -d "$dir" && ! -L "$dir" ]] || fail "Resume requires the existing target directory: $dir"
        [[ ! -e "$dir/BUILD_READY" && ! -L "$dir/BUILD_READY" ]] || fail "Refusing to resume a ready build: $dir"
    done
    [[ ! -e "$solver/provenance" || ( -d "$solver/provenance" && ! -L "$solver/provenance" ) ]] || fail 'Unsafe existing provenance.'
    [[ ! -L "$solver/provenance" ]] || fail 'Symlinked provenance is not supported.'
    check_pilot_caches
else
    [[ ! -e "$solver" && ! -L "$solver" && ! -e "$dl" && ! -L "$dl" ]] ||
        fail 'Pilot build directories already exist; use --resume for a failed build.'
fi
facade="$solver/aix-prebuilt-source"
if [[ -e "$facade" || -L "$facade" ]]; then
    [[ -d "$facade" && ! -L "$facade" ]] || fail 'Unsafe AIx facade directory.'
    for part in src cmake; do
        [[ -L "$facade/$part" && "$(readlink "$facade/$part")" == "$aix/$part" ]] || fail "Unexpected AIx facade link: $part"
    done
fi
if (( preflight )); then
        printf 'Verified log directory: %s\n' "$mini/logs"
        printf 'Slurm stdout: %s\nSlurm stderr: %s\n' \
            "$mini/logs/build_scorep_pilots_%j.out" "$mini/logs/build_scorep_pilots_%j.err"
        printf 'Submit (not executed): sbatch --export=NONE %q' "$mini/build_retest_scorep_pilots.sbatch"
        if (( resume )); then printf ' --resume'; fi
        printf '\n'
        exit 0
fi
[[ -n "${SLURM_JOB_ID:-}" && "${SLURM_JOB_PARTITION:-}" == devel ]] ||
    fail 'Compilation requires the devel Slurm allocation; use --preflight on login.'
[[ "${SLURM_CPUS_PER_TASK:-0}" -ge 8 ]] || fail '8 allocated CPUs required.'

# The site helper initializes Lmod and may reference unset module variables.
# It suppresses module-load failures, so validate the resulting tools below.
export CMI_DIR="$cmi" USE_SCOREP=1
export SMARTSIM_SCOREP_ROOT="$scorep" SMARTSIM_PAPI_ROOT="$papi"
set +u
source "$cmi/set_env_claix23_cuda12.4.sh"
set -u
source "$cmi/env_scorep.sh"
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS='--mpp=mpi --nocompiler --user --io=none --memory=none --thread=none --nocuda'
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
export CUDA_ROOT="$cuda" CUDA_HOME="$cuda"
export LIBCLANG_PATH="${EBROOTCLANG:?Clang module failed to load}/lib"
# Build-only driver/NVML resolution on CPU nodes. Never export this environment
# to pilots or execute a CUDA application here; stubs are not runtime drivers.
export LD_LIBRARY_PATH="$stubs:$prefix/lib:$phy/lib:$torch/lib:$cmi/extern/SmartRedis/install/lib64:$cuda/lib64:$cuda/extras/CUPTI/lib64:$scorep/lib:$papi/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$cuda/lib64:$stubs:${LIBRARY_PATH:-}"
export SLURM_EXPORT_ENV=ALL
[[ "$(command -v mpicxx)" == *OpenMPI/4.1.4* ]] || fail 'Expected OpenMPI/4.1.4.'
[[ "$(command -v gcc)" == *11.3.0* ]] || fail 'Expected GCC 11.3.0.'
"$python" -c 'import clang.cindex; clang.cindex.Index.create()'
"$cuda/bin/nvcc" --version
"$hdf5/bin/h5pcc" -showconfig | grep -E 'Parallel HDF5: *yes' || fail 'Parallel HDF5 required.'

if (( ! resume )); then mkdir "$solver" "$dl"; fi
provenance="$solver/provenance"
if [[ -d "$provenance" ]]; then
    previous="$(mktemp -d "$solver/provenance_previous_${SLURM_JOB_ID}_$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")"
    mv -T "$provenance" "$previous/provenance"
    printf 'Previous provenance preserved: %s\n' "$previous/provenance"
fi
mkdir "$provenance"
exec > >(tee "$provenance/build.log") 2>&1
printf 'Build job %s on %s at %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)"
module list 2> "$provenance/modules.txt"
printf '%s\n' \
    'CPU-node build inspection only; BUILD_READY is not GPU runtime validation.' \
    "BUILD_ONLY_LD_LIBRARY_PATH=$LD_LIBRARY_PATH" \
    "CUDA driver/NVML stubs: $stubs (existing SONAME symlinks; no copied stubs)." \
    'Before pilots, use the driver runtime environment, with no CUDA stub paths.' \
    'Then check ldd -r for solver and DL client: real libcuda/NVML, no missing/undefined symbols or stubs.' \
    'GPU execution and the readiness protocol require subsequent pilot runtime checks.' \
    'Build and pilot dependency paths target x86_64 Sapphire Rapids; no host-native architecture flags added.' \
    > "$provenance/build-linkage-notes.txt"
lscpu > "$provenance/build-cpu.txt"
"$scorep/bin/scorep-info" config-summary > "$provenance/scorep-config.txt"
"$scorep/bin/scorep-config" --mpp=mpi --nocompiler --user --libs > "$provenance/scorep-libs.txt"
grep -q 'scorep_adapter_mpi' "$provenance/scorep-libs.txt"
for repo in "$mini" "$cmi" "$aix"; do
    printf '\nRepository: %s\n' "$repo"
    git -C "$repo" rev-parse HEAD
    git -C "$repo" status --short
    git -C "$repo" diff --binary
done > "$provenance/source-git.txt"
cp "$mini/solver_cpp/build_scorep_cmi_fresh/CMakeCache.txt" "$provenance/solver-reference-cache.txt"
cp "$cmi/dl_clients/build-scorep/CMakeCache.txt" "$provenance/dl-reference-cache.txt"
if [[ -r "$prefix/.git_commit" ]]; then
    cp "$prefix/.git_commit" "$provenance/aix-original-stamp.txt"
fi
tar -czf "$provenance/source.tar.gz" -C "$root" \
    mini_app/build_retest_scorep_pilots.sbatch mini_app/solver_cpp/CMakeLists.txt \
    mini_app/solver_cpp/terrain_solver.cpp mini_app/solver_cpp/include \
    CPP-ML-Interface/CMakeLists.txt CPP-ML-Interface/include CPP-ML-Interface/src \
    CPP-ML-Interface/scripts CPP-ML-Interface/dl_clients/CMakeLists.txt \
    CPP-ML-Interface/dl_clients/dl_client.cpp CPP-ML-Interface/dl_clients/phydll_dl_runtime.cpp \
    CPP-ML-Interface/dl_clients/phydll_dl_runtime.hpp
find "$mini/solver_cpp/include" "$cmi/include" "$cmi/src" "$cmi/scripts" -type f -print0 |
    sort -z | xargs -0 sha256sum > "$provenance/source.sha256"
sha256sum "$mini/build_retest_scorep_pilots.sbatch" "$mini/solver_cpp/terrain_solver.cpp" "$mini/solver_cpp/CMakeLists.txt" \
    "$cmi/CMakeLists.txt" "$cmi/dl_clients/CMakeLists.txt" "$cmi/dl_clients/dl_client.cpp" \
    "$cmi/dl_clients/phydll_dl_runtime.cpp" "$cmi/dl_clients/phydll_dl_runtime.hpp" \
    "$prefix/lib/libAIxeleratorService.so" "$phy/lib/libphydll.so" \
    "$torch/lib/libtorch.so" "$stubs/libcuda.so" "$stubs/libnvidia-ml.so" >> "$provenance/source.sha256"
strings "$prefix/lib/libAIxeleratorService.so" > "$provenance/aix-strings.txt"
for region in 'torchInference::inference' 'torchInference::forward' 'gatherInputData' 'scatterOutputData' 'SCOREP_User_RegionBegin'; do
    grep -Fxq "$region" "$provenance/aix-strings.txt" || fail "AIx instrumentation missing: $region"
done

# Read-only AIx facade: prevent CMI's git-stamp bootstrap from rebuilding/installing
# stale prebuilt dependencies. The real binary/stamp and source revision are archived
# above, not relabeled as a fresh AIx build. Git cannot ascend beyond the build dir.
if [[ ! -d "$facade" ]]; then
    mkdir "$facade"
    ln -s "$aix/src" "$facade/src"
    ln -s "$aix/cmake" "$facade/cmake"
fi
if GIT_CEILING_DIRECTORIES="$solver" git -C "$facade" rev-parse HEAD >/dev/null 2>&1; then
    fail 'AIx facade unexpectedly resolves a Git revision; bootstrap would be unsafe.'
fi
common=( -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    "-DCMAKE_CXX_COMPILER=$scorep/bin/scorep-mpicxx" "-DCMAKE_C_COMPILER=$scorep/bin/scorep-mpicc"
    -DWITH_SCOREP=ON -DCPPML_SCOREP_MPP=mpi -DSCOREP_MPP_SYSTEM=mpi
    -DSCOREP_ENABLE_COMPILER=OFF -DSCOREP_ENABLE_USER=ON
    "-DSCOREP_ROOT_DIR=$scorep" "-DSCOREP_CONFIG_EXECUTABLE=$scorep/bin/scorep-config"
    "-DSCOREP_INFO_EXECUTABLE=$scorep/bin/scorep-info"
    -DWITH_TORCH=ON -DPHYDLL_DL_WITH_TORCH=ON -DTORCH_VERSION=2.4.0
    "-DLIBTORCH_DIR=$torch" "-DTorch_DIR=$torch/share/cmake/Torch"
    "-DPHYDLL_BUILD_DIR=$phy" "-DCUDA_TOOLKIT_ROOT_DIR=$cuda"
    "-DCUDAToolkit_ROOT=$cuda" "-DCMAKE_CUDA_COMPILER=$cuda/bin/nvcc"
    "-DCUDA_HOST_COMPILER=$(command -v gcc)"
    "-DCUDA_CUDA_LIBRARY=$stubs/libcuda.so" "-DCUDA_cuda_driver_LIBRARY=$stubs/libcuda.so"
    "-DCMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES=/lib;/lib64;/usr/lib;/usr/lib64;$stubs;$cuda/targets/x86_64-linux/lib/stubs"
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF
    "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,$stubs"
    "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,$stubs" )
GIT_CEILING_DIRECTORIES="$solver" cmake -S "$mini/solver_cpp" -B "$solver" "${common[@]}" \
    -DUSE_CPP_ML_INTERFACE=ON -DWITH_DIRECT_SMARTSIM=OFF -DWITH_DIRECT_AIX=OFF \
    -DWITH_SMARTSIM=ON -DWITH_AIX=ON -DWITH_PHYDLL=ON -DWITH_FORTRAN=OFF -DBUILD_TESTING=OFF \
    -DAIX_USE_PREBUILT=ON -DFORCE_AIX_REBUILD=OFF "-DAIXELERATOR_DIR=$facade" \
    "-DAIXELERATOR_PREBUILT_INSTALL_PREFIX=$prefix" "-DAIXELERATOR_PREBUILT_LIB_DIR=$prefix/lib" \
    "-DSMARTSIM_PYTHON=$python" "-DHDF5_ROOT=$hdf5" "-DHDF5_C_COMPILER_EXECUTABLE=$hdf5/bin/h5pcc" \
    "-DFETCHCONTENT_SOURCE_DIR_TOMLPLUSPLUS=$toml" -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake -S "$cmi/dl_clients" -B "$dl" "${common[@]}"
check_pilot_caches
grep -q 'PHYDLL_DL_USE_TORCH' "$dl/compile_commands.json" || fail 'Native DL Torch silently disabled.'
# Incremental builds retain existing objects/binaries and compile only stale targets.
cmake --build "$solver" --target terrain_solver --parallel "$SLURM_CPUS_PER_TASK" --verbose
cmake --build "$dl" --target phydll_dl_client --parallel "$SLURM_CPUS_PER_TASK" --verbose
grep -Fq 'solver_readiness_wait' "$solver/generated_registry.hpp"
cmp "$solver/generated_registry.hpp" "$solver/cpp-ml-interface-build/generated_registry.hpp"
grep -Fq -- '-DUSE_CPP_ML_INTERFACE' "$solver/compile_commands.json"
for binary in "$solver/terrain_solver" "$dl/phydll_dl_client" \
    "$prefix/lib/libAIxeleratorService.so" "$solver/cpp-ml-interface-build/libcpp_ml_interface_library.so" \
    "$dl/libphydll_dl_runtime.so"; do
    [[ -r "$binary" ]] || fail "Missing built artifact: $binary"
    name="$(basename "$binary")"
    case "$binary" in
        "$solver/terrain_solver"|"$dl/phydll_dl_client")
            # Executables supply Score-P subsystem symbols: relocation checks stay strict.
            [[ -x "$binary" ]] || fail "Not executable: $binary"
            ldd -r "$binary" > "$provenance/$name.build-only.ldd.txt" 2>&1 ;;
        *)
            # Isolated DSOs consume scorep_subsystems/scorep_number_of_subsystems
            # supplied by the instrumented executable, not by their own dependencies.
            ldd "$binary" > "$provenance/$name.build-only.ldd.txt" 2>&1 ;;
    esac
    ! grep -q 'not found\|undefined symbol' "$provenance/$name.build-only.ldd.txt" || fail "Unresolved build linkage: $binary"
    grep -q 'libscorep' "$provenance/$name.build-only.ldd.txt"
    # The transport-only DL runtime does not itself use Torch; its client does.
    if [[ "$binary" != "$dl/libphydll_dl_runtime.so" ]]; then
        grep -q 'libtorch' "$provenance/$name.build-only.ldd.txt"
    fi
    grep -q 'libmpi' "$provenance/$name.build-only.ldd.txt"
    ! grep -q 'libscorep_mpp_mockup' "$provenance/$name.build-only.ldd.txt"
    readelf -d "$binary" > "$provenance/$name.dynamic.txt"
    readelf --dyn-syms --wide "$binary" > "$provenance/$name.symbols.txt"
    grep -q 'SCOREP_User_RegionBegin' "$provenance/$name.symbols.txt"
done
for binary in "$solver/terrain_solver" "$solver/cpp-ml-interface-build/libcpp_ml_interface_library.so" \
    "$dl/phydll_dl_client" "$dl/libphydll_dl_runtime.so"; do
    readelf -d "$binary" > "$provenance/$(basename "$binary").dynamic.txt"
    if grep -E 'RPATH|RUNPATH|NEEDED' "$provenance/$(basename "$binary").dynamic.txt" | grep -q '/stubs'; then
        fail "CUDA stub path embedded in artifact: $binary"
    fi
    readelf -h "$binary" > "$provenance/$(basename "$binary").elf-header.txt"
    grep -q 'Advanced Micro Devices X86-64' "$provenance/$(basename "$binary").elf-header.txt"
done
for name in terrain_solver phydll_dl_client; do
    grep -q 'libscorep_mpp_mpi' "$provenance/$name.build-only.ldd.txt"
    grep -q 'libscorep_adapter_user' "$provenance/$name.build-only.ldd.txt"
done
strings "$solver/terrain_solver" > "$provenance/solver-strings.txt"
strings "$dl/phydll_dl_client" > "$provenance/dl-strings.txt"
grep -Fq 'phydll_readiness_wait' "$provenance/solver-strings.txt"
grep -Fq 'PhyDLL readiness sequence mismatch' "$provenance/solver-strings.txt"
grep -Fq 'dl_torch_forward' "$provenance/dl-strings.txt"
grep -Fq 'invalid solver_readiness_wait flag' "$provenance/dl-strings.txt"
sha256sum --check "$provenance/source.sha256" > "$provenance/source-check.txt"
sha256sum "$solver/terrain_solver" "$solver/cpp-ml-interface-build/libcpp_ml_interface_library.so" \
    "$dl/phydll_dl_client" "$dl/libphydll_dl_runtime.so" > "$provenance/binaries.sha256"
sha256sum --check "$provenance/binaries.sha256" > "$provenance/binaries-check.txt"
cp "$solver/CMakeCache.txt" "$provenance/solver-cache.txt"
cp "$dl/CMakeCache.txt" "$provenance/dl-cache.txt"
printf 'SOLVER_BIN=%q\nPHYDLL_DL_CLIENT=%q\nUSE_SCOREP=1\nSCOREP_MPP=mpi\nPROVENANCE=%q\n' \
    "$solver/terrain_solver" "$dl/phydll_dl_client" "$provenance" > "$solver/build.env"
printf 'BUILD_READY job=%s completed=%s validation=cpu-build-only runtime_validation=pending\n' \
    "$SLURM_JOB_ID" "$(date -u +%FT%TZ)" > "$solver/BUILD_READY.tmp"
sha256sum "$provenance/binaries.sha256" >> "$solver/BUILD_READY.tmp"
cp "$solver/BUILD_READY.tmp" "$dl/BUILD_READY.tmp"
cmp "$solver/BUILD_READY.tmp" "$dl/BUILD_READY.tmp"
mv "$dl/BUILD_READY.tmp" "$dl/BUILD_READY"
mv "$solver/BUILD_READY.tmp" "$solver/BUILD_READY"
printf 'BUILD_READY: %s\nMatched environment: %s\n' "$solver/BUILD_READY" "$solver/build.env"
