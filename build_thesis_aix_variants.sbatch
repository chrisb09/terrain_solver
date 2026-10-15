#!/usr/bin/env bash
# Six isolated runtime DSOs. --preflight never loads modules or writes files.
#SBATCH --job-name=build-thesis-aix
#SBATCH --account=default
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/build_thesis_aix_variants_%j.out
#SBATCH --error=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/build_thesis_aix_variants_%j.err
set -Eeuo pipefail
root=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim
mini="$root/mini_app"
cmi="$root/CPP-ML-Interface"
aix="$cmi/extern/AIxeleratorService"
targets="$mini/solver_cpp/build_thesis_aix_variants"
expected_sha=86968b64380307a5113db8077ffe4b9494309c71
refresh_job=4938224
software=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software
cuda="$software/CUDA/12.4.0"
hwloc="$software/hwloc/2.7.1-GCCcore-11.3.0"
torch="$cmi/extern/libtorch"
scorep="$cmi/tmp/opencode/scorep-8.4-papi72-install"
papi="$cmi/tmp/opencode/papi-7.2.0-install"
python="$root/python/smartsim_cuda-12/bin/python"
config="$mini/solver_cpp/build_gpu_refresh_scorep/scorep-config-refresh"
stubs="$cuda/lib/stubs"
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
trap 'printf "Failed at line %s: %s; unfinished variant has no READY.\n" "$LINENO" "$BASH_COMMAND" >&2' ERR
preflight=0; resume=0
for arg in "$@"; do
    case "$arg" in --preflight) preflight=1;; --resume) resume=1;; *) fail "Unknown argument: $arg";; esac
done
[[ "$(git -C "$aix" rev-parse HEAD)" == "$expected_sha" ]] || fail 'AIx source revision mismatch.'
[[ -z "$(git -C "$aix" status --porcelain --untracked-files=no)" ]] || fail 'Tracked AIx source is dirty.'
for dir in "$mini/logs" "$mini/solver_cpp" "$aix"; do
    [[ -d "$dir" && ! -L "$dir" ]] || fail "Unsafe/missing parent: $dir"
done
for file in "$python" "$config" "$torch/lib/libtorch_cuda.so" "$torch/share/cmake/Torch/TorchConfig.cmake" \
    "$cuda/bin/nvcc" "$stubs/libcuda.so" "$stubs/libnvidia-ml.so" "$hwloc/lib/libhwloc.so" \
    "$hwloc/include/hwloc.h" /usr/include/numa.h /usr/lib64/libnuma.so "$scorep/bin/scorep-config"; do
    [[ -r "$file" ]] || fail "Missing dependency: $file"
done
grep -Fq -- '--nocompiler --user --mpp=mpi' "$config" || fail 'Unexpected Score-P config wrapper.'
# Validate old readiness/binary manifests, NOT old source manifests: AIx has advanced.
for mode in plain scorep; do
    solver="$mini/solver_cpp/build_gpu_refresh_$mode"
    dl="$cmi/dl_clients/build-gpu-refresh-$mode"
    grep -Eq "^BUILD_READY job=$refresh_job mode=$mode " "$solver/BUILD_READY" || fail 'GPU refresh readiness mismatch.'
    cmp "$solver/BUILD_READY" "$dl/BUILD_READY"
    while IFS= read -r line; do
        [[ "$line" == BUILD_READY* ]] || printf '%s\n' "$line"
    done < "$solver/BUILD_READY" | sha256sum --check --status
    sha256sum --check --status "$solver/provenance-$refresh_job/binaries.sha256"
    [[ -x "$solver/terrain_solver" && -x "$dl/phydll_dl_client" ]] || fail 'Missing verified executables.'
done
# Only marked, non-symlink trees belong to this builder. Never repair a ready tree.
for dir in "$targets" "$targets"/{default,affine,numa}_{plain,scorep}; do
    if [[ -e "$dir" || -L "$dir" ]]; then
        (( resume )) || fail "Target exists (use --resume): $dir"
        [[ -d "$dir" && ! -L "$dir" && -f "$dir/THESIS_AIX_BUILD" && ! -L "$dir/THESIS_AIX_BUILD" ]] || fail "Unowned target: $dir"
        [[ "$(< "$dir/THESIS_AIX_BUILD")" == "$dir" ]] || fail "Ownership mismatch: $dir"
        [[ -z "$(find "$dir" -type l -print -quit)" ]] || fail "Symlink in owned target: $dir"
    fi
done
for layout in default affine numa; do
    for mode in plain scorep; do
        variant="$targets/${layout}_${mode}"
        if [[ -e "$variant/READY" ]]; then
            for entry in "layout=$layout" "mode=$mode" "gitsha=$expected_sha" "gpu_refresh_job=$refresh_job" \
                "prefix=$variant/aix-install" "library=$variant/aix-install/lib/libAIxeleratorService.so"; do
                grep -Fxq "$entry" "$variant/READY" || fail "READY mismatch: $variant ($entry)"
            done
            digest="$(sha256sum "$variant/checksums.sha256")"; digest="${digest%% *}"
            grep -Fxq "checksums_sha256=$digest" "$variant/READY" || fail "READY checksum mismatch: $variant"
            sha256sum --check --status "$variant/checksums.sha256"
        fi
    done
done
if (( preflight )); then
    printf 'Preflight passed: AIx=%s GPU_REFRESH_BUILD_JOB=%s targets=%s\n' "$expected_sha" "$refresh_job" "$targets"
    exit 0
fi
[[ "${SLURM_JOB_PARTITION:-}" == devel && "${SLURM_CPUS_PER_TASK:-0}" -ge 8 && -n "${SLURM_JOB_ID:-}" ]] || fail 'Requires devel CPU allocation.'
[[ -z "${SLURM_JOB_GPUS:-}" && -z "${SLURM_GPUS:-}" ]] || fail 'GPU reservation forbidden.'
export CMI_DIR="$cmi" USE_SCOREP=0
set +u
source "$cmi/set_env_claix23_cuda12.4.sh"
set -u
[[ "$(command -v mpicxx)" == *OpenMPI/4.1.4-GCC-11.3.0* && "$(command -v gcc)" == *11.3.0* ]] || fail 'Compiler/MPI mismatch.'
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
export CUDA_ROOT="$cuda" CUDA_HOME="$cuda"
base_ld="${LD_LIBRARY_PATH:-}"
[[ "$base_ld" != *stubs* ]] || fail 'CUDA stubs in runtime environment.'
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS='--mpp=mpi --nocompiler --user --io=none --memory=none --thread=none --nocuda'
exec 3>&1 4>&2
for dir in "$targets"; do
    if [[ ! -d "$dir" ]]; then ls -d "$(dirname "$dir")"; mkdir "$dir"; printf '%s\n' "$dir" > "$dir/THESIS_AIX_BUILD"; fi
done
for layout in default affine numa; do
    for mode in plain scorep; do
        variant="$targets/${layout}_${mode}"
        [[ ! -e "$variant/READY" ]] || { printf 'Preserving READY: %s\n' "$variant"; continue; }
        if [[ ! -d "$variant" ]]; then ls -d "$targets"; mkdir "$variant"; printf '%s\n' "$variant" > "$variant/THESIS_AIX_BUILD"; fi
        build="$variant/aix-build"; prefix="$variant/aix-install"
        solver="$mini/solver_cpp/build_gpu_refresh_$mode/terrain_solver"
        dl="$cmi/dl_clients/build-gpu-refresh-$mode/phydll_dl_client"
        exec > >(tee -a "$variant/build.log" >&3) 2>&1
        on=OFF; hw=OFF; numa=OFF
        [[ "$mode" != scorep ]] || on=ON
        [[ "$layout" == default ]] || hw=ON
        [[ "$layout" != numa ]] || numa=ON
        runtime_ld="$prefix/lib:$torch/lib:$cuda/lib64:$cuda/extras/CUPTI/lib64:$hwloc/lib:$base_ld"
        [[ "$mode" != scorep ]] || runtime_ld="$scorep/lib:$papi/lib:$runtime_ld"
        export LD_LIBRARY_PATH="$stubs:$runtime_ld"
        export LIBRARY_PATH="$cuda/lib64:$stubs:$scorep/lib:$papi/lib"
        export CPATH="$scorep/include:$papi/include"
        module list > "$variant/modules.txt" 2>&1
        printf 'layout=%s mode=%s job=%s\nBUILD_LD_LIBRARY_PATH=%s\nRUNTIME_LD_LIBRARY_PATH=%s\n' \
            "$layout" "$mode" "$SLURM_JOB_ID" "$LD_LIBRARY_PATH" "$runtime_ld" > "$variant/environment.txt"
        git -C "$aix" ls-files -z | while IFS= read -r -d '' file; do
            sha256sum "$aix/$file"
        done > "$variant/source.sha256"
        sha256sum "$config" "$solver" "$dl" "$torch/lib/libtorch_cuda.so" >> "$variant/source.sha256"
        cmake -S "$aix" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            "-DCMAKE_INSTALL_PREFIX=$prefix" "-DCMAKE_C_COMPILER=$(command -v mpicc)" \
            "-DCMAKE_CXX_COMPILER=$(command -v mpicxx)" "-DCMAKE_Fortran_COMPILER=$(command -v mpifort)" \
            -DBUILD_TESTS=OFF -DWITH_FORTRAN_INTERFACE=OFF -DWITH_TORCH=ON "-DWITH_SCOREP=$on" \
            "-DWITH_HWLOC=$hw" "-DWITH_NUMA_LOCAL_GROUPS=$numa" "-DHWLOC_ROOT=$hwloc" -DNUMA_ROOT=/usr \
            -DAIX_SKIP_VENV_CREATION=ON "-DAIX_VENV_DIR=$root/python/smartsim_cuda-12" "-DPython3_EXECUTABLE=$python" \
            "-DLIBTORCH_DIR=$torch" "-DTorch_DIR=$torch/share/cmake/Torch" -DTORCH_VERSION=2.4.0 \
            -DUSE_PYTHON_TORCH_CMAKE_PREFIX=OFF -DFETCHCONTENT_FULLY_DISCONNECTED=ON \
            "-DCMAKE_CUDA_COMPILER=$cuda/bin/nvcc" "-DCUDAToolkit_ROOT=$cuda" "-DCUDA_TOOLKIT_ROOT_DIR=$cuda" \
            "-DCUDA_CUDA_LIBRARY=$stubs/libcuda.so" "-DCUDA_cuda_driver_LIBRARY=$stubs/libcuda.so" \
            "-DSCOREP_ROOT_DIR=$scorep" "-DSCOREP_CONFIG_EXECUTABLE=$config" \
            -DSCOREP_ENABLE_COMPILER=OFF -DSCOREP_ENABLE_USER=ON -DSCOREP_MPP_SYSTEM=mpi \
            "-DCMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES=/lib;/lib64;/usr/lib;/usr/lib64;$stubs;$cuda/targets/x86_64-linux/lib/stubs" \
            "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,$stubs" -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF
        for entry in "WITH_SCOREP=$on" "WITH_HWLOC=$hw" "WITH_NUMA_LOCAL_GROUPS=$numa" 'WITH_TORCH=ON' \
            'WITH_FORTRAN_INTERFACE=OFF' 'BUILD_TESTS=OFF' 'NUMA_ROOT=/usr' "HWLOC_ROOT=$hwloc" \
            "CMAKE_INSTALL_PREFIX=$prefix" "CMAKE_HOME_DIRECTORY=$aix" "Python3_EXECUTABLE=$python" \
            "CMAKE_C_COMPILER=$(command -v mpicc)" "CMAKE_CXX_COMPILER=$(command -v mpicxx)" \
            "Torch_DIR=$torch/share/cmake/Torch" 'SCOREP_ENABLE_COMPILER=OFF'; do
            grep -Eq "^${entry%%=*}:[^=]+=${entry#*=}$" "$build/CMakeCache.txt" || fail "Cache mismatch: $entry"
        done
        ! grep -Eq 'finstrument-functions|fplugin|scorep-mpic|libscorep_adapter_compiler' "$build/compile_commands.json" || fail 'Compiler instrumentation present.'
        if [[ "$mode" == plain ]]; then
            ! grep -Eq 'DSCOREP|scorep-' "$build/compile_commands.json" || fail 'Plain compile has Score-P.'
        else
            grep -Fq 'SCOREP_USER_ENABLE' "$build/compile_commands.json" || fail 'Missing user instrumentation.'
        fi
        cmake --build "$build" --target AIxeleratorServiceLib --parallel "$SLURM_CPUS_PER_TASK" --verbose
        ls -d "$variant"
        mkdir -p "$prefix/lib" "$prefix/include"
        cp "$build/src/libAIxeleratorService.so" "$prefix/lib/"
        cp -a "$aix/src/." "$prefix/include/"
        library="$prefix/lib/libAIxeleratorService.so"
        export LD_LIBRARY_PATH="$runtime_ld"
        ldd -r "$library" > "$variant/library.ldd.txt" 2>&1
        ! grep -Eq 'not found|/stubs' "$variant/library.ldd.txt" || fail 'Missing dependency or runtime CUDA stubs.'
        if [[ "$mode" == scorep ]]; then
            ! grep 'undefined symbol' "$variant/library.ldd.txt" | grep -Ev '^undefined symbol: scorep_(subsystems|number_of_subsystems)[[:space:]]+\([^)]*/libscorep_measurement\.so[^)]*\)$' || fail 'Undefined DSO symbol.'
            grep -q 'libscorep_adapter_user' "$variant/library.ldd.txt"
            grep -q 'libscorep_mpp_mpi' "$variant/library.ldd.txt"
        else
            ! grep -Eq 'undefined symbol|libscorep' "$variant/library.ldd.txt" || fail 'Invalid plain DSO.'
        fi
        ! grep -Eq 'libscorep_mpp_mockup|libscorep_adapter_(compiler|cuda)' "$variant/library.ldd.txt" || fail 'Forbidden Score-P adapter.'
        grep -q 'libtorch_cuda' "$variant/library.ldd.txt"
        readelf -d "$library" > "$variant/library.dynamic.txt"
        ! grep -E '(RPATH|RUNPATH|NEEDED).*/stubs' "$variant/library.dynamic.txt" || fail 'Embedded CUDA stubs.'
        cp "$build/CMakeCache.txt" "$variant/aix-cache.txt"
        sha256sum --check --status "$variant/source.sha256"
        [[ "$(git -C "$aix" rev-parse HEAD)" == "$expected_sha" && -z "$(git -C "$aix" status --porcelain --untracked-files=no)" ]] || fail 'AIx changed during build.'
        sha256sum "$variant/source.sha256" "$library" "$build/CMakeCache.txt" "$variant/aix-cache.txt" \
            "$build/compile_commands.json" "$variant/library.ldd.txt" "$variant/library.dynamic.txt" > "$variant/checksums.sha256"
        # Include individual sources/binaries so --resume detects changes, not just a changed manifest.
        while IFS= read -r line; do printf '%s\n' "$line"; done < "$variant/source.sha256" >> "$variant/checksums.sha256"
        sha256sum --check --status "$variant/checksums.sha256"
        digest="$(sha256sum "$variant/checksums.sha256")"; digest="${digest%% *}"
        printf 'format=THESIS_AIX_READY_V1\nlayout=%s\nmode=%s\njob=%s\ngitsha=%s\ngpu_refresh_job=%s\nprefix=%s\nlibrary=%s\ncache=%s\nsolver=%s\ndl_client=%s\nchecksums=%s\nchecksums_sha256=%s\n' \
            "$layout" "$mode" "$SLURM_JOB_ID" "$expected_sha" "$refresh_job" "$prefix" "$library" "$build/CMakeCache.txt" \
            "$solver" "$dl" "$variant/checksums.sha256" "$digest" > "$variant/READY.tmp"
        stat --printf='library_size=%s\nlibrary_mtime=%Y\n' "$library" >> "$variant/READY.tmp"
        printf 'validation=cpu-build-only\nruntime_validation=pending\n' >> "$variant/READY.tmp"
        mv "$variant/READY.tmp" "$variant/READY"
        printf 'READY: %s\n' "$variant/READY"
        exec 1>&3 2>&4
    done
done
