#!/usr/bin/env bash
#SBATCH --job-name=build-colocated-campaign
#SBATCH --account=default
#SBATCH --partition=devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --chdir=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/build_colocated_campaign_%j.out
#SBATCH --error=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/build_colocated_campaign_%j.err

set -Eeuo pipefail
trap 'printf "ERROR line=%s status=%s command=%s\n" "$LINENO" "$?" "$BASH_COMMAND" >&2' ERR
root=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim
mini="$root/mini_app"
cmi="$root/CPP-ML-Interface"
[[ "${SLURM_JOB_PARTITION:-}" == devel && "${SLURM_CPUS_PER_TASK:-0}" == 96 ]]
ls -d "$mini/logs" "$mini/solver_cpp" "$root/tmp" "$root/scorep_runs"
logs="$mini/logs/build_colocated_campaign_${SLURM_JOB_ID}"
mkdir "$logs"
cp "$mini/build_colocated_campaign.sbatch" "$logs/submitted-script.sbatch"
printf 'job=%s node=%s date=%s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -u +%FT%TZ)"

# Never pass an existing directory to build.sh: it unconditionally removes it.
suffix=''
while :; do
    occupied=0
    for variant in direct cmi scorep; do
        dir="$mini/solver_cpp/build_colocated_20261012_${variant}${suffix}"
        [[ ! -e "$dir" && ! -L "$dir" ]] || occupied=1
    done
    (( occupied )) || break
    suffix="_${SLURM_JOB_ID}_${RANDOM}"
done
for variant in direct cmi scorep; do
    printf '%s\t%s\n' "$variant" "$mini/solver_cpp/build_colocated_20261012_${variant}${suffix}"
done | tee "$logs/build-dirs.tsv"

for repo in "$mini" "$cmi" "$root/SmartRedis" "$cmi/extern/SmartRedis"; do
    name="$(basename "$repo")_$(printf '%s' "$repo" | sha256sum | cut -c1-8)"
    git -C "$repo" rev-parse HEAD > "$logs/$name.commit"
    git -C "$repo" status --short > "$logs/$name.status"
    git -C "$repo" diff --binary > "$logs/$name.worktree.diff"
    git -C "$repo" diff --cached --binary > "$logs/$name.index.diff"
done
# Hash actual compilation inputs, including untracked headers, not generated builds.
ls -d "$mini/solver_cpp/include" "$cmi/include" "$cmi/src" "$cmi/scripts" "$cmi/extern/SmartRedis/include"
find "$mini/solver_cpp/include" "$cmi/include" "$cmi/src" "$cmi/scripts" "$cmi/extern/SmartRedis/include" \
    -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum > "$logs/source.sha256"
sha256sum "$mini/solver_cpp/terrain_solver.cpp" "$mini/solver_cpp/CMakeLists.txt" \
    "$mini/build.sh" "$root/install.sh" "$root/env.sh" "$root/set_env_claix23_cuda12.4.sh" \
    "$cmi/CMakeLists.txt" "$cmi/env_scorep.sh" \
    "$cmi/extern/AIxeleratorService/cmake/FindScorep.cmake" >> "$logs/source.sha256"
sha256sum "$cmi/extern/SmartRedis/install/lib64/libsmartredis.so" \
    "$root/SmartRedis/install/lib64/libsmartredis.so" > "$logs/dependencies.sha256"

# Initialize the same GCC/OpenMPI stack once; each variant inherits it in isolation.
export USE_SCOREP=0 SMARTSIM_BUILD_SCOREP=0 SMARTSIM_CREATE_RUNTIME_TARS=0
export SMARTSIM_CREATE_RUNTIME_SYMLINK=0 SMARTSIM_FORCE_LIBTORCH_SYMLINK=0
export SMARTSIM_RUNTIME_ROOT="$root/python"
unset CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS BUILD_ML_MODE MODE
unset SCOREP_WRAPPER_INSTRUMENTER_FLAGS
set +u
source "$root/set_env_claix23_cuda12.4.sh"
source "$root/env.sh"
set -u
[[ "$(command -v gcc)" == *11.3.0* && "$(command -v mpicxx)" == *OpenMPI/4.1.4* ]]
export CAMPAIGN_CC="$(command -v gcc)" CAMPAIGN_CXX="$(command -v g++)"
export CAMPAIGN_MPICC="$(command -v mpicc)" CAMPAIGN_MPICXX="$(command -v mpicxx)"
export CAMPAIGN_CMAKE="$(command -v cmake)"
export CAMPAIGN_TOML="$mini/solver_cpp/build_scorep_cmi_fresh/_deps/tomlplusplus-src"
ls -d "$CAMPAIGN_TOML"
module list > "$logs/modules.txt" 2>&1
gcc --version > "$logs/compiler.txt"
mpicxx --showme >> "$logs/compiler.txt"
mpirun --version >> "$logs/compiler.txt"

# build.sh has no extra-CMake-arguments interface. Intercept only its configure
# invocation to select the campaign's SmartSim-only providers and matching flags.
cmake() {
    if [[ "${1:-}" == -S ]]; then
        local cc="$CAMPAIGN_CC" cxx="$CAMPAIGN_CXX"
        if [[ "$USE_SCOREP" == 1 ]]; then
            cc="$SMARTSIM_SCOREP_ROOT/bin/scorep-mpicc"
            cxx="$SMARTSIM_SCOREP_ROOT/bin/scorep-mpicxx"
        fi
        local args=( "$@" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
            "-DCMAKE_C_COMPILER=$cc" "-DCMAKE_CXX_COMPILER=$cxx"
            "-DMPI_C_COMPILER=$CAMPAIGN_MPICC" "-DMPI_CXX_COMPILER=$CAMPAIGN_MPICXX"
            -DWITH_SMARTSIM=ON -DWITH_AIX=OFF -DWITH_PHYDLL=OFF -DWITH_FORTRAN=OFF
            -DBUILD_TESTING=OFF -DSCOREP_ENABLE_COMPILER=OFF -DSCOREP_ENABLE_USER=ON
            -DCPPML_SCOREP_MPP=mpi -DSCOREP_MPP_SYSTEM=mpi -DSCOREP_IO_SYSTEM=none
            -DSCOREP_MEMORY_SYSTEM=none -DSCOREP_THREADING_SYSTEM=none
            "-DFETCHCONTENT_SOURCE_DIR_TOMLPLUSPLUS=$CAMPAIGN_TOML"
            -DFETCHCONTENT_FULLY_DISCONNECTED=ON )
        printf 'CONFIGURE: '; printf '%q ' "$CAMPAIGN_CMAKE" "${args[@]}"; printf '\n'
        "$CAMPAIGN_CMAKE" "${args[@]}"
    else
        "$CAMPAIGN_CMAKE" "$@"
    fi
}
export -f cmake
export SKIP_TIMESTAMP_CHECK=1 SCOREP_MPP=mpi
export SCOREP_METRIC_PAPI='' SCOREP_METRIC_PAPI_PER_PROCESS=''
export SCOREP_ENABLE_PROFILING=false SCOREP_ENABLE_TRACING=false

cd "$mini"
for variant in direct cmi scorep; do
    (
        set -Eeuo pipefail
        export USE_SCOREP=0 BUILD_ML_MODE=cmi
        [[ "$variant" != direct ]] || export BUILD_ML_MODE=direct_smartsim
        if [[ "$variant" == scorep ]]; then
            export USE_SCOREP=1 CMI_DIR="$cmi"
            export SMARTSIM_PAPI_ROOT="$cmi/tmp/opencode/papi-7.2.0-install"
            export SMARTSIM_SCOREP_ROOT="$cmi/tmp/opencode/scorep-8.4-papi72-install"
            export SCOREP_WRAPPER_INSTRUMENTER_FLAGS='--nocompiler --user --mpp=mpi --io=none --memory=none --thread=none --nocuda'
            source "$cmi/env_scorep.sh"
            scorep-config --version
            scorep-config --nocompiler --user --mpp=mpi --io=none --memory=none --thread=none --nocuda --libs
        fi
        build="build_colocated_20261012_${variant}${suffix}"
        dir="$mini/solver_cpp/$build"
        [[ ! -e "$dir" && ! -L "$dir" ]]
        sha256sum --check "$logs/source.sha256" > "$logs/${variant}.source-before.txt"
        sha256sum --check "$logs/dependencies.sha256" > "$logs/${variant}.dependencies-before.txt"
        printf 'BUILD: USE_SCOREP=%s BUILD_ML_MODE=%s bash ./build.sh %q\n' "$USE_SCOREP" "$BUILD_ML_MODE" "$build"
        # Installer scripts assume optional variables may be unset.
        bash ./build.sh "$build"
        cp "$dir/CMakeCache.txt" "$logs/${variant}.CMakeCache.txt"
        cp "$dir/compile_commands.json" "$logs/${variant}.compile_commands.json"
        grep -Fx 'CMAKE_BUILD_TYPE:STRING=Release' "$dir/CMakeCache.txt"
        expected=OFF; [[ "$variant" != scorep ]] || expected=ON
        grep -Ex "WITH_SCOREP:(BOOL|UNINITIALIZED)=$expected" "$dir/CMakeCache.txt"
        if [[ "$variant" == direct ]]; then
            grep -Fx 'WITH_DIRECT_SMARTSIM:BOOL=ON' "$dir/CMakeCache.txt"
            grep -Fx 'USE_CPP_ML_INTERFACE:BOOL=OFF' "$dir/CMakeCache.txt"
        else
            grep -Fx 'USE_CPP_ML_INTERFACE:BOOL=ON' "$dir/CMakeCache.txt"
            grep -Fx 'WITH_AIX:BOOL=OFF' "$dir/CMakeCache.txt"
        fi
        binary="$dir/terrain_solver"
        [[ -x "$binary" ]]
        sha256sum "$binary" > "$logs/${variant}.binaries.sha256"
        [[ ! -f "$dir/cpp-ml-interface-build/libcpp_ml_interface_library.so" ]] || \
            sha256sum "$dir/cpp-ml-interface-build/libcpp_ml_interface_library.so" >> "$logs/${variant}.binaries.sha256"
        # Driverless build nodes need NVML only for loader validation, not GPU work.
        nvml_stubs=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/lib/stubs
        [[ -r "$nvml_stubs/libnvidia-ml.so.1" ]]
        LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$nvml_stubs" \
            ldd "$binary" > "$logs/${variant}.ldd.txt"
        readelf -d "$binary" > "$logs/${variant}.dynamic.txt"
        if grep -q 'not found' "$logs/${variant}.ldd.txt"; then
            printf 'ERROR: missing runtime dependency for %s\n' "$variant" >&2
            exit 1
        fi
        if [[ "$variant" == scorep ]]; then
            grep -q libscorep_mpp_mpi "$logs/${variant}.ldd.txt"
            grep -q libscorep_adapter_user "$logs/${variant}.ldd.txt"
            # Score-P always links memory management support, even with memory=none.
            if grep -E 'libscorep_(mpp_mockup|adapter_(compiler|cuda|memory_event|pthread|opari|ompt|posix_io))' "$logs/${variant}.ldd.txt"; then
                printf 'ERROR: forbidden Score-P subsystem\n' >&2
                exit 1
            fi
        else
            if grep -q libscorep "$logs/${variant}.ldd.txt"; then
                printf 'ERROR: baseline links Score-P\n' >&2
                exit 1
            fi
        fi
        while read -r library arrow resolved rest; do
            [[ "$arrow" != '=>' || ! -f "$resolved" ]] || sha256sum "$resolved"
        done < "$logs/${variant}.ldd.txt" > "$logs/${variant}.resolved-dependencies.sha256"
        sha256sum --check "$logs/source.sha256" > "$logs/${variant}.source-after.txt"
        sha256sum --check "$logs/dependencies.sha256" > "$logs/${variant}.dependencies-after.txt"
        printf 'BUILD_READY job=%s variant=%s date=%s\n' "$SLURM_JOB_ID" "$variant" "$(date -u +%FT%TZ)" > "$dir/BUILD_READY"
        printf 'BUILD_READY %s\n' "$dir"
    ) 2>&1 | tee "$logs/${variant}.build.log"
done
printf 'export CAMPAIGN_DIRECT_BUILD=%q\nexport CAMPAIGN_CMI_BUILD=%q\nexport CAMPAIGN_SCOREP_BUILD=%q\n' \
    "build_colocated_20261012_direct${suffix}" "build_colocated_20261012_cmi${suffix}" \
    "build_colocated_20261012_scorep${suffix}" > "$logs/build-dirs.env"
printf 'All three builds validated. Logs: %s\n' "$logs"
