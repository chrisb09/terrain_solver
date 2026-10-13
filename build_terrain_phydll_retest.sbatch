#!/usr/bin/env bash
#SBATCH --job-name=build-terrain-phydll
#SBATCH --account=rwth2150
#SBATCH --partition=c23ms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app/logs/build_terrain_phydll_%j.log
set -Eeuo pipefail
mini=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
cmi=${mini%/mini_app}/CPP-ML-Interface
build=$mini/solver_cpp/build_terrain_phydll_retest
dl=$build/dl-client
phy=$build/phydll
provenance=$build/provenance
scorep=$cmi/tmp/opencode/scorep-8.4-papi72-install
papi=$cmi/tmp/opencode/papi-7.2.0-install
torch=$cmi/extern/libtorch
cuda=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0
stubs=$cuda/lib/stubs
hdf5=/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/HDF5/1.12.2-gompi-2022a
python=${mini%/mini_app}/python/smartsim_cuda-12/bin/python
toml=$mini/solver_cpp/build_scorep_cmi_fresh/_deps/tomlplusplus-src
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ $# == 0 || ( $# == 1 && $1 == --preflight ) ]] || fail 'Usage: [--preflight]'
remote=$(git -C "$cmi" ls-remote origin refs/heads/master)
revision=${remote%%[[:space:]]*}
[[ $revision =~ ^[0-9a-f]{40}$ && $(git -C "$cmi" rev-parse HEAD) == "$revision" ]] || fail 'Checkout must equal actual remote origin/master; do not change shared checkout automatically.'
[[ -z $(git -C "$cmi" status --porcelain --untracked-files=no) ]] || fail 'CMI tracked source is dirty.'
[[ -z $(git -C "$cmi/extern/phydll" status --porcelain --untracked-files=no) ]] || fail 'PhyDLL tracked source is dirty.'
git -C "$cmi" merge-base --is-ancestor 7952128 HEAD
grep -Fq 'phydll_get_field_view(&buffer_ptr, label)' "$cmi/dl_clients/phydll_dl_runtime.cpp"
grep -Fq 'frame.data.swap(combined_data_)' "$cmi/dl_clients/phydll_dl_runtime.cpp"
for dir in "$mini/logs" "$mini/solver_cpp" "$toml"; do [[ -d $dir && ! -L $dir ]] || fail "Missing/unsafe parent: $dir"; done
for file in "$scorep/bin/scorep-mpicxx" "$python" "$torch/lib/libtorch.so" "$stubs/libcuda.so.1" "$stubs/libnvidia-ml.so.1" "$hdf5/bin/h5pcc"; do
    [[ -r $file ]] || fail "Missing dependency: $file"
done
for file in "$mini/input/prep_watercnn_4k.h5" "$mini/input/prep_giant_240x192.h5" \
    "$mini/train_models/model_a/watercnn_cpu.pt" "$mini/train_models/model_a/giant_cpu.pt" \
    "$mini/config_terrain_phydll_false.toml" "$mini/config_terrain_phydll_true.toml" \
    "$scorep/bin/cube_dump" "$mini/terrain_memory_sampler.py"; do
    [[ -r $file ]] || fail "Missing retest input/tool: $file"
done
# Never resume or mutate a published tree. A validated ready tree is read-only reuse.
if [[ -e $build || -L $build ]]; then
    [[ -d $build && ! -L $build && -f $build/BUILD_READY ]] || fail 'Existing unfinished/unsafe build; choose a new dedicated path after investigation.'
    grep -Fxq "CMI_COMMIT=$revision" "$build/BUILD_READY" || fail 'Ready build is not current master.'
    sha256sum --check "$provenance/source.sha256"
    sha256sum --check "$provenance/binaries.sha256"
    printf 'Validated ready build, preserved without recompilation: %s\n' "$build"
    exit 0
fi
if [[ ${1:-} == --preflight ]]; then
    printf 'Preflight passed: origin/master=%s dedicated build=%s\n' "$revision" "$build"
    exit 0
fi
[[ ${SLURM_JOB_PARTITION:-} == c23ms && ${SLURM_CPUS_PER_TASK:-0} -ge 8 && -z ${SLURM_JOB_GPUS:-} ]] || fail 'Requires CPU-only c23ms allocation.'
unset PYTHONHOME PYTHONPATH PYTHONOPTIMIZE LD_PRELOAD CC CXX CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
export CMI_DIR=$cmi USE_SCOREP=1 SMARTSIM_SCOREP_ROOT=$scorep SMARTSIM_PAPI_ROOT=$papi
set +u
source "$cmi/set_env_claix23_cuda12.4.sh"
set -u
source "$cmi/env_scorep.sh"
export SCOREP_WRAPPER_INSTRUMENTER_FLAGS='--mpp=mpi --nocompiler --user --io=none --memory=none --thread=none --nocuda'
export CUDA_ROOT=$cuda CUDA_HOME=$cuda LIBCLANG_PATH=${EBROOTCLANG:?}/lib
export LD_LIBRARY_PATH="$stubs:$phy/lib:$dl:$torch/lib:$scorep/lib:$papi/lib:$cmi/extern/SmartRedis/install/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$stubs:$cuda/lib64:${LIBRARY_PATH:-}"
[[ $(command -v mpicxx) == *OpenMPI/4.1.4* && $(command -v gcc) == *11.3.0* ]] || fail 'Compiler/MPI mismatch.'
"$python" -c 'import clang.cindex; clang.cindex.Index.create()'
ls -d "$mini/solver_cpp"
mkdir "$build"
ls -d "$build"
mkdir -p "$provenance" "$phy/lib" "$phy/include"
exec > >(tee "$provenance/build.log") 2>&1
module list 2> "$provenance/modules.txt"
printf '%s\n' "$revision" > "$provenance/cmi-commit.txt"
git -C "$cmi/extern/phydll" rev-parse HEAD > "$provenance/phydll-commit.txt"
git -C "$mini" diff --binary -- solver_cpp > "$provenance/mini-solver.diff"
# Hash the exact source set before and after compilation to detect concurrent edits.
find "$mini/solver_cpp/include" "$cmi/include" "$cmi/src" "$cmi/scripts" "$cmi/extern/phydll/src/core" -type f -print0 |
    sort -z | xargs -0 sha256sum > "$provenance/source.sha256"
sha256sum "$mini/solver_cpp/CMakeLists.txt" "$mini/solver_cpp/terrain_solver.cpp" \
    "$cmi/CMakeLists.txt" "$cmi/dl_clients/CMakeLists.txt" "$cmi/dl_clients/dl_client.cpp" \
    "$cmi/dl_clients/phydll_dl_runtime.cpp" "$cmi/dl_clients/phydll_dl_runtime.hpp" \
    "$cmi/extern/phydll/Makefile" "$mini/build_terrain_phydll_retest.sbatch" \
    "$cmi/env_scorep.sh" "$cmi/set_env_claix23_cuda12.4.sh" \
    "$cmi/extern/AIxeleratorService/cmake/FindScorep.cmake" \
    "$torch/lib/libtorch.so" >> "$provenance/source.sha256"
tar -czf "$provenance/source.tar.gz" -C "${mini%/mini_app}" mini_app/solver_cpp/include \
    mini_app/solver_cpp/CMakeLists.txt mini_app/solver_cpp/terrain_solver.cpp \
    CPP-ML-Interface/include CPP-ML-Interface/src CPP-ML-Interface/scripts \
    CPP-ML-Interface/dl_clients/dl_client.cpp CPP-ML-Interface/dl_clients/phydll_dl_runtime.cpp \
    CPP-ML-Interface/dl_clients/phydll_dl_runtime.hpp CPP-ML-Interface/dl_clients/CMakeLists.txt \
    CPP-ML-Interface/CMakeLists.txt CPP-ML-Interface/extern/phydll/src/core \
    CPP-ML-Interface/extern/phydll/Makefile mini_app/build_terrain_phydll_retest.sbatch \
    CPP-ML-Interface/env_scorep.sh CPP-ML-Interface/set_env_claix23_cuda12.4.sh \
    CPP-ML-Interface/extern/AIxeleratorService/cmake/FindScorep.cmake
make -C "$cmi/extern/phydll" compile "BUILD=$phy" "CC=$(command -v mpicc)" 'CFLAGS=-O2 -g -Wall -Wextra -std=c99'
nm -D --defined-only "$phy/lib/libphydll.so" | grep ' phydll_get_field_view$'
common=(-DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    "-DCMAKE_C_COMPILER=$scorep/bin/scorep-mpicc" "-DCMAKE_CXX_COMPILER=$scorep/bin/scorep-mpicxx"
    -DWITH_SCOREP=ON -DCPPML_SCOREP_MPP=mpi -DSCOREP_MPP_SYSTEM=mpi
    -DSCOREP_ENABLE_COMPILER=OFF -DSCOREP_ENABLE_USER=ON "-DSCOREP_ROOT_DIR=$scorep"
    "-DSCOREP_CONFIG_EXECUTABLE=$scorep/bin/scorep-config" "-DSCOREP_INFO_EXECUTABLE=$scorep/bin/scorep-info"
    "-DPHYDLL_BUILD_DIR=$phy" "-DLIBTORCH_DIR=$torch" "-DTorch_DIR=$torch/share/cmake/Torch"
    -DTORCH_VERSION=2.4.0 -DPHYDLL_DL_WITH_TORCH=ON
    "-DCUDA_TOOLKIT_ROOT_DIR=$cuda" "-DCUDAToolkit_ROOT=$cuda" "-DCMAKE_CUDA_COMPILER=$cuda/bin/nvcc"
    "-DCUDA_CUDA_LIBRARY=$stubs/libcuda.so" "-DCUDA_cuda_driver_LIBRARY=$stubs/libcuda.so"
    "-DCMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES=/lib;/lib64;/usr/lib;/usr/lib64;$stubs"
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=OFF "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,$stubs")
# PhyDLL-only provider avoids CMI's AIx git-stamp bootstrap and shared installs.
cmake -S "$mini/solver_cpp" -B "$build" "${common[@]}" -DUSE_CPP_ML_INTERFACE=ON \
    -DWITH_AIX=OFF -DWITH_SMARTSIM=OFF -DWITH_PHYDLL=ON -DWITH_FORTRAN=OFF -DBUILD_TESTING=OFF \
    "-DSMARTSIM_PYTHON=$python" "-DHDF5_ROOT=$hdf5" "-DHDF5_C_COMPILER_EXECUTABLE=$hdf5/bin/h5pcc" \
    "-DFETCHCONTENT_SOURCE_DIR_TOMLPLUSPLUS=$toml" -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake -S "$cmi/dl_clients" -B "$dl" "${common[@]}"
grep -Fq PHYDLL_DL_USE_TORCH "$dl/compile_commands.json"
cmake --build "$build" --target terrain_solver --parallel "$SLURM_CPUS_PER_TASK" --verbose
cmake --build "$dl" --target phydll_dl_client --parallel "$SLURM_CPUS_PER_TASK" --verbose
cmp "$build/generated_registry.hpp" "$build/cpp-ml-interface-build/generated_registry.hpp"
grep -Fq solver_readiness_wait "$build/generated_registry.hpp"
for binary in "$build/terrain_solver" "$dl/phydll_dl_client"; do
    name=$(basename "$binary")
    ldd -r "$binary" > "$provenance/$name.ldd.txt" 2>&1
    ! grep -Eq 'not found|undefined symbol|libscorep_mpp_mockup' "$provenance/$name.ldd.txt"
    grep -q libscorep_mpp_mpi "$provenance/$name.ldd.txt"
    grep -q libscorep_adapter_user "$provenance/$name.ldd.txt"
    readelf --dyn-syms --wide "$binary" | grep SCOREP_User_RegionBegin
done
strings "$dl/phydll_dl_client" | grep -F dl_torch_forward
for binary in "$build/terrain_solver" "$dl/phydll_dl_client" "$dl/libphydll_dl_runtime.so" "$build/cpp-ml-interface-build/libcpp_ml_interface_library.so"; do
    readelf -d "$binary" > "$provenance/$(basename "$binary").dynamic.txt"
    ! grep -E 'RPATH|RUNPATH' "$provenance/$(basename "$binary").dynamic.txt" | grep -q /stubs
done
sha256sum --check "$provenance/source.sha256"
[[ $(git -C "$cmi" rev-parse HEAD) == "$revision" ]] || fail 'CMI revision changed during build.'
sha256sum "$build/terrain_solver" "$dl/phydll_dl_client" "$dl/libphydll_dl_runtime.so" \
    "$build/cpp-ml-interface-build/libcpp_ml_interface_library.so" "$phy/lib/libphydll.so" \
    "$build/generated_registry.hpp" "$build/CMakeCache.txt" "$dl/CMakeCache.txt" > "$provenance/binaries.sha256"
cp "$build/CMakeCache.txt" "$provenance/solver-cache.txt"
cp "$dl/CMakeCache.txt" "$provenance/dl-cache.txt"
printf 'CMI_COMMIT=%s\nBUILD_JOB=%s\n' "$revision" "$SLURM_JOB_ID" > "$build/BUILD_READY.tmp"
sha256sum "$provenance/source.sha256" "$provenance/binaries.sha256" >> "$build/BUILD_READY.tmp"
mv "$build/BUILD_READY.tmp" "$build/BUILD_READY"
printf 'Ready (build-only, CPU runtime/profile validation pending): %s\n' "$build"
