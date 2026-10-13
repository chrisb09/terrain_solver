#!/bin/bash
# Sourced by timing mode; executable --preflight performs read-only allocation-free checks.
if [[ ${1:-} == --preflight ]]; then
    set -eo pipefail
    MINI_APP_DIR=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
    CPP_DIR=${MINI_APP_DIR%/mini_app}/CPP-ML-Interface
    unset USE_SCOREP PYTHONHOME PYTHONPATH PYTHONOPTIMIZE LD_PRELOAD
    source "${MINI_APP_DIR%/mini_app}/set_env_claix23_cuda12.4.sh"
fi
expected=$MINI_APP_DIR/solver_cpp/build_gpu_refresh_plain
provenance=$expected/provenance-4938224
dl=$CPP_DIR/dl_clients/build-gpu-refresh-plain
phy=$CPP_DIR/extern/phydll/build-gpu-refresh
aix=$CPP_DIR/extern/AIxeleratorService/INSTALL
[[ -d $expected && ! -L $expected && -f $expected/BUILD_READY ]] || exit 1
grep -Fxq 'BUILD_READY job=4938224 mode=plain completed=2026-10-12T06:16:48Z validation=cpu-build-only runtime_validation=pending' "$expected/BUILD_READY"
cmp "$expected/BUILD_READY" "$dl/BUILD_READY"
{ read -r marker; while IFS= read -r hash; do printf '%s\n' "$hash"; done; } < "$expected/BUILD_READY" | sha256sum --check --quiet
sha256sum --check --quiet "$provenance/source.sha256"
sha256sum --check --quiet "$provenance/binaries.sha256"
for tree in "$expected" "$dl"; do
    grep -Fxq 'WITH_SCOREP:BOOL=OFF' "$tree/CMakeCache.txt"
    grep -Eq '^CMAKE_CXX_COMPILER:[^=]+=.*OpenMPI/4.1.4-GCC-11.3.0/bin/mpicxx$' "$tree/CMakeCache.txt"
    ! grep -Eq 'USE_SCOREP|scorep-mpi|finstrument-functions|fplugin=' "$tree/compile_commands.json" || exit 1
done
for provider in WITH_AIX WITH_PHYDLL WITH_SMARTSIM; do
    grep -Fxq "$provider:BOOL=ON" "$expected/CMakeCache.txt"
done
grep -Fq solver_readiness_wait "$expected/generated_registry.hpp"
grep -Fq PHYDLL_DL_USE_TORCH "$dl/compile_commands.json"
export TERRAIN_PLAIN_BUILD=$expected TERRAIN_PLAIN_PROVENANCE=$provenance
export SOLVER_BIN=$expected/terrain_solver PHYDLL_DL_CLIENT=$dl/phydll_dl_client
export SLURM_EXPORT_ENV=ALL
export LD_LIBRARY_PATH="/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/lib/stubs:$phy/lib:$dl:$expected/cpp-ml-interface-build:$aix/lib:$CPP_DIR/extern/libtorch/lib:$CPP_DIR/extern/SmartRedis/install/lib64:${LD_LIBRARY_PATH:-}"
for binary in "$SOLVER_BIN" "$PHYDLL_DL_CLIENT"; do
    linkage=$(ldd -r "$binary" 2>&1)
    [[ $linkage != *'not found'* && $linkage != *'undefined symbol'* && $linkage != *libscorep* ]] || exit 1
    [[ $linkage == *"libphydll.so => $phy/lib/libphydll.so"* ]] || exit 1
done
linkage=$(ldd "$SOLVER_BIN")
[[ $linkage == *"libAIxeleratorService.so => $aix/lib/libAIxeleratorService.so"* ]] || exit 1
for binary in "$SOLVER_BIN" "$PHYDLL_DL_CLIENT" "$expected/cpp-ml-interface-build/libcpp_ml_interface_library.so" \
    "$dl/libphydll_dl_runtime.so" "$aix/lib/libAIxeleratorService.so" "$phy/lib/libphydll.so"; do
    symbols=$(nm -a "$binary")
    [[ $symbols != *SCOREP_* && $symbols != *scorep_* && $symbols != *__cyg_profile* ]] || exit 1
done
export APPROVED_PHYDLL_SHA256=3abfc405906bac756ff9f20cd9627f6a4441f46d397a09e05de566696eb76b7c
if [[ ${CONFIGS:-} == *PHYDLL_CPP* ]]; then
    [[ ${TERRAIN_READINESS:-} == false || ${TERRAIN_READINESS:-} == true ]] || exit 1
    export TERRAIN_PHYDLL_CONFIG=$MINI_APP_DIR/config_terrain_phydll_${TERRAIN_READINESS}.toml
    grep -Fxq "solver_readiness_wait = $TERRAIN_READINESS" "$TERRAIN_PHYDLL_CONFIG"
fi
printf 'Validated actual plain build (no Score-P linkage/symbols): %s\n' "$expected"
