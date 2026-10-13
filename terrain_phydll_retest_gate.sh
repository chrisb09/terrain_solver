# Sourced only by terrain_cpu_final.sbatch after canonical module setup.
expected=$MINI_APP_DIR/solver_cpp/build_terrain_phydll_retest
[[ $TERRAIN_RETEST_BUILD == "$expected" && -d $expected && ! -L $expected && -f $expected/BUILD_READY ]] || exit 1
[[ $CONFIGS == PHYDLL_CPP && ( $TERRAIN_READINESS == false || $TERRAIN_READINESS == true ) ]] || exit 1
[[ $TERRAIN_PHYDLL_CONFIG == "$MINI_APP_DIR/config_terrain_phydll_${TERRAIN_READINESS}.toml" ]] || exit 1
grep -Fxq "solver_readiness_wait = $TERRAIN_READINESS" "$TERRAIN_PHYDLL_CONFIG"
# The marker attests the manifests; the manifests attest sources and artifacts.
{
    read -r commit
    read -r job
    [[ $commit =~ ^CMI_COMMIT=[0-9a-f]{40}$ && $job =~ ^BUILD_JOB=[0-9]+$ ]] || exit 1
    while IFS= read -r hash; do printf '%s\n' "$hash"; done
} < "$expected/BUILD_READY" | sha256sum --check
sha256sum --check "$expected/provenance/source.sha256"
sha256sum --check "$expected/provenance/binaries.sha256"
export SOLVER_BIN=$expected/terrain_solver PHYDLL_DL_CLIENT=$expected/dl-client/phydll_dl_client
export SLURM_EXPORT_ENV=ALL
export LD_LIBRARY_PATH="$expected/phydll/lib:$expected/dl-client:$expected/cpp-ml-interface-build:$CPP_DIR/extern/libtorch/lib:$LD_LIBRARY_PATH"
export APPROVED_PHYDLL_SHA256=$(sha256sum "$expected/phydll/lib/libphydll.so")
APPROVED_PHYDLL_SHA256=${APPROVED_PHYDLL_SHA256%% *}
for binary in "$SOLVER_BIN" "$PHYDLL_DL_CLIENT"; do
    linkage=$(ldd -r "$binary" 2>&1)
    [[ $linkage != *'not found'* && $linkage != *'undefined symbol'* && $linkage != *libscorep_mpp_mockup* ]] || exit 1
    [[ $linkage == *libscorep_mpp_mpi* && $linkage == *libscorep_adapter_user* ]] || exit 1
    [[ $linkage == *"libphydll.so => $expected/phydll/lib/libphydll.so"* ]] || exit 1
done
