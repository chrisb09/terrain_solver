#!/bin/zsh
# Narrow adapter for fixed drivers: preserve fresh runtime libraries and outputs.
set -euo pipefail
(( $# == 1 )) || exit 2
driver=$1
[[ ${driver:t} == 5r_driver_single.zsh || ${driver:t} == 5r_driver_hetsurrogate.zsh ]] || exit 2
builtin source "${GPU_REFRESH_BUILD_ENV:?}"
export SMARTSIM_CREATE_RUNTIME_TARS=0
[[ $SOLVER_BIN == */build_gpu_refresh_*/terrain_solver && $PHYDLL_DL_CLIENT == */build-gpu-refresh-*/phydll_dl_client ]] || exit 2
source() {
  builtin source "$@"
  export LD_LIBRARY_PATH="$GPU_REFRESH_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
}
realpath() {
  if [[ $# == 1 && $1 == */extern/phydll/build/lib ]]; then
    print -r -- "$PHYDLL_BUILD_DIR/lib"
  else
    command realpath "$@"
  fi
}
TRAPDEBUG() {
  # Match executed commands, not a compound if containing an unused build branch.
  if [[ $ZSH_DEBUG_CMD == 'echo '*'Compiling solver...'* || $ZSH_DEBUG_CMD == 'cmake -S '* || $ZSH_DEBUG_CMD == 'cmake --build '* || $ZSH_DEBUG_CMD == 'rm -rf "solver_cpp/'* ]]; then
    print -u2 -- 'ERROR: GPU refresh forbids driver compilation'
    exit 90
  fi
  if [[ $ZSH_DEBUG_CMD == 'PREP_H5='* ]]; then
    EXTERNAL_DIR=${GPU_REFRESH_OUTPUT_ROOT:?}
  fi
}
builtin source "$driver"
