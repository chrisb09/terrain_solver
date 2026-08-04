#!/usr/bin/env zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
MODE="${1:-both}"

build_mode() {
  local mode="$1"
  local use_cpp build_dir
  case "${mode}" in
    direct)
      use_cpp=OFF
      build_dir=build_key_balance_direct
      ;;
    cpp)
      use_cpp=ON
      build_dir=build_key_balance_cpp
      ;;
    *)
      print -u2 "Usage: $0 [direct|cpp|both]"
      return 2
      ;;
  esac

  print "Building ${mode} SmartSim benchmark in solver_cpp/${build_dir} without Score-P"
  (
    cd "${SCRIPT_DIR}"
    USE_SCOREP=0 USE_CPP_ML_INTERFACE="${use_cpp}" ./build.sh "${build_dir}"
  )
}

case "${MODE}" in
  direct|cpp)
    build_mode "${MODE}"
    ;;
  both)
    build_mode direct
    build_mode cpp
    ;;
  *)
    print -u2 "Usage: $0 [direct|cpp|both]"
    exit 2
    ;;
esac
