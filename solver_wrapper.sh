#!/bin/bash
# Unique Score-P directory per rank to avoid write conflicts in MPMD mode
if [[ -n "${OMPI_COMM_WORLD_RANK:-}" ]]; then
  export SCOREP_EXPERIMENT_DIRECTORY="${SCOREP_EXPERIMENT_DIRECTORY}_rank_${OMPI_COMM_WORLD_RANK}"
elif [[ -n "${PMIX_RANK:-}" ]]; then
  export SCOREP_EXPERIMENT_DIRECTORY="${SCOREP_EXPERIMENT_DIRECTORY}_rank_${PMIX_RANK}"
fi
exec ./solver_cpp/build/terrain_solver "$@"
