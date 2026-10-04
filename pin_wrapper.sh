#!/bin/bash
# Paired core pin wrapper for 96-core Sapphirerapids nodes.
# Maps solver ranks 0..95 to cores 0..95 and DL ranks 96..191 to cores 0..95.
if [[ -n "${OMPI_COMM_WORLD_RANK:-}" ]]; then
    target_core=$(( OMPI_COMM_WORLD_RANK % 96 ))
    taskset -p -c "$target_core" $$ >/dev/null 2>&1 || true
    if [[ "$OMPI_COMM_WORLD_RANK" -eq 0 || "$OMPI_COMM_WORLD_RANK" -eq 1 || \
          "$OMPI_COMM_WORLD_RANK" -eq 95 || "$OMPI_COMM_WORLD_RANK" -eq 96 || \
          "$OMPI_COMM_WORLD_RANK" -eq 97 || "$OMPI_COMM_WORLD_RANK" -eq 191 ]]; then
        actual_aff=$(taskset -p -c $$ 2>/dev/null | awk -F': ' '{print $2}')
        echo "[PIN_WRAPPER] rank $OMPI_COMM_WORLD_RANK target_core=$target_core actual_affinity=$actual_aff host=$(hostname)" >&2
    fi
fi
exec "$@"
