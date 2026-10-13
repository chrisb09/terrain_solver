#!/usr/bin/env bash
# BASH_ENV checks the actual rank wrapper after launcher/module setup.
if [[ "$0" == */solver_wrapper.sh || "$0" == solver_wrapper.sh ]]; then
    if [[ "${SKIP_PAPI_METRICS:-}" != 1 || -n "${SCOREP_METRIC_PAPI:-}" || -n "${SCOREP_METRIC_PAPI_PER_PROCESS:-}" || "${CUDA_LAUNCH_BLOCKING:-}" != 1 ]]; then
        printf 'Campaign environment mismatch on rank %s\n' "${SLURM_PROCID:-unknown}" >&2
        exit 1
    fi
    printf 'rank=%s SKIP_PAPI_METRICS=%s SCOREP_METRIC_PAPI=%s CUDA_LAUNCH_BLOCKING=%s\n' \
        "${SLURM_PROCID:?}" "$SKIP_PAPI_METRICS" "${SCOREP_METRIC_PAPI:-}" "$CUDA_LAUNCH_BLOCKING" \
        > "${CAMPAIGN_ENV_DIR:?}/rank_${SLURM_PROCID}.txt"
fi
