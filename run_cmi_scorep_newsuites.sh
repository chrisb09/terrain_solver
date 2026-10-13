#!/usr/bin/env bash
# Submit the two new Score-P suites over scenarios s3, s5, s6, s7:
#   hh3_g1080 : benchmark_giant_mlp @ 1920x1080, bs 500k, 12 steps (1 warm + 5 steady ML)
#   hh3_w8k   : watercnn           @ 7680x4320, bs 20M,  22 steps (1 warm + 10 steady ML)
# Full framework matrix per scenario:
#   AIx coll/p2p/p2p_fullcredits x layouts default/affine/numa
#   SmartSim c0/c3 (all scenarios) + per_gpu_db (4-GPU scenarios s6/s7) + PhyDLL
# Classes:
#   s3, s6 -> G-class jobs (hh3_wrapper_G.sh, c23g, rwth0792)
#   s5, s7 -> H-class SURROGATE jobs (hh3_wrapper_H_surrogate.sh, c23g+c23g,
#             rwth0792, CPU-side GPUs hidden)
# Env is passed through the environment with --export=ALL (comma-safe).
set -euo pipefail
cd "$(dirname "$0")"

eval "$(grep -m1 '^BASE_EXPORTS=' run_cmi_scorep_het3_suite.sh)"
# BASE_EXPORTS sets model/steps/batch for the 4K watercnn suite; strip the
# suite-specific values and keep only the generic profiling/behaviour flags.
strip() {
    printf '%s\n' "$1" | tr ',' '\n' | grep -vE '^(ALL|MODEL_NAME_ENV|TOTAL_STEPS_ENV|SAVE_EVERY_ENV|TARGET_WIDTH_ENV|TARGET_HEIGHT_ENV|ML_BATCH_SIZE_ENV)=' \
      | paste -sd, -
}
BASE_EXPORTS="$(strip "$BASE_EXPORTS")"

source <(sed -n '/^declare -A AIX_CONFIG_EXPORTS/,/^AIX_CONFIGS=/p' run_cmi_scorep_het3_suite.sh)
source <(sed -n '/^declare -A NON_AIX_CONFIG_EXPORTS/,/^AIX_CONFIGS=/p' run_cmi_scorep_het3_suite.sh)

export_base() {
    local kv old_ifs="$IFS"
    IFS=','
    for kv in $1; do
        [[ "$kv" == "ALL" || -z "$kv" ]] && continue
        export "$kv"
    done
    IFS="$old_ifs"
}

declare -A SUITE_EXPORTS=(
    [hh3_g1080]="MODEL_NAME_ENV=benchmark_giant_mlp,TARGET_WIDTH_ENV=1920,TARGET_HEIGHT_ENV=1080,TOTAL_STEPS_ENV=12,SAVE_EVERY_ENV=12,ML_BATCH_SIZE_ENV=500000"
    [hh3_w8k]="MODEL_NAME_ENV=watercnn,TARGET_WIDTH_ENV=7680,TARGET_HEIGHT_ENV=4320,TOTAL_STEPS_ENV=22,SAVE_EVERY_ENV=22,ML_BATCH_SIZE_ENV=20000000"
)

submit() {  # $1 suite, $2 class (G|H), $3 config, $4 layout, $5 scenarios, $6 wrapper, $7 tag, $8 walltime
    local suite="$1" class="$2" cfg="$3" layout="$4" scen="$5" wrapper="$6" tag="$7" walltime="${8:-00:15:00}"
    if [[ "${cfg}" == smartsim_per_gpu_db ]]; then
        # per-GPU-DB needs DB_NODES=4 -> only meaningful in the 4-GPU scenarios
        if [[ "${class}" == "G" ]]; then scen="6"; else scen="7"; fi
    fi
    if [[ "${cfg}" == aix* ]]; then
        export_base "${AIX_CONFIG_EXPORTS[${cfg}]}"
    else
        export_base "${NON_AIX_CONFIG_EXPORTS[${cfg}]}"
        layout="default"
    fi
    # The 5-stage prediction reads the Score-P h2d_copy/forward/d2h_copy regions
    # from the collective controller; those are only emitted by the synchronous
    # inference path (the default async GPU batch pipeline has no user regions).
    if [[ "${cfg}" == "aix_coll" ]]; then
        export AIX_ASYNC_GPU_PIPELINE=0
    else
        unset AIX_ASYNC_GPU_PIPELINE
    fi
    export_base "${BASE_EXPORTS}"
    export_base "${SUITE_EXPORTS[${suite}]}"
    export HH3_CONFIG_ENV="${cfg}" HH3_LAYOUT_ENV="${layout}" HH3_SUITE_ENV="${suite}" HH3_SCENARIOS_ENV="${scen}"
    jid=$(sbatch --parsable --export=ALL --job-name="${tag}" --time="${walltime}" "${wrapper}")
    echo "submitted ${tag}: ${jid} (scen=${scen}, layout=${layout})"
    echo "${tag} ${jid} ${cfg} ${layout} ${suite} ${scen}" >> "logs/newsuites_jobids.txt"
}

# Optional filter: RUN_CONFIGS_ENV=aix_coll (comma separated) restricts the run.
want_cfg() { [[ -z "${RUN_CONFIGS_ENV:-}" ]] || [[ ",${RUN_CONFIGS_ENV}," == *",${1},"* ]]; }
[[ -z "${RUN_CONFIGS_ENV:-}" ]] && : > logs/newsuites_jobids.txt
for suite in hh3_g1080 hh3_w8k; do
    # walltime: giant@1080p runs ~40 s per ML step on 1-GPU scenarios (2
    # scenarios per job) -> 30 min; watercnn 8K steps are <1 s -> 15 min.
    if [[ "${suite}" == "hh3_g1080" ]]; then WT="00:30:00"; else WT="00:15:00"; fi
    for cfg in aix_coll aix_p2p aix_p2p_fullcredits smartsim_c0 smartsim_c3 smartsim_per_gpu_db phydll_cpp; do
        want_cfg "${cfg}" || continue
        if [[ "${cfg}" == aix* ]]; then
            layouts=(default affine numa)
        else
            layouts=(default)   # SmartSim/PhyDLL have no layout variants
        fi
        for layout in "${layouts[@]}"; do
            # G-class: s3 (0+24c,1g) and s6 (0+96c,4g) under rwth0792
            submit "${suite}" G "${cfg}" "${layout}" "3,6" hh3_wrapper_G.sh "HH3_G_${suite}_${cfg}_${layout}" "${WT}"
            # H-class surrogate: s5 (24+24c,1g) and s7 (96+96c,4g) under rwth0792
            submit "${suite}" H "${cfg}" "${layout}" "5,7" hh3_wrapper_H_surrogate.sh "HH3_Hs_${suite}_${cfg}_${layout}" "${WT}"
        done
    done
done

echo "All submissions recorded in logs/newsuites_jobids.txt"
