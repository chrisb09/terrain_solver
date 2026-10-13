#!/usr/bin/env bash
# Submit the affine H-class reruns (s4,s5) with the scenario list passed through
# the environment (commas survive, unlike a --export string).
set -euo pipefail
cd "$(dirname "$0")"

eval "$(grep -m1 '^BASE_EXPORTS=' run_cmi_scorep_het3_suite.sh)"
source <(sed -n '/^declare -A AIX_CONFIG_EXPORTS/,/^AIX_CONFIGS=/p' run_cmi_scorep_het3_suite.sh)

export_all() {
    local kv old_ifs="$IFS"
    IFS=','
    for kv in $BASE_EXPORTS; do
        [[ "$kv" == "ALL" || -z "$kv" ]] && continue
        export "$kv"
    done
    IFS="$old_ifs"
}

export_all

for cfg in aix_p2p aix_p2p_fullcredits; do
    old_ifs="$IFS"; IFS=','
    for kv in ${AIX_CONFIG_EXPORTS[$cfg]}; do
        [[ -z "$kv" ]] && continue
        export "$kv"
    done
    IFS="$old_ifs"
    export HH3_CONFIG_ENV="$cfg" HH3_LAYOUT_ENV=affine HH3_SCENARIOS_ENV="4,5"
    jid=$(sbatch --parsable --export=ALL --job-name="HH3_H_${cfg}_affine_rerun2" \
                 --account=thes2181 --time=00:15:00 hh3_wrapper_H.sh)
    echo "submitted ${cfg}: ${jid} (HH3_SCENARIOS_ENV=${HH3_SCENARIOS_ENV})"
done
