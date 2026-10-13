#!/usr/bin/env bash
# Dry-run by default. Only --submit calls sbatch. No analysis/automatic follow-on submission.
set -euo pipefail
mini=/rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
[[ $# == 0 || ( $# == 1 && $1 == --submit ) ]] || exit 1
campaign=${RETEST_CAMPAIGN:-phydll_master_$(date -u +%Y%m%dT%H%M%SZ)}
[[ $campaign =~ ^[A-Za-z0-9_-]+$ ]] || exit 1
bash "$mini/build_terrain_phydll_retest.sbatch" --preflight
if [[ ${1:-} != --submit ]]; then
    printf 'Dry-run only. Campaign: %s\n' "$campaign"
    printf 'Build: sbatch --parsable --export=NONE %s/build_terrain_phydll_retest.sbatch\n' "$mini"
    printf 'Performance: sbatch --parsable --dependency=afterok:BUILD_ID --export=MODE=performance,RETEST_CAMPAIGN=%s %s/terrain_phydll_retest.sbatch\n' "$campaign" "$mini"
    printf 'Memory: sbatch --parsable --dependency=afterok:BUILD_ID:PERFORMANCE_ID --export=MODE=memory,RETEST_CAMPAIGN=%s %s/terrain_phydll_retest.sbatch\n' "$campaign" "$mini"
    exit 0
fi
# Avoid inherited SBATCH options silently overriding resource/array methodology.
while read -r name; do [[ $name != SBATCH_* ]] || unset "$name"; done < <(compgen -e)
build=$(sbatch --parsable --export=NONE "$mini/build_terrain_phydll_retest.sbatch")
build=${build%%;*}
[[ $build =~ ^[0-9]+$ ]] || exit 1
performance=$(sbatch --parsable --dependency="afterok:$build" \
    --export="MODE=performance,RETEST_CAMPAIGN=$campaign" "$mini/terrain_phydll_retest.sbatch")
performance=${performance%%;*}
[[ $performance =~ ^[0-9]+$ ]] || exit 1
memory=$(sbatch --parsable --dependency="afterok:$build:$performance" \
    --export="MODE=memory,RETEST_CAMPAIGN=$campaign" "$mini/terrain_phydll_retest.sbatch")
printf 'campaign=%s build=%s performance=%s memory=%s\n' "$campaign" "$build" "$performance" "$memory"
