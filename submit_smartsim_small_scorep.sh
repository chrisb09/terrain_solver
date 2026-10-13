#!/usr/bin/env bash
# Exactly four small SmartSim pilots; reuse a verified completed build, never dependencies.
set -euo pipefail
mini=$(dirname "$(readlink -f "$0")")
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ $# == 3 && ( $1 == --dry-run || $1 == --submit ) ]] || fail 'Usage: submit_smartsim_small_scorep.sh --dry-run|--submit BUILD_JOB_ID CAMPAIGN_ID'
action=$1 build=$2 campaign=$3
[[ $build =~ ^[0-9]+$ && $campaign =~ ^smartsim_small_[A-Za-z0-9_.-]+$ ]] || fail 'Invalid build job ID or smartsim_small_ campaign ID'
solver=$mini/solver_cpp/build_scorep_pilots
dl=$mini/../CPP-ML-Interface/dl_clients/build-scorep-pilots
[[ -r $solver/BUILD_READY && -r $dl/BUILD_READY ]] || fail 'Completed build markers missing'
cmp "$solver/BUILD_READY" "$dl/BUILD_READY" || fail 'Build markers differ'
grep -Eq "^BUILD_READY job=$build completed=" "$solver/BUILD_READY" || fail 'Completed build ID mismatch'
sha256sum --check "$solver/provenance/binaries.sha256" >&2 || fail 'Completed build binaries changed'
manifest=$mini/results/scorep_pilots/$campaign/submissions.csv
[[ -d $mini/logs && -d $mini/results ]] || fail 'logs and results parents must exist'
if [[ $action == --submit ]]; then
  [[ ! -e ${manifest%/*} ]] || fail 'Campaign already exists; choose a unique ID'
  [[ -x $HOME/scripts/swatch ]] || fail 'Mandatory swatch executable missing'
  mkdir -p "${manifest%/*}"
  printf 'config,model,provider,variant,mode,tpq,steps,ml_calls,warm,steady,job_id,manifest_file\n' > "$manifest"
fi
jobids=()
count=0
for cfg in 96c4g 96p96c4g; do
  for mode in profile trace; do
    tpq=2 steps=12 calls=6 steady=5
    if [[ $mode == trace ]]; then tpq=1 steps=8 calls=4 steady=3; fi
    exports="PILOT_CAMPAIGN_ENV=$campaign,PILOT_BUILD_JOB_ID_ENV=$build,PILOT_MODE_ENV=$mode,PILOT_TPQ_ENV=$tpq"
    cmd=(sbatch --parsable --chdir="$mini" --job-name="${campaign}_${cfg}_${mode}_tpq${tpq}"
         --export="$exports" "$mini/scorep_pilot_$cfg.sbatch" giantmlp288 SMARTSIM pergpu)
    if [[ $action == --dry-run ]]; then
      printf '%q ' "${cmd[@]}"; printf '\n'
    else
      reply=$("${cmd[@]}") || fail "sbatch failed after $count submissions; inspect $manifest"
      [[ $reply =~ ^([0-9]+)(\;[A-Za-z0-9_.-]+)?$ ]] || fail "Unexpected sbatch stdout: $reply"
      jid=${BASH_REMATCH[1]}
      run=scorep_pilot_${campaign}_${cfg}_giantmlp288_smartsim_pergpu_${mode}_tpq${tpq}_${jid}
      printf '%s,giantmlp288,SMARTSIM,pergpu,%s,%s,%s,%s,1,%s,%s,%s\n' \
        "$cfg" "$mode" "$tpq" "$steps" "$calls" "$steady" "$jid" "${manifest%/*}/$run/manifest.csv" >> "$manifest"
      jobids+=("$jid")
      printf 'SMARTSIM_SMALL_SUBMITTED config=%s mode=%s tpq=%s job_id=%s\n' "$cfg" "$mode" "$tpq" "$jid"
    fi
    count=$((count + 1))
  done
done
[[ $count == 4 ]] || fail "Expected exactly four pilots, got $count"
if [[ $action == --submit ]]; then
  "$HOME/scripts/swatch" --jobid "${jobids[@]}" || fail "swatch registration failed; jobs already submitted, inspect $manifest"
fi
printf 'Pilots: %s\nSubmission manifest: %s\n' "$count" "$manifest"
