#!/usr/bin/env bash
# No submission unless --submit is explicitly requested.
set -euo pipefail
mini=$(dirname "$(readlink -f "$0")")
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ $# == 3 && ( $1 == --dry-run || $1 == --submit ) ]] || fail 'Usage: submit_scorep_pilots.sh --dry-run|--submit BUILD_JOB_ID CAMPAIGN_ID'
mode=$1 build=$2 campaign=$3
[[ $build =~ ^[0-9]+$ && $campaign =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail 'Invalid build job ID or campaign ID'
manifest="$mini/results/scorep_pilots/$campaign/submissions.csv"
[[ -d $mini/logs && -d $mini/results ]] || fail 'logs and results parent directories must exist'
if [[ $mode == --submit ]]; then
  [[ ! -e ${manifest%/*} ]] || fail 'Campaign already exists; choose a unique ID'
  [[ -x $HOME/scripts/swatch ]] || fail 'Mandatory swatch executable missing'
  mkdir -p "${manifest%/*}"
  [[ -d ${manifest%/*} ]] || fail 'Submission manifest parent missing'
  printf 'config,model,provider,variant,job_id\n' > "$manifest"
fi
count=0
jobids=()
dependency=(--dependency="afterok:$build")
if [[ -r $mini/solver_cpp/build_scorep_pilots/BUILD_READY ]] &&
   grep -Eq "^BUILD_READY job=$build completed=" "$mini/solver_cpp/build_scorep_pilots/BUILD_READY"; then
  dependency=()
fi
while read -r cfg model provider variant; do
  cmd=(sbatch --parsable "${dependency[@]}" --chdir="$mini"
       --export="PILOT_CAMPAIGN_ENV=$campaign,PILOT_BUILD_JOB_ID_ENV=$build"
       "$mini/scorep_pilot_$cfg.sbatch" "$model" "$provider" "$variant")
  if [[ $mode == --dry-run ]]; then
    printf '%q ' "${cmd[@]}"; printf '\n'
  else
    # Keep stderr separate: --parsable stdout alone contains jobid[;cluster].
    reply=$("${cmd[@]}") || fail "sbatch failed after $count submissions; inspect $manifest"
    [[ $reply =~ ^([0-9]+)(\;[A-Za-z0-9_.-]+)?$ ]] || fail "Unexpected sbatch stdout: $reply"
    jid=${BASH_REMATCH[1]}
    [[ -d ${manifest%/*} ]] || fail 'Manifest parent disappeared'
    printf '%s,%s,%s,%s,%s\n' "$cfg" "$model" "$provider" "$variant" "$jid" >> "$manifest"
    jobids+=("$jid")
    printf 'PILOT_SUBMITTED %s %s %s %s job_id=%s\n' "$cfg" "$model" "$provider" "$variant" "$jid"
  fi
  count=$((count + 1))
done < <(zsh "$mini/retest_scorep_pilot.zsh" --list)
[[ $count == 22 ]] || fail "Expected 22 pilots, got $count"
if [[ $mode == --submit ]]; then
  "$HOME/scripts/swatch" --jobid "${jobids[@]}" || fail "swatch registration failed (pilots already submitted; inspect $manifest)"
fi
printf 'Pilots: %s\nSubmission manifest: %s\nPer-pilot manifests: %s/*/manifest.csv\n' "$count" "$manifest" "${manifest%/*}"
