#!/usr/bin/env bash
set -euo pipefail
ROOT="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
phase=${1:?Usage: bash submit_colocated_campaign.sh smoke|pilot|production cpu|gpu [--submit]}
platform=${2:?platform required}
[[ $# == 2 || ( $# == 3 && $3 == --submit ) ]] || exit 2
case "$phase" in smoke) reps=1; cpu_minutes=45; gpu_minutes=60;; pilot) reps=1; cpu_minutes=60; gpu_minutes=90;; production) reps=5; cpu_minutes=90; gpu_minutes=120;; *) exit 2;; esac
case "$platform" in cpu) partition=c23ms; account=rwth2150; minutes=$cpu_minutes; gpu_args=();; gpu) partition=c23g; account=default; minutes=$gpu_minutes; gpu_args=(--gres=gpu:4);; *) exit 2;; esac
export CAMPAIGN_ID="${CAMPAIGN_ID:-colocated_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
[[ "$CAMPAIGN_ID" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
# Explicit fresh directories are required; no fallback to historical builds.
if [[ -n "${CAMPAIGN_BUILD_JOB:-}" ]]; then
    [[ "$CAMPAIGN_BUILD_JOB" =~ ^[0-9]+$ ]] || exit 2
    export CAMPAIGN_BUILD_JOB
    CAMPAIGN_DEPENDENCY="afterok:$CAMPAIGN_BUILD_JOB"
    # The validated build manifest is loaded by the allocation, not guessed here.
    export CAMPAIGN_DIRECT_BUILD=pending CAMPAIGN_CMI_BUILD=pending CAMPAIGN_SCOREP_BUILD=pending
fi
: "${CAMPAIGN_DIRECT_BUILD:?set fresh direct build directory name}" "${CAMPAIGN_CMI_BUILD:?set fresh CMI build directory name}" "${CAMPAIGN_SCOREP_BUILD:?set fresh Score-P build directory name}"
export CAMPAIGN_DIRECT_BUILD CAMPAIGN_CMI_BUILD CAMPAIGN_SCOREP_BUILD
for build in "$CAMPAIGN_DIRECT_BUILD" "$CAMPAIGN_CMI_BUILD" "$CAMPAIGN_SCOREP_BUILD"; do
    [[ "$build" =~ ^[A-Za-z0-9_-]+$ ]] || { printf 'Build must be a directory name under solver_cpp\n' >&2; exit 2; }
    if [[ -z "${CAMPAIGN_BUILD_JOB:-}" ]]; then
        [[ -x "$ROOT/solver_cpp/$build/terrain_solver" && -f "$ROOT/solver_cpp/$build/CMakeCache.txt" && -f "$ROOT/solver_cpp/$build/BUILD_READY" ]] || { printf 'Missing validated fresh build: %s\n' "$build" >&2; exit 1; }
    fi
done
if [[ -z "${CAMPAIGN_BUILD_JOB:-}" ]]; then
    [[ "$CAMPAIGN_DIRECT_BUILD" != "$CAMPAIGN_CMI_BUILD" && "$CAMPAIGN_SCOREP_BUILD" != "$CAMPAIGN_DIRECT_BUILD" && "$CAMPAIGN_SCOREP_BUILD" != "$CAMPAIGN_CMI_BUILD" ]] || exit 2
fi
previous="${CAMPAIGN_DEPENDENCY:-}"
for ((rep=1; rep<=reps; rep++)); do
    args=(--parsable --account="${CAMPAIGN_ACCOUNT:-$account}" --partition="$partition" --exclusive --nodes=1 --ntasks=96 --ntasks-per-node=96 --cpus-per-task=1 --mem=0 --time="$minutes" --export=ALL --chdir="$ROOT" --job-name="${CAMPAIGN_ID}_${platform}_${phase}_r${rep}" --output="$ROOT/logs/${CAMPAIGN_ID}_${platform}_${phase}_r${rep}_%j.out")
    [[ -z "$previous" ]] || args+=(--dependency="$previous")
    args+=("${gpu_args[@]}" "$ROOT/run_colocated_campaign.sh" "$phase" "$platform" "$rep")
    if [[ ${3:-} != --submit ]]; then
        printf 'sbatch '; printf '%q ' "${args[@]}"; printf '\n'
    else
        raw=$(sbatch "${args[@]}"); job=${raw%%;*}
        [[ "$job" =~ ^[0-9]+$ ]] || { printf 'Invalid sbatch response: %s\n' "$raw" >&2; exit 1; }
        printf '%s\t%s\t%s\t%s\t%s\n' "$CAMPAIGN_ID" "$platform" "$phase" "$rep" "$job" >> "$ROOT/logs/${CAMPAIGN_ID}_${platform}_${phase}_submitted.tsv"
    fi
done
