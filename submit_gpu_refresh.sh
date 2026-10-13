#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && ( $1 == --dry-run || $1 == --submit ) ]] || { printf 'Usage: bash submit_gpu_refresh.sh --dry-run|--submit CAMPAIGN_TOKEN\n' >&2; exit 2; }
exec python3 "$(dirname "$(readlink -f "$0")")/gpu_refresh_campaign.py" "$@"
