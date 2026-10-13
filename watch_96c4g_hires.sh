#!/usr/bin/env bash
# Watches the chained 96c4g hires suite (jobs 3738506..3738560) and runs the
# Score-P analysis automatically once all 9 jobs have left the queue.
set -u
cd /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/mini_app
JOBS="3738506 3738507 3738512 3738517 3738522 3738533 3738549 3738550 3738560"
LOCK=/tmp/ro092286_watch_96c4g_hires.lock
LOG=logs/watch_96c4g_hires.log

exec 9>"${LOCK}"
flock -n 9 || exit 0

while true; do
    remaining=$(squeue -j "${JOBS// /,}" -h -o "%i" 2>/dev/null | wc -l)
    if [[ "${remaining}" -eq 0 ]]; then
        # All jobs left the queue - wait a grace period for copy-back, then analyze
        sleep 120
        ./analyze_scorep_suites.sh 96c4g_hires > logs/analyze_96c4g_hires.log 2>&1
        status=$?
        echo "$(date '+%F %T') chain finished, analysis exit=${status}" >> "${LOG}"
        # Second pass: configs that were skipped while copy-back was still ongoing
        sleep 60
        ./analyze_scorep_suites.sh 96c4g_hires >> logs/analyze_96c4g_hires.log 2>&1
        echo "$(date '+%F %T') second analysis pass done" >> "${LOG}"
        exit 0
    fi
    sleep 120
done
