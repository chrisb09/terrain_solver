#!/bin/zsh
JOB_ID="${1:-2043512}"
LOG_FILE="logs/mini_app_output_${JOB_ID}.txt"
WATCH_LOG="logs/watch_${JOB_ID}.log"

echo "[$(date)] Watching job ${JOB_ID}..." > "${WATCH_LOG}"

while true; do
    STATE=$(sacct -j "${JOB_ID}" --format=State -P 2>/dev/null | head -n 2 | tail -n 1)
    echo "[$(date)] Job ${JOB_ID} state: ${STATE}" >> "${WATCH_LOG}"
    if [[ "${STATE}" != "PENDING" && "${STATE}" != "RUNNING" && -n "${STATE}" ]]; then
        echo "[$(date)] Job ${JOB_ID} finished with state: ${STATE}" >> "${WATCH_LOG}"
        break
    fi
    sleep 30
done

echo "[$(date)] Summary of run log:" >> "${WATCH_LOG}"
grep -E "Score-P|SmartRedis|Step 10|Solving time|profile.cubex|Finished|ERROR|Error" "${LOG_FILE}" 2>/dev/null >> "${WATCH_LOG}" || true
