#!/usr/bin/env bash
# Submit 5 replicate runs of the 5-node SmartSim traffic comparison benchmark
# Each job allocates 5 exclusive nodes on c23ms under account rwth2150:
# - 4 dedicated DB nodes (96 cores each)
# - 1 dedicated solver node (96 ranks)
# - Alternates Clustered-first vs Standalone-first to eliminate ordering bias
# - Chained dependencies ensure zero queue clutter and zero port collisions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RUNNER="${SCRIPT_DIR}/run_5node_traffic_comparison.sbatch"
MANIFEST_TXT="${SCRIPT_DIR}/5node_traffic_manifest.txt"
MANIFEST_CSV="${SCRIPT_DIR}/5node_traffic_manifest.csv"

REPLICATES=5
SLURM_ACCOUNT="rwth2150"
PARTITION="c23ms"

rm -f "${MANIFEST_TXT}" "${MANIFEST_CSV}"
echo "replicate,job_id,order,partition,account,nodes,steps" > "${MANIFEST_CSV}"

echo "========================================================================"
echo " Submitting 5-Node SmartSim Traffic Benchmark Suite (5 Replicates)"
echo " Account:    ${SLURM_ACCOUNT} | Partition: ${PARTITION} (5 exclusive nodes)"
echo " Topology:   1 solver node (96 ranks) + 4 DB nodes (96 cores/shard)"
echo " Workload:   4K WaterCNN, 102 steps (51 ML steps) per case"
echo " Replicates: ${REPLICATES} (Chained dependency sequence)"
echo "========================================================================"

prev_job_id=""

for rep in $(seq 1 "${REPLICATES}"); do
    if (( rep % 2 == 1 )); then
        order="clustered_first"
    else
        order="standalone_first"
    fi

    dep_args=()
    if [[ -n "${prev_job_id}" ]]; then
        dep_args=("--dependency=afterany:${prev_job_id}")
    fi

    job_name="ss_5n_r${rep}_${order:0:4}"

    submit_out=$(sbatch --parsable \
        "${dep_args[@]}" \
        --account="${SLURM_ACCOUNT}" \
        --partition="${PARTITION}" \
        --job-name="${job_name}" \
        --export="ALL,ORDER_ENV=${order}" \
        "${RUNNER}")

    job_id=$(echo "${submit_out}" | grep -oE '[0-9]+' | tail -n 1)

    if [[ -z "${job_id}" ]]; then
        echo "ERROR: Failed to submit replicate ${rep}: ${submit_out}" >&2
        exit 1
    fi

    echo "[${rep}/${REPLICATES}] Submitted rep ${rep} (${order}): Job ID ${job_id}"
    echo "${rep} ${job_id} ${order}" >> "${MANIFEST_TXT}"
    echo "${rep},${job_id},${order},${PARTITION},${SLURM_ACCOUNT},5,102" >> "${MANIFEST_CSV}"

    prev_job_id="${job_id}"
done

echo "========================================================================"
echo " All ${REPLICATES} replicate jobs submitted successfully."
echo " Manifest written to: ${MANIFEST_TXT} and ${MANIFEST_CSV}"
echo " To monitor progress:"
echo "   squeue -u \$USER"
echo " To analyze completed results:"
echo "   python3 analyze_5node_suite.py --manifest ${MANIFEST_TXT}"
echo "========================================================================"
