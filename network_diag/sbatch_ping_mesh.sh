#!/bin/bash
#SBATCH --job-name=ping_mesh_diag
#SBATCH --account=thes2181
#SBATCH --partition=c23g
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=logs/ping_mesh_%j.out

set -euo pipefail

mkdir -p logs

SAMPLES="${SAMPLES:-180}"
PING_TIMEOUT_SEC="${PING_TIMEOUT_SEC:-1}"
RAW_DIR="${RAW_DIR:-logs/ping_mesh_raw_${SLURM_JOB_ID}}"
CSV_OUT="${CSV_OUT:-logs/ping_mesh_${SLURM_JOB_ID}.csv}"
STATS_CSV="${STATS_CSV:-${CSV_OUT%.csv}_stats.csv}"
P95_MATRIX_CSV="${P95_MATRIX_CSV:-${CSV_OUT%.csv}_p95_matrix.csv}"
LOSS_MATRIX_CSV="${LOSS_MATRIX_CSV:-${CSV_OUT%.csv}_loss_matrix.csv}"

mapfile -t NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [[ "${#NODES[@]}" -eq 0 ]]; then
  echo "ERROR: No nodes resolved from SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
  exit 1
fi

TARGETS_CSV="$(IFS=,; echo "${NODES[*]}")"
START_EPOCH="$(python3 - <<'PY'
import time
print(time.time() + 3.0)
PY
)"

mkdir -p "${RAW_DIR}"

echo "Ping mesh diagnostic"
echo "  JobID: ${SLURM_JOB_ID}"
echo "  Nodes: ${TARGETS_CSV}"
echo "  Samples: ${SAMPLES}"
echo "  Timeout(sec): ${PING_TIMEOUT_SEC}"
echo "  Raw dir: ${RAW_DIR}"
echo "  Output CSV: ${CSV_OUT}"
echo "  Stats CSV: ${STATS_CSV}"
echo "  P95 matrix CSV: ${P95_MATRIX_CSV}"
echo "  Loss matrix CSV: ${LOSS_MATRIX_CSV}"

srun --export=ALL --nodes="${#NODES[@]}" --ntasks="${#NODES[@]}" --ntasks-per-node=1 \
  python3 network_diag/ping_mesh.py worker \
    --targets "${TARGETS_CSV}" \
    --samples "${SAMPLES}" \
    --start-epoch "${START_EPOCH}" \
    --timeout-sec "${PING_TIMEOUT_SEC}" \
    --out-dir "${RAW_DIR}"

python3 network_diag/ping_mesh.py aggregate \
  --targets "${TARGETS_CSV}" \
  --samples "${SAMPLES}" \
  --out-dir "${RAW_DIR}" \
  --csv "${CSV_OUT}"

python3 network_diag/ping_mesh_stats.py \
  --csv "${CSV_OUT}" \
  --stats-csv "${STATS_CSV}" \
  --p95-matrix-csv "${P95_MATRIX_CSV}" \
  --loss-matrix-csv "${LOSS_MATRIX_CSV}"

echo "Done. CSV written to ${CSV_OUT}"
echo "Stats written to ${STATS_CSV}"
echo "P95 matrix written to ${P95_MATRIX_CSV}"
echo "Loss matrix written to ${LOSS_MATRIX_CSV}"

# Example: run on known problematic nodes (adjust to availability)
# sbatch --nodes=3 --nodelist=n23g0015,n23g0014,n23g0005 network_diag/sbatch_ping_mesh.sh
# Example 5-minute run:
# SAMPLES=300 sbatch --nodes=3 --nodelist=n23g0015,n23g0014,n23g0005 network_diag/sbatch_ping_mesh.sh
