#!/usr/bin/env bash
# Run the 5-stage AIx pipeline prediction (compare_cmi_frameworks.py) for every
# (suite, scenario, layout) with a collective + pipelined result pair.
#
# The script detects frameworks by directory name, so per (scenario, layout)
# staging symlinks with the names it expects are created.
#
# N (pipelining depth in tau_pipe=(tau_step+(N-1)*tau_max)/N) = number of GPU
# workgroups: 1 for the 1-GPU scenarios (s3, s5), 4 for the 4-GPU ones (s6, s7).
#
# Usage: run_hh3_predictions.sh <suite> [suite ...]
set -euo pipefail
cd "$(dirname "$0")"
SUITE="${1:?suite required, e.g. hh3_g1080}"
shift || true

declare -A SUITE_META=(
    [hh3_g1080]="benchmark_giant_mlp|1920x1080|12"
    [hh3_w8k]="watercnn|7680x4320|22"
    [hethires3]="watercnn|3840x2160|202"
)
meta="${SUITE_META[${SUITE}]:-}"
if [[ -z "${meta}" ]]; then
    echo "unknown suite ${SUITE}" >&2; exit 1
fi
IFS='|' read -r MODEL RES STEPS <<< "${meta}"

CPP_ML_DIR="/hpcwork/ro092286/smartsim/CPP-ML-Interface"
STAGE="${SUITE}_prediction_stage"
rm -rf "${STAGE}"; mkdir -p "${STAGE}"

for layout in default affine numa; do
    for scen in 3 5 6 7; do
        coll="results/${SUITE}/s${scen}_aix_coll_${layout}"
        p2p="results/${SUITE}/s${scen}_aix_p2p_${layout}"
        [[ -f "${coll}/cmi_phase_summary.csv" && -f "${p2p}/cmi_phase_summary.csv" ]] || { echo "[SKIP] s${scen}_${layout}: missing pair"; continue; }
        # N = GPU workgroups of the scenario
        case "${scen}" in 3|5) N=1 ;; 6|7) N=4 ;; *) N=4 ;; esac
        out="results/${SUITE}/predictions/s${scen}_${layout}"
        mkdir -p "${out}"
        ln -sfn "$(realpath "${coll}")" "${STAGE}/results_gpu_aix_collective"
        ln -sfn "$(realpath "${p2p}")" "${STAGE}/results_gpu_aix_pipelined"
        echo "--- s${scen}_${layout} (N=${N}) ---"
        python3 "${CPP_ML_DIR}/scripts/compare_cmi_frameworks.py" \
            --results-dirs "${STAGE}/results_gpu_aix_collective" "${STAGE}/results_gpu_aix_pipelined" \
            --output-dir "${out}" \
            --n-workgroups "${N}" \
            --model "${MODEL}" --resolution "${RES}" --steps "${STEPS}" \
            --no-plots
    done
done

# Combined markdown index of all predictions
mkdir -p "results/${SUITE}/predictions"
{
    echo "# AIx 5-stage pipeline predictions — ${SUITE}"
    echo
    echo "| scenario | layout | N | predicted pipe stage (ms) | predicted total step (ms) | measured p2p step (ms) | error |"
    echo "|---|---|---|---|---|---|---|"
    for f in results/${SUITE}/predictions/s*/aix_pipeline_prediction.md; do
        [[ -f "${f}" ]] || continue
        scen_layout="$(basename "$(dirname "${f}")")"
        scen="${scen_layout%%_*}"; layout="${scen_layout#*_}"
        N=$(grep -oE "Workgroup Size \(Ranks / Chunks, N\):\*\* [0-9]+" "${f}" | grep -oE "[0-9]+$" | head -1)
        pred_stage=$(grep -oE "tau_pipelined_stages = .*= [0-9.]+ ms" "${f}" | grep -oE "[0-9.]+ ms" | head -1 | grep -oE "[0-9.]+")
        pred_total=$(grep -oE "Predicted Total Pipelined Step .* = [0-9.]+ ms" "${f}" | grep -oE "[0-9.]+ ms" | tail -1 | grep -oE "[0-9.]+")
        actual=$(grep -oE "Total Warm Step Duration \| [0-9.]+ \| \*\*[0-9.]+" "${f}" | grep -oE "[0-9.]+$" | head -1)
        err=$(grep -oE "Relative Prediction Error \| — \| — \| \*\*[0-9.]+%" "${f}" | grep -oE "[0-9.]+%")
        echo "| ${scen} | ${layout} | ${N} | ${pred_stage} | ${pred_total} | ${actual} | ${err} |"
    done
} > "results/${SUITE}/predictions/README.md"
echo "[✓] predictions written under results/${SUITE}/predictions/"
