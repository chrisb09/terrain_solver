#!/usr/bin/env bash

# Exit on error
set -euo pipefail

# analyze_scorep_suites.sh
# ========================
# Post-run analysis driver for the three Score-P benchmark suites (het24, 96c1g, 96c4g).
# For every suite/configuration it:
#   1. Locates the Score-P CUBE profiles in scorep_runs/ (per-rank profiling dirs
#      "<tag>_rank_*" or a single shared tracing dir "<tag>/profile.cubex").
#   2. Runs analyze_cmi_scorep_profiles.py to produce icicle/sunburst/stage-breakdown
#      plots and cmi_phase_summary.csv/md inside results/<suite>/<config>/.
#   3. For aix_p2p configs, additionally renders the per-step P2P timeline Gantts and
#      overlap analysis from logs/aix_p2p_timeline_<jobid>/ (OTF2 tracing is disabled
#      for the async pipelined mode, so the CSV timeline is the source of truth).
#
# Usage:
#   ./analyze_scorep_suites.sh                 # analyze all 25 configurations
#   ./analyze_scorep_suites.sh het24           # analyze one suite only
#   ./analyze_scorep_suites.sh het24 aix_p2p   # analyze one specific configuration

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CPP_ML_DIR="$(realpath "${SCRIPT_DIR}/../CPP-ML-Interface")"
ANALYZE="${CPP_ML_DIR}/scripts/analyze_cmi_scorep_profiles.py"
P2P_PLOT="${CPP_ML_DIR}/scripts/plot_p2p_timeline.py"

# suite|config|job_id  (job_id locates logs/aix_p2p_timeline_<jid> for pipelined runs)
CONFIG_TABLE=(
    "het24|smartsim_c0|3547235"
    "het24|smartsim_per_node_db|3547243"
    "het24|smartsim_c1|3547249"
    "het24|smartsim_c3|3547257"
    "het24|aix_coll|3547260"
    "het24|aix_p2p|3547266"
    "het24|phydll_cpp|3547269"
    "het24|phydll_py|3547272"
    "96c1g|smartsim_c0|3547277"
    "96c1g|smartsim_per_node_db|3547279"
    "96c1g|smartsim_c1|3547281"
    "96c1g|smartsim_c3|3547285"
    "96c1g|aix_coll|3547287"
    "96c1g|aix_p2p|3547289"
    "96c1g|phydll_cpp|3547291"
    "96c1g|phydll_py|3547293"
    "96c4g|smartsim_c0|3547296"
    "96c4g|smartsim_per_node_db|3547298"
    "96c4g|smartsim_per_gpu_db|3547300"
    "96c4g|smartsim_c1|3547302"
    "96c4g|smartsim_c3|3547305"
    "96c4g|aix_coll|3547309"
    "96c4g|aix_p2p|3547316"
    "96c4g|phydll_cpp|3547318"
    "96c4g|phydll_py|3547320"
)

# Optional positional filters: <suite> [<config>]
FILTER_SUITE="${1:-}"
FILTER_CONFIG="${2:-}"

find_cubex_input() {
    local tag="$1"
    # Prefer per-rank profiling dirs (profiling-only runs)
    local per_rank
    per_rank=$(compgen -G "scorep_runs/${tag}_rank_*/profile.cubex" || true)
    if [[ -n "${per_rank}" ]]; then
        echo "scorep_runs/${tag}_rank_*"
        return 0
    fi
    # Fall back to a single shared tracing experiment directory
    if [[ -f "scorep_runs/${tag}/profile.cubex" ]]; then
        echo "scorep_runs/${tag}/profile.cubex"
        return 0
    fi
    # Last resort: the tracing dir may contain a nested scorep-<timestamp> experiment
    local nested
    nested=$(compgen -G "scorep_runs/${tag}/*/profile.cubex" || true)
    if [[ -n "${nested}" ]]; then
        echo "scorep_runs/${tag}/*/profile.cubex"
        return 0
    fi
    return 1
}

find_dl_cubex_input() {
    local tag="$1"
    local dl_glob
    dl_glob=$(compgen -G "scorep_runs/${tag}_py_client*/profile.cubex" || true)
    if [[ -z "${dl_glob}" ]]; then
        dl_glob=$(compgen -G "scorep_runs/${tag}_cpp_client*/profile.cubex" || true)
    fi
    if [[ -n "${dl_glob}" ]]; then
        # Echo the directory pattern (strip the /profile.cubex suffix)
        echo "${dl_glob%/profile.cubex}"
    fi
}

analyze_config() {
    local suite="$1"
    local config="$2"
    local job_id="$3"
    local tag="${suite}_${config}"
    local out_dir="results/${suite}/${config}"

    echo ""
    echo "======================================================================"
    echo " Analyzing: ${suite} / ${config}  (tag=${tag}, job=${job_id})"
    echo "======================================================================"

    if ! cubex_input="$(find_cubex_input "${tag}")"; then
        echo "[SKIP] No CUBE profiles found for ${tag} (job pending, failed, or traces still being written)."
        return 1
    fi
    echo "  Solver profiles: ${cubex_input}"

    mkdir -p "${out_dir}"

    local dl_args=()
    if [[ "${config}" == phydll_* ]]; then
        if dl_input="$(find_dl_cubex_input "${tag}")"; then
            echo "  DL-client profiles: ${dl_input}"
            dl_args=(--dl-cubex "${dl_input}")
        else
            echo "  [WARN] No separate DL-client profiles found for ${tag} (likely merged into the shared tracing dir)."
        fi
    fi

    python3 "${ANALYZE}" \
        --cubex "${cubex_input}" \
        "${dl_args[@]}" \
        --model watercnn \
        --resolution 1920x1080 \
        --output-dir "${out_dir}" \
        --title "${config} (${suite})" || {
            echo "[ERROR] analyze_cmi_scorep_profiles.py failed for ${tag}"
            return 1
        }

    # AIx P2P pipelined: render per-step timeline Gantts + overlap analysis from CSV
    if [[ "${config}" == "aix_p2p" ]]; then
        local timeline_dir="logs/aix_p2p_timeline_${job_id}"
        if [[ -d "${timeline_dir}" ]]; then
            echo "  P2P timeline CSVs: ${timeline_dir}"
            python3 "${P2P_PLOT}" "${timeline_dir}" --output-dir "${out_dir}" \
                || echo "[WARN] plot_p2p_timeline.py failed for ${tag}"
            python3 "${ANALYZE}" --p2p-timeline-dir "${timeline_dir}" \
                --output-dir "${out_dir}" --title "${config} timeline (${suite})" \
                || echo "[WARN] p2p-timeline analysis failed for ${tag}"
        else
            echo "  [WARN] Timeline dir ${timeline_dir} not found for ${tag}."
        fi
    fi

    echo "  -> Artifacts written to ${out_dir}"
    return 0
}

FAILED=0
SKIPPED=0
DONE=0
for row in "${CONFIG_TABLE[@]}"; do
    IFS='|' read -r suite config job_id <<< "${row}"
    if [[ -n "${FILTER_SUITE}" && "${suite}" != "${FILTER_SUITE}" ]]; then
        continue
    fi
    if [[ -n "${FILTER_CONFIG}" && "${config}" != "${FILTER_CONFIG}" ]]; then
        continue
    fi
    if analyze_config "${suite}" "${config}" "${job_id}"; then
        DONE=$((DONE + 1))
    else
        if [[ -n "$(compgen -G "scorep_runs/${suite}_${config}*" 2>/dev/null || true)" ]]; then
            FAILED=$((FAILED + 1))
        else
            SKIPPED=$((SKIPPED + 1))
        fi
    fi
done

echo ""
echo "======================================================================"
echo " Analysis summary: ${DONE} analyzed, ${SKIPPED} skipped (profiles missing), ${FAILED} failed"
echo "======================================================================"
[[ "${FAILED}" -eq 0 ]]
