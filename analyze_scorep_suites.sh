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
# Updated Sep 6 2026: valid run of the 96c4g suite (mockup-MPP adapter fixed in
# build.sh). The older Sep 3 job IDs below were removed - those runs died at
# Score-P init (all-ranks experiment-dir race, libscorep_mpp_mockup) and never
# produced profile data. phydll_py is excluded (structural Score-P limitation).
# Optional positional filters: <suite> [<config>]
# Updated Sep 6 2026 (2): hires re-run of the 96c4g suite: 3840x2160 world,
# batch 20M, 202 steps (101 Regular / 101 ML). 9th case aix_p2p_fullcredits
# runs the pipelined mode with AIX_P2P_INITIAL_CREDITS=23 (all workers send
# immediately) as a fully independent comparison against aix_p2p.
CONFIG_TABLE=(
    "96c4g|smartsim_c0|3631719"
    "96c4g|smartsim_per_node_db|3631724"
    "96c4g|smartsim_per_gpu_db|3631726"
    "96c4g|smartsim_c1|3631728"
    "96c4g|smartsim_c3|3631730"
    "96c4g|aix_coll|3631732"
    "96c4g|aix_p2p|3631734"
    "96c4g|phydll_cpp|3631736"
    "96c4g_hires|smartsim_c0|3738506"
    "96c4g_hires|smartsim_per_node_db|3738507"
    "96c4g_hires|smartsim_per_gpu_db|3738512"
    "96c4g_hires|smartsim_c1|3738517"
    "96c4g_hires|smartsim_c3|3738522"
    "96c4g_hires|aix_coll|3738533"
    "96c4g_hires|aix_p2p|3738549"
    "96c4g_hires|phydll_cpp|3738550"
    "96c4g_hires|aix_p2p_fullcredits|3738560"
    "96c4g_hires|smartsim_c0_batch20m|3768110"
    "96c4g_hires|smartsim_c0_tpq4|3768111"
    "96c4g_hires|smartsim_c0_batch20m_tpq4|3768112"
    "hethires2|s1_aix_coll|3988906"
    "hethires2|s1_aix_p2p|3988911"
    "hethires2|s1_aix_p2p_fullcredits|3988915"
    "hethires2|s1_smartsim_c0|3988920"
    "hethires2|s1_smartsim_c3|3988924"
    "hethires2|s1_phydll_cpp|3988928"
    "hethires2|s1_smartsim_per_gpu_db|3988933"
    "hethires2|s2_aix_coll|3989191"
    "hethires2|s2_aix_p2p|3989194"
    "hethires2|s2_aix_p2p_fullcredits|3989197"
    "hethires2|s2_smartsim_c0|3989200"
    "hethires2|s2_smartsim_c3|3989203"
    "hethires2|s2_phydll_cpp|3989206"
    "hethires2|s3_aix_coll|3988906"
    "hethires2|s3_aix_p2p|3988911"
    "hethires2|s3_aix_p2p_fullcredits|3988915"
    "hethires2|s3_smartsim_c0|3988920"
    "hethires2|s3_smartsim_c3|3988924"
    "hethires2|s3_phydll_cpp|3988928"
    "hethires2|s3_smartsim_per_gpu_db|3988933"
    "hethires2|s4_aix_coll|3989191"
    "hethires2|s4_aix_p2p|3989194"
    "hethires2|s4_aix_p2p_fullcredits|3989197"
    "hethires2|s4_smartsim_c0|3989200"
    "hethires2|s4_smartsim_c3|3989203"
    "hethires2|s4_phydll_cpp|3989206"
    "hethires2|s5_aix_coll|3989191"
    "hethires2|s5_aix_p2p|3989194"
    "hethires2|s5_aix_p2p_fullcredits|3989197"
    "hethires2|s5_smartsim_c0|3989200"
    "hethires2|s5_smartsim_c3|3989203"
    "hethires2|s5_phydll_cpp|4020257"
    "hethires2|s6_aix_coll|3988906"
    "hethires2|s6_aix_p2p|3988911"
    "hethires2|s6_aix_p2p_fullcredits|3988915"
    "hethires2|s6_smartsim_c0|3988920"
    "hethires2|s6_smartsim_c3|3988924"
    "hethires2|s6_phydll_cpp|3988928"
    "hethires2|s6_smartsim_per_gpu_db|3988933"
    "hethires2|s7_aix_coll|3989191"
    "hethires2|s7_aix_p2p|3989194"
    "hethires2|s7_aix_p2p_fullcredits|3989197"
    "hethires2|s7_smartsim_c0|3989200"
    "hethires2|s7_smartsim_c3|3989203"
    "hethires2|s7_phydll_cpp|4020257"
)
# HH3 rows are generated from the suite bookkeeping; see tmp_hh3_rows.txt.
if [[ -f "${SCRIPT_DIR}/hh3_config_table.txt" ]]; then
    while IFS= read -r _row; do
        [[ -z "${_row}" ]] && continue
        CONFIG_TABLE+=("${_row}")
    done < "${SCRIPT_DIR}/hh3_config_table.txt"
fi
# Additional suite tables (one <suite>|<config>|<jobid> row per line).
for _tbl in "${SCRIPT_DIR}"/hh3_config_table_*.txt; do
    [[ -f "${_tbl}" ]] || continue
    while IFS= read -r _row; do
        [[ -z "${_row}" ]] && continue
        CONFIG_TABLE+=("${_row}")
    done < "${_tbl}"
done

# Optional positional filters: <suite> [<config>]
FILTER_SUITE="${1:-}"
FILTER_CONFIG="${2:-}"

# Stage guards for the parallel driver:
#   SKIP_CROSS_ENV=1   -> run only the per-config analysis (no cross-config tail)
#   CROSS_ONLY_ENV=1   -> run only the cross-config tail (dirs must already exist)
SKIP_CROSS="${SKIP_CROSS_ENV:-0}"
CROSS_ONLY="${CROSS_ONLY_ENV:-0}"

# Scale scenarios (s8-s11) carry 288-1632-rank OTF2 traces; otf2-print on those
# is prohibitively slow, so trace-based analyses are restricted to the base
# scenarios (the CSV timelines and CUBE profiles still cover the scale runs).
_is_scale_cfg() {
    [[ "$1" =~ ^s(8|9|10|11)_ ]]
}

# HH3_PGF_ENV=1 -> all suite-level figures are also written as .pgf/.pdf
_hh3_pgf_args=()
if [[ "${HH3_PGF_ENV:-0}" == "1" ]]; then
    _hh3_pgf_args=(--pgf)
fi

find_cubex_input() {
    local tag="$1"
    local job_id="${2:-}"
    # Layout since the per-job experiment-dir fix: unique dir per job
    # (<tag>_<jobid>[_rank_N]). Fall back to the legacy fixed-tag layout
    # (<tag>[_rank_N]) for older runs. Prefer the exact job's dirs first so
    # stale dirs from previous suite invocations are never analyzed.
    local candidates=()
    if [[ -n "${job_id}" ]]; then
        candidates+=("${tag}_${job_id}")
    fi
    # HH2 multi-scenario wrappers set RUN_ID_ENV = SCOREP_DIR_TAG, so the
    # Score-P dirs carry the doubled name <tag>_<tag> (optionally with a
    # -<timestamp> archive suffix).
    candidates+=("${tag}_${tag}" "${tag}")
    local cand
    for cand in "${candidates[@]}"; do
        local per_rank
        per_rank=$(compgen -G "scorep_runs/${cand}_rank_*/profile.cubex" || true)
        if [[ -n "${per_rank}" ]]; then
            echo "scorep_runs/${cand}_rank_*"
            return 0
        fi
        if [[ -f "scorep_runs/${cand}/profile.cubex" ]]; then
            echo "scorep_runs/${cand}/profile.cubex"
            return 0
        fi
        # The tracing dir may contain a nested scorep-<timestamp> experiment
        local nested
        nested=$(compgen -G "scorep_runs/${cand}/*/profile.cubex" || true)
        if [[ -n "${nested}" ]]; then
            echo "scorep_runs/${cand}/*/profile.cubex"
            return 0
        fi
    done
    return 1
}

find_dl_cubex_input() {
    local tag="$1"
    local job_id="${2:-}"
    local dl_glob=""
    if [[ -n "${job_id}" ]]; then
        dl_glob=$(compgen -G "scorep_runs/${tag}_${job_id}_py_client*/profile.cubex" || true)
        if [[ -z "${dl_glob}" ]]; then
            dl_glob=$(compgen -G "scorep_runs/${tag}_${job_id}_cpp_client*/profile.cubex" || true)
        fi
    fi
    if [[ -z "${dl_glob}" ]]; then
        dl_glob=$(compgen -G "scorep_runs/${tag}_py_client*/profile.cubex" || true)
    fi
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

    # Idempotency: skip configs whose analysis artifacts already exist
    # (FORCE_ENV=1 re-analyzes from scratch).
    if [[ -f "${out_dir}/cmi_phase_summary.csv" && "${FORCE_ENV:-0}" != "1" ]]; then
        echo "[SKIP-DONE] ${suite} / ${config} (results already present)"
        return 0
    fi

    echo ""
    echo "======================================================================"
    echo " Analyzing: ${suite} / ${config}  (tag=${tag}, job=${job_id})"
    echo "======================================================================"

    if ! cubex_input="$(find_cubex_input "${tag}" "${job_id}")"; then
        echo "[SKIP] No CUBE profiles found for ${tag} (job pending, failed, or traces still being written)."
        return 1
    fi
    echo "  Solver profiles: ${cubex_input}"

    mkdir -p "${out_dir}"

    # Run parameters from the job's own run_metadata.json (written by the job
    # script into the Score-P experiment dir). Falls back to the legacy 1080p
    # defaults when the metadata is missing (very old runs).
    local run_model="" run_res="" run_batch=""
    local input_dir="${cubex_input%/profile.cubex}"
    local input_base="${input_dir%_rank_*}"
    local meta_json="" meta_cand
    for meta_cand in "${input_base}/run_metadata.json" "${input_dir}/run_metadata.json"; do
        if [[ -f "${meta_cand}" ]]; then
            meta_json="${meta_cand}"
            break
        fi
    done
    if [[ -n "${meta_json}" ]]; then
        mapfile -t _meta_fields <<< "$(python3 - "${meta_json}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for key in ("model", "resolution", "batch_size"):
    print(d.get(key) or "")
PY
)"
        run_model="${_meta_fields[0]:-}"
        run_res="${_meta_fields[1]:-}"
        run_batch="${_meta_fields[2]:-}"
    fi
    local param_args=(--model "${run_model:-watercnn}")
    if [[ -n "${run_res}" ]]; then
        param_args+=(--resolution "${run_res}")
    else
        param_args+=(--resolution 1920x1080)
    fi
    [[ -n "${run_batch}" ]] && param_args+=(--batch-size "${run_batch}")
    echo "  Run parameters: model=${run_model:-watercnn}, resolution=${run_res:-1920x1080}, batch=${run_batch:-n/a}"

    local dl_args=()
    if [[ "${config}" == phydll_* ]]; then
        if dl_input="$(find_dl_cubex_input "${tag}" "${job_id}")"; then
            echo "  DL-client profiles: ${dl_input}"
            dl_args=(--dl-cubex "${dl_input}")
        else
            echo "  [WARN] No separate DL-client profiles found for ${tag} (likely merged into the shared tracing dir)."
        fi
    fi

    python3 "${ANALYZE}" \
        --cubex "${cubex_input}" \
        "${dl_args[@]}" \
        "${param_args[@]}" \
        --output-dir "${out_dir}" \
        --title "${config} (${suite})" || {
            echo "[ERROR] analyze_cmi_scorep_profiles.py failed for ${tag}"
            return 1
        }

    # AIx P2P pipelined: render per-step timeline Gantts + overlap analysis from CSV
    if [[ "${config}" == *aix_p2p* ]]; then
        # HH2 wrappers key the timeline dir by RUN_ID (= the full tag).
        local timeline_dir="logs/aix_p2p_timeline_${job_id}"
        if [[ ! -d "${timeline_dir}" ]]; then
            timeline_dir="logs/aix_p2p_timeline_${suite}_${config}"
        fi
        if [[ ! -d "${timeline_dir}" ]]; then
            timeline_dir="logs/aix_p2p_timeline_${config}"
        fi
        if [[ -d "${timeline_dir}" ]]; then
            echo "  P2P timeline CSVs: ${timeline_dir}"
            # Gantt rendering (per-step, per-workgroup) is only feasible for the
            # base scenarios; scale runs (288-1632 ranks) keep the metrics CSVs
            # only. P2P_GANTT_ENV tri-state: 1=always render, 0=never render,
            # unset/auto=render only for base scenarios.
            _gantt_args=()
            _want_gantt=0
            case "${P2P_GANTT_ENV:-auto}" in
                1) _want_gantt=1 ;;
                0) _want_gantt=0 ;;
                *) _is_scale_cfg "${config}" || _want_gantt=1 ;;
            esac
            if [[ "${_want_gantt}" -eq 1 ]]; then
                _gantt_args=(--p2p-gantt)
                python3 "${P2P_PLOT}" "${timeline_dir}" --output-dir "${out_dir}" \
                    || echo "[WARN] plot_p2p_timeline.py failed for ${tag}"
            fi
            python3 "${ANALYZE}" --p2p-timeline-dir "${timeline_dir}" \
                ${_gantt_args[@]+"${_gantt_args[@]}"} \
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
if [[ "${CROSS_ONLY}" != "1" ]]; then
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
fi

if [[ "${SKIP_CROSS}" != "1" ]]; then
echo ""
echo "======================================================================"
echo " Analysis summary: ${DONE} analyzed, ${SKIPPED} skipped (profiles missing), ${FAILED} failed"
echo "======================================================================"

# Cross-configuration comparison: stacked pre/inference/post decomposition of
# the steady ML step for every analyzed config of this suite (bar chart).
CONFIG_COMPARE="${CPP_ML_DIR}/scripts/plot_config_comparison.py"

# Per-ML-step makespan series (rank 0, from the solver logs) + per-workgroup
# makespan spread for the AIx runs (p2p: CSV timelines, collective: OTF2 trace).
MAKESPAN="${CPP_ML_DIR}/scripts/analyze_ml_step_makespan.py"
if [[ -z "${OTF2_PRINT:-}" ]]; then
    OTF2_PRINT="$(command -v otf2-print || true)"
    if [[ -z "${OTF2_PRINT}" && -x "${CPP_ML_DIR}/tmp/opencode/scorep-8.4-papi72-install/bin/otf2-print" ]]; then
        OTF2_PRINT="${CPP_ML_DIR}/tmp/opencode/scorep-8.4-papi72-install/bin/otf2-print"
    fi
fi

# P2P vs collective phase analysis (last steady ML step): ready-range sums/counts,
# input/output windows per controller, overlay + comparison graphs.
P2P_PHASES="${CPP_ML_DIR}/scripts/analyze_p2p_inference_phases.py"

for suite_dir in results/*/; do
    suite_name="$(basename "${suite_dir}")"
    if [[ -n "${FILTER_SUITE}" && "${suite_name}" != "${FILTER_SUITE}" ]]; then
        continue
    fi
    if compgen -G "${suite_dir}*/cmi_phase_summary.csv" > /dev/null; then
        echo ""
        echo "Rendering cross-config phase comparison for ${suite_name}..."
        python3 "${CONFIG_COMPARE}" "${suite_dir%/}" \
            --output "${suite_dir%/}/config_comparison.png" \
            || echo "[WARN] config comparison failed for ${suite_name}"

        echo "Rendering average warm ML-step time per configuration for ${suite_name}..."
        python3 "${CPP_ML_DIR}/scripts/plot_avg_step_time.py" "${suite_dir%/}" \
            || echo "[WARN] avg warm step time plot failed for ${suite_name}"
    fi

    if compgen -G "${suite_dir}*/cmi_phase_summary.csv" > /dev/null; then
        local_pairs=() timeline_pairs=() trace_pairs=()
        for row in "${CONFIG_TABLE[@]}"; do
            IFS='|' read -r s c j <<< "${row}"
            [[ "${s}" != "${suite_name}" ]] && continue
            # Solver log: HH2 runs write logs/mini_app_output_<tag>.txt
            # (RUN_ID = tag); legacy suites used logs/mini_app_output_<jobid>.txt
            _log="logs/mini_app_output_${s}_${c}.txt"
            [[ -f "${_log}" ]] || _log="logs/mini_app_output_${j}.txt"
            local_pairs+=("${c}=${_log}")
            if [[ "${c}" == *aix_p2p* ]]; then
                _tl="logs/aix_p2p_timeline_${s}_${c}"
                [[ -d "${_tl}" ]] || _tl="logs/aix_p2p_timeline_${j}"
                [[ -d "${_tl}" ]] && timeline_pairs+=("${c}=${_tl}")
            fi
            if [[ "${c}" == *aix_coll* ]] && ! _is_scale_cfg "${c}"; then
                _tr="scorep_runs/${s}_${c}_${j}"
                [[ -f "${_tr}/traces.otf2" ]] || _tr="scorep_runs/${s}_${c}_${s}_${c}"
                [[ -f "${_tr}/traces.otf2" ]] && trace_pairs+=("${c}=${_tr}")
            fi
        done
        echo ""
        echo "Rendering per-step makespan / workgroup-imbalance analysis for ${suite_name}..."
        OTF2_PRINT="${OTF2_PRINT:-otf2-print}" python3 "${MAKESPAN}" \
            --output-dir "${suite_dir%/}" \
            --log-config ${local_pairs[@]+"${local_pairs[@]}"} \
            ${timeline_pairs[@]+"--timeline" "${timeline_pairs[@]}"} \
            ${trace_pairs[@]+"--trace" "${trace_pairs[@]}"} \
            || echo "[WARN] makespan analysis failed for ${suite_name}"

        # P2P vs collective step-structure analysis (needs the coll + one
        # smartsim trace; both p2p timeline variants). Traces are only kept
        # for tracing-enabled configs (aix_coll, smartsim_c*, phydll_c*).
        # Scenario-aware grouping: HH2 tags are <s<N>>_<config>, in which case
        # each scenario's collective is compared against ITS OWN p2p timelines
        # (one invocation per scenario, prefixed combined graphs). Legacy
        # configs (no s<N>_ prefix) keep the single suite-wide comparison.
        _has_scenarios=0
        for row in "${CONFIG_TABLE[@]}"; do
            IFS='|' read -r s c j <<< "${row}"
            [[ "${s}" != "${suite_name}" ]] && continue
            [[ "${c}" =~ ^s[0-9]+_ ]] && _has_scenarios=1 && break
        done

        _resolve_coll_trace() {  # $1=suite $2=config $3=jobid
            local tr="scorep_runs/$1_$2_$3"
            [[ -f "${tr}/traces.otf2" ]] || tr="scorep_runs/$1_$2_$1_$2"
            [[ -f "${tr}/traces.otf2" ]] && echo "${tr}"
        }
        _resolve_timeline_dir() {  # $1=suite $2=config $3=jobid
            local tl="logs/aix_p2p_timeline_$1_$2"
            [[ -d "${tl}" ]] || tl="logs/aix_p2p_timeline_$3"
            [[ -d "${tl}" ]] && echo "${tl}"
        }

        if [[ "${_has_scenarios}" -eq 1 ]]; then
            declare -A _scen_coll=() _scen_smart=() _scen_tl=() _scen_seen=()
            _scen_order=()
            for row in "${CONFIG_TABLE[@]}"; do
                IFS='|' read -r s c j <<< "${row}"
                [[ "${s}" != "${suite_name}" ]] && continue
                scen="${c}"
                if [[ "${c}" =~ ^(s[0-9]+)_ ]]; then scen="${BASH_REMATCH[1]}"; fi
                if [[ -z "${_scen_seen[${scen}]:-}" ]]; then
                    _scen_seen[${scen}]=1; _scen_order+=("${scen}")
                fi
                if [[ "${c}" == *_aix_coll* ]] && ! _is_scale_cfg "${c}"; then
                    _tr="$(_resolve_coll_trace "${s}" "${c}" "${j}")"
                    [[ -n "${_tr}" ]] && _scen_coll[${scen}]="${c}=${_tr}"
                fi
                if [[ "${c}" == *_smartsim_c0* && -z "${_scen_smart[${scen}]:-}" ]] && ! _is_scale_cfg "${c}"; then
                    _tr="$(_resolve_coll_trace "${s}" "${c}" "${j}")"
                    [[ -n "${_tr}" ]] && _scen_smart[${scen}]="${c}=${_tr}"
                fi
                if [[ "${c}" == *_aix_p2p* ]]; then
                    _tl="$(_resolve_timeline_dir "${s}" "${c}" "${j}")"
                    [[ -n "${_tl}" ]] && _scen_tl[${scen}]="${_scen_tl[${scen}]:-} ${c}=${_tl}"
                fi
            done
            for scen in "${_scen_order[@]}"; do
                [[ -z "${_scen_coll[${scen}]:-}" || -z "${_scen_tl[${scen}]:-}" ]] && continue
                echo ""
                echo "Rendering P2P vs collective step-structure analysis for ${suite_name} / ${scen}..."
                ss_args=()
                [[ -n "${_scen_smart[${scen}]:-}" ]] && ss_args=(--smartsim-trace "${_scen_smart[${scen}]}")
                OTF2_PRINT="${OTF2_PRINT:-otf2-print}" python3 "${P2P_PHASES}" \
                    --output-dir "${suite_dir%/}" \
                    --timeline ${_scen_tl[${scen}]} \
                    --coll-trace "${_scen_coll[${scen}]}" \
                    ${ss_args[@]+"${ss_args[@]}"} \
                    ${_hh3_pgf_args[@]+"${_hh3_pgf_args[@]}"} \
                    --file-prefix "${scen}" \
                    || echo "[WARN] p2p phase analysis failed for ${suite_name} / ${scen}"
            done
            unset _scen_coll _scen_smart _scen_tl _scen_seen _scen_order
        else
        timeline_phase_pairs=() coll_trace_pair="" smartsim_trace_pair=""
        for row in "${CONFIG_TABLE[@]}"; do
            IFS='|' read -r s c j <<< "${row}"
            [[ "${s}" != "${suite_name}" ]] && continue
            if [[ "${c}" == *aix_p2p* ]]; then
                _tl="$(_resolve_timeline_dir "${s}" "${c}" "${j}")"
                [[ -n "${_tl}" ]] && timeline_phase_pairs+=("${c}=${_tl}")
            fi
            if [[ "${c}" == *aix_coll* ]] && ! _is_scale_cfg "${c}"; then
                _tr="$(_resolve_coll_trace "${s}" "${c}" "${j}")"
                [[ -n "${_tr}" ]] && coll_trace_pair="${c}=${_tr}"
            fi
            if [[ "${c}" == *smartsim_c0* && -z "${smartsim_trace_pair}" ]] && ! _is_scale_cfg "${c}"; then
                _tr="$(_resolve_coll_trace "${s}" "${c}" "${j}")"
                [[ -n "${_tr}" ]] && smartsim_trace_pair="${c}=${_tr}"
            fi
        done
        if [[ -n "${coll_trace_pair}" && ${#timeline_phase_pairs[@]} -gt 0 ]]; then
            echo ""
            echo "Rendering P2P vs collective step-structure analysis for ${suite_name}..."
            coll_args=(); [[ -n "${coll_trace_pair}" ]] && coll_args=(--coll-trace "${coll_trace_pair}")
            ss_args=(); [[ -n "${smartsim_trace_pair}" ]] && ss_args=(--smartsim-trace "${smartsim_trace_pair}")
            OTF2_PRINT="${OTF2_PRINT:-otf2-print}" python3 "${P2P_PHASES}" \
                --output-dir "${suite_dir%/}" \
                --timeline ${timeline_phase_pairs[@]+"${timeline_phase_pairs[@]}"} \
                ${coll_args[@]+"${coll_args[@]}"} \
                ${ss_args[@]+"${ss_args[@]}"} \
                ${_hh3_pgf_args[@]+"${_hh3_pgf_args[@]}"} \
                || echo "[WARN] p2p phase analysis failed for ${suite_name}"
        fi
        fi
    fi
done
fi

[[ "${FAILED}" -eq 0 ]]
