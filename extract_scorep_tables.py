#!/usr/bin/env python3
"""Callpath-aware steady Score-P tables, normalized per rank per steady step.

--campaign-root accepts campaign_MODEL_repN_JOB, campaign_MODEL_rN_JOB,
and new job_*/MODEL/repN layouts.
--results-input overrides discovery with a CSV manifest: model,config,rep,cubex.
"""
import argparse
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "CPP-ML-Interface" / "scripts"
sys.path.insert(0, str(SCRIPTS))
from analyze_cmi_scorep_profiles import parse_cubex_file  # noqa: E402

ORDER = ["AIx", "PHYDLL_CPP", "SS_DEFAULT", "TPQ1_I96", "TPQ12_I8", "TPQ96_I1"]
ALIASES = {"AIX": "AIx", "SS_TPQ1_I96_B96": "TPQ1_I96",
           "SS_TPQ12_I8_B96": "TPQ12_I8", "SS_TPQ96_I1_B96": "TPQ96_I1",
           **{c: c for c in ORDER}}
APP = {n: n for n in ("app_prepare_input", "app_provider_inference", "app_finalize_output")}
DL = {n: n for n in ("dl_input_unpack", "dl_h2d", "dl_torch_forward", "dl_d2h", "dl_output_reorder")}
DL.update({"dl_recv": "dl_recv / dl_frame_copy", "dl_frame_copy": "dl_recv / dl_frame_copy",
           "dl_send_output": "dl_send_output", "dl_send": "dl_send_output",
           "dl_input_allocate": "dl_allocate", "dl_output_allocate": "dl_allocate"})
SS = {n: n for n in ("smartsim_put_tensor", "smartsim_run_model", "smartsim_unpack_tensor",
                     "smartsim_chunk_plan", "smartsim_token_wait")}
AIX = {n: n for n in ("gatherInputData", "torchInference::forward", "scatterOutputData",
                      "cpu_chunk_slice", "cpu_chunk_assign", "h2d_copy", "d2h_copy",
                      "aix_collective_worker_wait_for_controller")}


def callpath(nodes, node):
    path = []
    while node is not None:
        path.append(node.name)
        node = nodes.get(node.parent_id)
    return tuple(reversed(path))


def extract(trees, meta, steady_steps=None):
    """Return raw callpaths and disjoint phases; reject ambiguous DL lifecycle data."""
    solver = [ns for ns in trees if any(n.name == "solver_step_ml_steady" and n.visits for n in ns.values())]
    if not solver:
        raise ValueError(f"No steady solver ranks: {meta['cubex']}")
    counts = [sum(n.visits for n in ns.values() if n.name == "solver_step_ml_steady") for ns in solver]
    if len(set(counts)) != 1 or (steady_steps is not None and counts[0] != steady_steps):
        raise ValueError(f"Inconsistent steady counts {set(counts)} (override {steady_steps})")
    steps = counts[0]
    dl = [ns for ns in trees if any(n.name == "dl_torch_forward" and n.visits for n in ns.values())]
    for ns in dl:
        for n in ns.values():
            if n.name in DL and n.visits and n.visits != steps:
                raise ValueError(f"DL {n.name}: {n.visits} visits != {steps} steady steps; cannot remove warmup from aggregated DL regions")
    raw, phases = [], []
    scopes = [("solver_steady", solver)]
    if dl:
        scopes.append(("dl_steady_count_validated", dl))
    for scope, ranks in scopes:
        denom = len(ranks) * steps
        path_totals = defaultdict(lambda: [0.0, 0.0, 0])
        for ns in ranks:
            for n in ns.values():
                path = callpath(ns, n)
                # Merged MPMD archives contain both skeletons on every rank.
                if scope.startswith("dl") != (path[0] == "phydll_dl_client"):
                    continue
                totals = path_totals[path]
                totals[0] += n.incl_time
                totals[1] += n.excl_time
                totals[2] += n.visits
        maps = {"application": APP, "provider": SS if meta['config'].startswith(("SS", "TPQ")) else AIX}
        if meta['config'] == "PHYDLL_CPP":
            maps.pop("provider")
        if scope.startswith("dl"):
            maps = {"provider_dl": DL}
        bins = defaultdict(lambda: [0.0, 0, set()])
        for path, (incl, excl, calls) in path_totals.items():
            selected = ("solver_step_ml_steady" in path and "solver_step_ml_warmup" not in path) if scope == "solver_steady" else any(n in DL for n in path)
            raw.append({**meta, "scope": scope, "rank_count": len(ranks), "steady_steps": steps,
                        "denominator": denom, "callpath": " -> ".join(path), "region": path[-1],
                        "selected_steady": selected, "raw_inclusive_s": incl, "raw_exclusive_s": excl,
                        "raw_calls": calls, "inclusive_ms_per_step": incl * 1000 / denom if selected else "",
                        "exclusive_ms_per_step": excl * 1000 / denom if selected else ""})
            if not selected:
                continue
            for view, mapping in maps.items():
                if view == "provider" and ("app_provider_inference" not in path or "app_finalize_output" in path):
                    continue
                # Deepest semantic ancestor owns self time, including MPI/library children.
                phase = next((mapping[n] for n in reversed(path) if n in mapping), "Other / overhead")
                b = bins[(view, phase)]
                b[0] += excl
                if path[-1] in mapping and not any(mapping.get(n) == phase for n in path[:-1]):
                    b[1] += calls
                if excl or calls:
                    b[2].add(path[-1])
        for view, mapping in maps.items():
            # Absent CPU transfer regions remain zero, not invented time.
            for phase in set(mapping.values()) | {"Other / overhead"}:
                bins[(view, phase)]
        for (view, phase), (seconds, calls, regions) in sorted(bins.items()):
            phases.append({**meta, "scope": scope, "rank_count": len(ranks), "steady_steps": steps,
                           "denominator": denom, "view": view, "phase": phase, "raw_total_s": seconds,
                           "raw_calls": calls, "ms_per_step": seconds * 1000 / denom,
                           "regions": "; ".join(sorted(regions))})
        if scope == "solver_steady":
            root_total = sum(v[0] for p, v in path_totals.items() if p[-1] == "solver_step_ml_steady")
            app_total = sum(v[0] for (view, _), v in bins.items() if view == "application")
            if not abs(root_total - app_total) <= max(1e-5, root_total * 1e-5):
                raise ValueError(f"Application tree sum mismatch: {app_total} != {root_total}")
    return raw, phases


def discover(roots, manifest):
    runs = []
    if manifest:
        with manifest.open(newline="") as stream:
            for r in csv.DictReader(stream):
                if r.get('status', 'SUCCESS') != 'SUCCESS':
                    continue
                if r['config'] not in ALIASES:
                    continue
                path = Path(r["cubex"])
                if not path.is_absolute():
                    # Production results.csv records only the archive basename.
                    candidate = manifest.parent / path
                    path = candidate if candidate.is_file() else manifest.parent / r['config'] / 'scorep' / path
                runs.append({"model": r["model"], "config": ALIASES.get(r["config"], r["config"]),
                             "rep": str(r["rep"]), "cubex": str(path.resolve())})
    else:
        for root in roots:
            # New production layout, including the campaign's results.csv status filter.
            rep_dirs = ([root] if re.fullmatch(r'rep\d+', root.name) else
                        list(root.glob('rep[0-9]*')) + list(root.glob('*/rep[0-9]*')) +
                        list(root.glob('job_*/*/rep[0-9]*')))
            for rep_dir in sorted(rep_dirs):
                if not rep_dir.is_dir():
                    continue
                if (rep_dir / 'results.csv').exists():
                    runs.extend(discover([], rep_dir / 'results.csv'))
                    continue
                for cfg in rep_dir.iterdir():
                    if cfg.name not in ALIASES or not cfg.is_dir():
                        continue
                    profiles = sorted(cfg.glob('scorep*/*.cubex'))
                    if len(profiles) > 1:
                        raise ValueError(f'Ambiguous profiles in {cfg}; use --results-input')
                    if profiles:
                        runs.append({'model': rep_dir.parent.name, 'config': ALIASES[cfg.name],
                                     'rep': rep_dir.name[3:], 'cubex': str(profiles[0].resolve())})
            campaigns = [root] if re.match(r"campaign_.+_(?:rep|r)\d+", root.name) else sorted(root.glob("campaign_*"))
            for campaign in campaigns:
                m = re.fullmatch(r"campaign_(.+)_(?:rep|r)(\d+)(?:_.*)?", campaign.name)
                if not m or not campaign.is_dir():
                    continue
                for cfg in campaign.iterdir():
                    if cfg.name not in ALIASES or not cfg.is_dir():
                        continue
                    profiles = sorted(cfg.glob("scorep*/*.cubex"))
                    if len(profiles) > 1:
                        raise ValueError(f"Ambiguous profiles in {cfg}; use --results-input")
                    if profiles:
                        runs.append({"model": m[1], "config": ALIASES[cfg.name], "rep": m[2],
                                     "cubex": str(profiles[0].resolve())})
    seen = set()
    for r in runs:
        key = (r['model'], r['config'], r['rep'])
        if key in seen:
            raise ValueError(f"Duplicate model/config/rep {key}; select roots or manifest explicitly")
        seen.add(key)
        if r['config'] not in ORDER:
            raise ValueError(f"Unsupported configuration: {r['config']}")
    return runs


def validate_campaign_coverage(memory_rows, phase_rows, reps, steady_steps, ranks=96):
    """Require exact terrain run/profile coverage and the expected measurement scopes."""
    expected = {(model, cfg, str(rep)) for model in ('watercnn', 'giant')
                for cfg in ORDER for rep in reps}
    memory_keys = [(r['model'], r['config'], str(r['rep'])) for r in memory_rows]
    if len(memory_keys) != len(expected) or set(memory_keys) != expected:
        raise ValueError(f'Memory/latency coverage mismatch: expected {len(expected)} runs; '
                         f'missing={sorted(expected - set(memory_keys))}, '
                         f'unexpected={sorted(set(memory_keys) - expected)}')
    for r in memory_rows:
        if int(r['solver_ranks']) != ranks or int(r['steady_steps']) != steady_steps:
            raise ValueError(f'Invalid memory/latency ranks or steady steps: {r}')
    profiles, sources, seen = defaultdict(set), {}, set()
    for r in phase_rows:
        key = (r['model'], r['config'], str(r['rep']))
        if key not in expected:
            raise ValueError(f'Unexpected Score-P run: {key}')
        if int(r['rank_count']) != ranks or int(r['steady_steps']) != steady_steps:
            raise ValueError(f'Invalid Score-P ranks or steady steps: {key}')
        scope = (r['view'], r['scope'])
        allowed = {('application', 'solver_steady')}
        allowed.add(('provider_dl', 'dl_steady_count_validated') if key[1] == 'PHYDLL_CPP'
                    else ('provider', 'solver_steady'))
        if scope not in allowed:
            raise ValueError(f'Unexpected Score-P scope: {key} {scope}')
        identity = (key, scope, r['phase'])
        if identity in seen:
            raise ValueError(f'Duplicate Score-P phase: {identity}')
        seen.add(identity)
        source = str(Path(r['cubex']).resolve())
        if key in sources and sources[key] != source:
            raise ValueError(f'Multiple Score-P profiles for {key}')
        sources[key] = source
        profiles[key].add(scope)
    if set(profiles) != expected:
        raise ValueError(f'Score-P coverage mismatch: missing={sorted(expected - set(profiles))}')
    if len(set(sources.values())) != len(expected):
        raise ValueError('Score-P profile reused across runs')
    for key, scopes in profiles.items():
        required = {('application', 'solver_steady')}
        required.add(('provider_dl', 'dl_steady_count_validated') if key[1] == 'PHYDLL_CPP'
                     else ('provider', 'solver_steady'))
        if scopes != required:
            raise ValueError(f'Missing Score-P scope: {key} {sorted(required - scopes)}')


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, action="append", help="Repeatable campaign parent or individual campaign")
    parser.add_argument("--results-input", type=Path, help="Profile manifest CSV: model,config,rep,cubex")
    parser.add_argument("--steady-steps", type=int, help="Validate expected count, not lifecycle subtraction")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "campaign_figures" / "steady_phases")
    args = parser.parse_args()
    runs = discover(args.campaign_root or [Path("/hpcwork/thes2181/mini_app")], args.results_input)
    if not runs:
        parser.error("No campaign profiles found")
    raw, phases = [], []
    for meta in runs:
        r, p = extract(parse_cubex_file(Path(meta['cubex'])), meta, args.steady_steps)
        raw.extend(r)
        phases.extend(p)
        print(f"{meta['model']} {meta['config']} rep={meta['rep']}: ranks={p[0]['rank_count']} steps={p[0]['steady_steps']}")
    groups = defaultdict(list)
    for r in phases:
        groups[(r['model'], r['config'], r['view'], r['phase'], r['scope'])].append(r)
    means = []
    for (model, cfg, view, phase, scope), rows in sorted(groups.items()):
        means.append({"model": model, "config": cfg, "view": view, "phase": phase, "scope": scope,
                      "rep_count": len(rows), "mean_ms_per_step": statistics.mean(r['ms_per_step'] for r in rows),
                      "std_ms_per_step": statistics.stdev(r['ms_per_step'] for r in rows) if len(rows) > 1 else 0,
                      "raw_total_s": sum(r['raw_total_s'] for r in rows), "raw_calls": sum(r['raw_calls'] for r in rows),
                      "rank_counts": ";".join(str(r['rank_count']) for r in rows),
                      "steady_step_counts": ";".join(str(r['steady_steps']) for r in rows)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("raw_callpaths", raw), ("phase_replicates", phases), ("phase_means", means)]:
        write_csv(args.output_dir / f"{name}.csv", rows)
    lines = ["# Steady Score-P Phases", "",
             "Normalization: sum seconds / (scope rank count * steady steps) * 1000. Arithmetic mean across replicates; never per-call averaging.",
             "Solver selection requires solver_step_ml_steady ancestry and excludes solver_step_ml_warmup. Stacks assign exclusive callpath time once to its deepest semantic ancestor; nested app_finalize_output is not double-counted in app_provider_inference.",
             "PhyDLL uses only DL-rank regions, with visits validated against solver steady steps. DL receive includes DL-side waiting/handling and frame copy; no solver send/receive enters this stack. dl_send_output includes dl_send once. DL root/lifecycle self time is excluded.",
             "SmartSim smartsim_unpack_tensor covers retrieval plus unpack. Model request is client elapsed time, not isolated server compute. Missing transfers are zero; measured CPU staging regions are retained.",
             "Raw lifecycle totals/calls are preserved in raw_callpaths.csv with selected_steady=False. Phase calls count semantic anchors, not nested MPI calls. Other / overhead preserves unattributed steady self time.", "",
             "| Model | Config | View | Phase | Scope | Reps | Mean ms/step | Raw total s | Raw calls | Rank counts | Step counts |",
             "|---|---|---|---|---|---:|---:|---:|---:|---|---|"]
    for r in means:
        lines.append(f"| {r['model']} | {r['config']} | {r['view']} | {r['phase']} | {r['scope']} | {r['rep_count']} | {r['mean_ms_per_step']:.6f} | {r['raw_total_s']:.9f} | {r['raw_calls']} | {r['rank_counts']} | {r['steady_step_counts']} |")
    lines.extend(["", "## Profile Sources", "", "| Model | Config | Rep | Cubex |", "|---|---|---|---|"])
    lines.extend(f"| {r['model']} | {r['config']} | {r['rep']} | `{r['cubex']}` |" for r in runs)
    (args.output_dir / "phase_summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(runs)} profiles, {len(raw)} callpaths, {len(phases)} replicate phases to {args.output_dir}")


if __name__ == "__main__":
    main()
