#!/usr/bin/env python3
"""Controlled terrain diagnostic schedules and replicate-level latency analysis."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def plan(suite, cycles=0, repeats=2):
    if cycles < 0 or repeats < 2:
        raise ValueError("cycles must be >=0; SmartSim repeats must be >=2")
    rows = []
    if suite in ("aix", "phydll"):
        for cycle in range(cycles or (2 if suite == "aix" else 1)):
            for offset, condition in enumerate(("off", "on", "on", "off")):
                rep = cycle * 2 + offset // 2 + 1
                rows.append((rep, suite, condition, 0, 1,
                             int(suite == "aix" and condition == "on"),
                             int(suite == "phydll" and condition == "on")))
    else:
        treatments = [(f"TPQ{tpq}_I{intra}", condition, tpq, intra)
                      for tpq, intra in ((1, 96), (12, 8), (96, 1))
                      for condition in ("off", "on")]
        for rep in range(1, repeats + 1):
            for cfg, condition, tpq, intra in treatments[::1 if rep % 2 else -1]:
                rows.append((rep, cfg, condition, tpq, intra, 0, 0))
    return [(f"run{i:02d}", *row) for i, row in enumerate(rows, 1)]


def timings(path, steps):
    expected = set(range(4, steps + 1, 2))
    found = {}
    for line in Path(path).read_text().splitlines():
        if not line.startswith("STEP_TIMING "):
            continue
        fields = dict(item.split("=", 1) for item in line.split()[1:] if "=" in item)
        step = int(fields["step"])
        if fields.get("solver") != "ML" or step == 2:
            continue
        if step not in expected or step in found:
            raise ValueError(f"Unexpected/duplicate steady step {step}: {path}")
        values = {key: float(fields[key]) for key in ("step_ms", "rank_min_ms", "rank_max_ms")}
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError(f"Invalid timing: {path}: {values}")
        if values["rank_min_ms"] > values["rank_max_ms"]:
            raise ValueError(f"Invalid rank range: {path}")
        found[step] = {"step": step, **values}
    if set(found) != expected:
        raise ValueError(f"Steady coverage {sorted(found)} != {sorted(expected)}: {path}")
    return [found[step] for step in sorted(found)]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hash_phydll_libraries(sources):
    paths = set()
    for source in sources:
        matches = re.findall(r"^\s*libphydll\.so(?:\.\d+)*\s+=>\s+(\S+)", source.read_text(), re.MULTILINE)
        if not matches or any(not Path(p).is_absolute() or not Path(p).is_file() for p in matches):
            raise ValueError(f"No resolved readable libphydll.so in {source}")
        paths.update(Path(p).resolve() for p in matches)
    for path in sorted(paths):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        print(f"{digest.hexdigest()}  {path}")


def aggregate(root, scorep=False):
    with (root / "results.csv").open(newline="") as stream:
        runs = list(csv.DictReader(stream))
    if not runs or any(r["status"] != "SUCCESS" for r in runs):
        raise ValueError("Empty or failed suite; inspect results.csv, do not aggregate partial evidence")
    scheduled = [line.split("\t") for line in (root / "plan.tsv").read_text().splitlines()]
    actual = [[r[k] for k in ("run", "rep", "config", "condition", "tpq", "intra", "sampler", "barriers")]
              for r in runs]
    if actual != scheduled:
        raise ValueError("Run manifest does not match the complete ordered plan")
    all_steps, per_run, profile_rows = [], [], []
    paired = defaultdict(dict)
    for run in runs:
        values = timings(run["log"], int(run["steps"]))
        meta = {k: run[k] for k in ("model", "suite", "run", "rep", "config", "condition")}
        all_steps.extend({**meta, **v} for v in values)
        summary = {**meta, "steady_steps": len(values), "wall_s": float(run["wall_s"])}
        for metric in ("step_ms", "rank_min_ms", "rank_max_ms"):
            summary[f"mean_{metric}"] = statistics.mean(v[metric] for v in values)
            summary[f"median_{metric}"] = statistics.median(v[metric] for v in values)
        per_run.append(summary)
        key = tuple(run[k] for k in ("model", "suite", "config", "rep"))
        if run["condition"] in paired[key]:
            raise ValueError(f"Duplicate paired condition: {key}")
        paired[key][run["condition"]] = summary
        if scorep:
            # Reuse callpath and MPMD rank-scope validation, not the campaign discovery.
            from extract_scorep_tables import extract, parse_cubex_file
            cfg = "PHYDLL_CPP" if run["suite"] == "phydll" else "AIx" if run["suite"] == "aix" else run["config"]
            raw, _ = extract(parse_cubex_file(Path(run["cubex"])),
                             {"model": run["model"], "config": cfg, "rep": run["rep"], "cubex": run["cubex"]},
                             steady_steps=len(values))
            for row in raw:
                if row["selected_steady"] and (row["region"] == "MPI_Barrier" or "forward" in row["region"]):
                    if row["rank_count"] != 96:
                        raise ValueError(f"Expected 96 ranks in scope: {row}")
                    profile_rows.append({**row, **meta})
    differences = []
    for key, conditions in paired.items():
        if set(conditions) != {"off", "on"}:
            raise ValueError(f"Unmatched conditions: {key}")
        a, b = conditions["off"], conditions["on"]
        differences.append({**dict(zip(("model", "suite", "config", "rep"), key)),
                            "off_mean_ms": a["mean_step_ms"], "on_mean_ms": b["mean_step_ms"],
                            "on_minus_off_mean_ms": b["mean_step_ms"] - a["mean_step_ms"],
                            "on_over_off_mean": b["mean_step_ms"] / a["mean_step_ms"],
                            "off_median_ms": a["median_step_ms"], "on_median_ms": b["median_step_ms"],
                            "on_minus_off_median_ms": b["median_step_ms"] - a["median_step_ms"]})
    groups = defaultdict(list)
    for row in per_run:
        groups[tuple(row[k] for k in ("model", "suite", "config", "condition"))].append(row)
    conditions = []
    for key, rows in groups.items():
        conditions.append({**dict(zip(("model", "suite", "config", "condition"), key)),
                           "replicates": len(rows),
                           "mean_of_rep_means_ms": statistics.mean(r["mean_step_ms"] for r in rows),
                           "median_of_rep_means_ms": statistics.median(r["mean_step_ms"] for r in rows),
                           "mean_of_rep_medians_ms": statistics.mean(r["median_step_ms"] for r in rows),
                           "median_of_rep_medians_ms": statistics.median(r["median_step_ms"] for r in rows)})
    pair_groups = defaultdict(list)
    for row in differences:
        pair_groups[tuple(row[k] for k in ("model", "suite", "config"))].append(row)
    pair_summary = []
    for key, rows in pair_groups.items():
        pair_summary.append({**dict(zip(("model", "suite", "config"), key)), "pairs": len(rows),
                             "mean_paired_delta_ms": statistics.mean(r["on_minus_off_mean_ms"] for r in rows),
                             "median_paired_delta_ms": statistics.median(r["on_minus_off_mean_ms"] for r in rows),
                             "median_paired_ratio": statistics.median(r["on_over_off_mean"] for r in rows)})
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    for name, rows in (("steady_steps", all_steps), ("per_run", per_run), ("paired", differences),
                       ("conditions", conditions), ("paired_summary", pair_summary),
                       ("scorep_forward_barrier", profile_rows)):
        write_csv(out / f"{name}.csv", rows)
    report = ["# Terrain CPU Diagnostics", "", "Replicate means/medians are the analysis units, not individual steps or ranks.",
              "No confidence intervals: only 2-4 matched replicates, serially run on one node.", "",
              "| Model | Suite | Config | Condition | Reps | Mean of rep means (ms) | Median of rep medians (ms) |",
              "|---|---|---|---|---:|---:|---:|"]
    for row in conditions:
        report.append(f"| {row['model']} | {row['suite']} | {row['config']} | {row['condition']} | {row['replicates']} | "
                      f"{row['mean_of_rep_means_ms']:.3f} | {row['median_of_rep_medians_ms']:.3f} |")
    report += ["", "| Config | Pairs | Mean paired delta (ms) | Median paired delta (ms) | Median on/off ratio |",
               "|---|---:|---:|---:|---:|"]
    for row in pair_summary:
        report.append(f"| {row['config']} | {row['pairs']} | {row['mean_paired_delta_ms']:.3f} | "
                      f"{row['median_paired_delta_ms']:.3f} | {row['median_paired_ratio']:.3f} |")
    report += ["", "Paired on-minus-off differences and ratios: `paired.csv`.",
               "On means sampler enabled (AIx), logging barriers enabled (PhyDLL), or DB batching enabled (SmartSim).",
               "Score-P inclusive forward/barrier times are per rank per steady step, not wall time; nested rows must not be summed.",
               "Score-P DL scope is accepted only when its visit counts match steady steps; no warmup subtraction is guessed.",
               "Score-P extraction: " + ("requested; see scorep_forward_barrier.csv" if scorep else "not requested (use --scorep).")]
    (out / "report.md").write_text("\n".join(report) + "\n")
    print(out / "report.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", choices=("aix", "phydll", "smartsim"))
    mode.add_argument("--root", type=Path)
    mode.add_argument("--validate-log", type=Path)
    mode.add_argument("--validate-memory", type=Path)
    mode.add_argument("--hash-phydll-libraries", type=Path, nargs="+")
    parser.add_argument("--model", choices=("watercnn", "giant"))
    parser.add_argument("--cycles", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--scorep", action="store_true")
    args = parser.parse_args()
    if args.steps < 6 or args.steps % 2:
        parser.error("steps must be even and >=6 (10 gives four steady ML steps)")
    if args.plan:
        if not args.model:
            parser.error("--plan requires --model")
        for row in plan(args.plan, args.cycles, args.repeats):
            print("\t".join(map(str, row)))
    elif args.hash_phydll_libraries:
        hash_phydll_libraries(args.hash_phydll_libraries)
    elif args.validate_log:
        timings(args.validate_log, args.steps)
    elif args.validate_memory:
        data = json.loads(args.validate_memory.read_text())
        if not data["coverage_ok"] or data["live_read_errors"] or data["cleanup_sigkill_pids"]:
            raise ValueError("Invalid sampler coverage/read quality/forced cleanup")
    else:
        aggregate(args.root, args.scorep)


if __name__ == "__main__":
    main()
