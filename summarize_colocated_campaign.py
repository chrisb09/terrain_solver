#!/usr/bin/env python3
"""Reduce calls within each job first, then compare paired job summaries."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics


def summarize(root, platform, phase):
    jobs = []
    excluded = []
    conditions = ("direct_nopreload", "direct_preload", "cmi_baseline", "cmi_profiling", "cmi_tracing")
    structured = re.compile(r"^STEP_TIMING step=(\d+) solver=(ML|Regular) step_ms=([0-9.eE+-]+)")
    legacy = re.compile(r"^Step\s+(\d+),\s*(ML|Regular),.*time:\s*([0-9.eE+-]+)\s*ms")
    manifests = sorted(root.glob(f"{platform}_{phase}_r*_*/runs.tsv"))
    for manifest in manifests:
        try:
            with manifest.open() as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            if len(rows) != 5 or {row["condition"] for row in rows} != set(conditions):
                raise ValueError("incomplete five-condition job")
            means = {}
            counts = {}
            for row in rows:
                if row["status"] != "0":
                    raise ValueError(f"failed {row['condition']}")
                new, old = {}, {}
                for line in Path(row["log"]).read_text(errors="replace").splitlines():
                    match = structured.match(line)
                    if match:
                        target = new
                    else:
                        match = legacy.match(line)
                        target = old
                    if match:
                        step, kind, value = int(match[1]), match[2], float(match[3])
                        if step in target:
                            raise ValueError("duplicate step record")
                        if not math.isfinite(value) or value <= 0:
                            raise ValueError("invalid step timing")
                        target[step] = (kind, value)
                records = new or old
                if len(records) != int(row["steps"]):
                    raise ValueError(f"incomplete log for {row['condition']}")
                calls = [value for _, (kind, value) in sorted(records.items()) if kind == "ML"]
                calls = calls[int(row["warmup"]):]
                if not calls:
                    raise ValueError("no post-warmup ML calls")
                means[row["condition"]] = statistics.mean(calls)
                counts[row["condition"]] = len(calls)
            jobs.append({"job": manifest.parent.name, "mean_ml_step_ms": means, "measured_calls": counts})
        except (OSError, ValueError, KeyError) as error:
            excluded.append({"job": manifest.parent.name, "reason": str(error)})
    comparisons = []
    for control, treatment in (
        ("direct_nopreload", "direct_preload"),
        ("direct_nopreload", "cmi_baseline"),
        ("cmi_baseline", "cmi_profiling"),
        ("cmi_baseline", "cmi_tracing"),
    ):
        differences = [job["mean_ml_step_ms"][treatment] - job["mean_ml_step_ms"][control] for job in jobs]
        percents = [100 * (job["mean_ml_step_ms"][treatment] / job["mean_ml_step_ms"][control] - 1) for job in jobs]
        n = len(differences)
        mean = statistics.mean(differences) if n else None
        sd = statistics.stdev(differences) if n > 1 else None
        # Campaign has at most five independent jobs, Student-t df = n - 1.
        critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n)
        half = critical * sd / math.sqrt(n) if critical is not None else None
        comparisons.append({"control": control, "treatment": treatment, "paired_jobs": n,
                            "mean_difference_ms": mean, "sd_difference_ms": sd,
                            "ci95_difference_ms": [mean - half, mean + half] if half is not None else None,
                            "mean_paired_percent": statistics.mean(percents) if n else None})
    return {"platform": platform, "phase": phase, "warmup_unit": "first ML calls, not solver steps",
            "jobs": jobs, "excluded_jobs": excluded, "paired_comparisons": comparisons}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_root", type=Path)
    parser.add_argument("platform", choices=("cpu", "gpu"))
    parser.add_argument("phase", choices=("smoke", "pilot", "production"))
    args = parser.parse_args()
    result = summarize(args.campaign_root, args.platform, args.phase)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result["jobs"] and not result["excluded_jobs"] else 1)
