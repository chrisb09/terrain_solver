#!/usr/bin/env python3
"""Per-run / per-node variance report for the 5-repeat AIx layout campaign.

Reads the campaign manifest CSVs (``run_id_cell,jobid,repeat,layout,comm,rc``),
resolves each wrapper job's node list via ``sacct``, computes the median ML step
time of every (-cell, repeat) run, and writes a flat CSV. This makes node-level
effects (e.g. one slow GPU node) visible instead of being hidden by the pooled
median.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path
from statistics import median

STEP_RE = re.compile(r"STEP_TIMING step=(\d+) solver=(\S+) step_ms=([0-9.]+)")


def ml_median(path: Path, skip_first: int = 1) -> float | None:
    if not path.exists():
        return None
    vals = []
    for line in path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if m and m.group(2) == "ML":
            vals.append(float(m.group(3)))
    vals = vals[skip_first:]
    return median(vals) if vals else None


def _base_job(jid: str) -> str:
    """Strip a '.batch'/'.extern' step suffix and a het group '+0'/'+1' suffix."""
    return re.sub(r"\+\d+$", "", jid.split(".")[0])


def node_lists(jobids: set[str]) -> dict[str, str]:
    if not jobids:
        return {}
    out = subprocess.run(
        ["sacct", "-j", ",".join(sorted(jobids)), "-P", "-o", "JobID,NodeList"],
        capture_output=True, text=True).stdout
    nodes: dict[str, str] = {}
    for line in out.splitlines():
        if not line or "|" not in line:
            continue
        jid, nlist = line.split("|", 1)
        base = _base_job(jid)
        if not nlist:
            continue
        # accumulate the het-group nodes (e.g. c23mm + c23g) once each
        existing = nodes.get(base, "")
        if nlist not in existing.split():
            nodes[base] = (existing + " " + nlist).strip()
    return nodes


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-run node-variance report")
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", type=Path, nargs="+", required=True)
    ap.add_argument("--logs-dir", type=Path, default=Path("mini_app/logs"))
    ap.add_argument("--skip-warmup", type=int, default=1)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    rows: list[dict[str, str | float | int]] = []
    all_jobids: set[str] = set()
    for manifest in args.manifest:
        if not manifest.exists():
            continue
        for line in manifest.read_text(errors="replace").splitlines()[1:]:
            parts = line.split(",")
            if len(parts) < 6:
                continue
            run_id, jobid, repeat, layout, comm, rc = (p.strip() for p in parts[:6])
            med = ml_median(args.logs_dir / f"mini_app_output_{run_id}.txt", args.skip_warmup)
            all_jobids.add(jobid)
            rows.append({
                "config": args.config, "layout": layout, "comm": comm,
                "repeat": int(repeat) if repeat.isdigit() else repeat,
                "run_id": run_id, "jobid": jobid, "rc": rc,
                "median_ms": "" if med is None else round(med, 4),
            })
    nodes = node_lists(all_jobids)
    for r in rows:
        r["nodes"] = nodes.get(str(r["jobid"]), "")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["config", "layout", "comm", "repeat", "run_id", "jobid", "nodes", "rc", "median_ms"]
    with args.out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["layout"], x["comm"], x["repeat"])):
            w.writerow(r)
    print(f"Wrote {args.out_csv}")
    for r in sorted(rows, key=lambda x: (x["layout"], x["comm"], x["repeat"])):
        print(f"{r['layout']:8s} {r['comm']:6s} r{r['repeat']}  "
              f"{str(r['median_ms']):>9s} ms  rc={r['rc']:>3s}  nodes={r['nodes']}")


if __name__ == "__main__":
    main()
