#!/usr/bin/env python3
"""Aggregate the AIx retest logs into per-(model,cfg,layout,comm) step-time stats.

Reads logs/retest_aix_*.txt, segments each log by the campaign's
"[ts] cell run_id=<id>" markers, extracts STEP_TIMING ML samples (skipping the
warmup step <=2), and reports the median synchronized makespan (step_ms) and the
median per-rank max work (rank_max_ms). Writes results/retest_aix/summary_<model>.csv.
"""
from __future__ import annotations

import glob
import re
import statistics
from collections import defaultdict
from pathlib import Path

LOG_GLOB = "logs/retest_aix_*.txt"
OUT_DIR = Path("results/retest_aix")
RID_RE = re.compile(r"retest_(?P<cfg>\S+?)_(?P<model>watercnn8k|giantmlp288)_(?P<layout>\w+?)_(?P<comm>\w+?)_r(?P<rep>\d+)$")
STEP_RE = re.compile(
    r"STEP_TIMING step=(\d+) solver=ML step_ms=([0-9.]+) local_moved=[-0-9.eE]+"
    r"(?: rank_min_ms=([0-9.]+) rank_max_ms=([0-9.]+))?"
)
LAYOUTS = ["default", "affine", "numa"]
COMMS = ["coll", "p2p1", "p2p24"]


def parse() -> dict:
    per_run: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for log in glob.glob(LOG_GLOB):
        text = Path(log).read_text(errors="replace")
        parts = re.split(r"\[.*?\] cell run_id=(\S+)", text)
        for i in range(1, len(parts), 2):
            rid, body = parts[i], parts[i + 1]
            for m in STEP_RE.finditer(body):
                if int(m.group(1)) <= 2:
                    continue
                step = float(m.group(2))
                rmax = float(m.group(4)) if m.group(4) else float("nan")
                per_run[rid].append((step, rmax))

    agg: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for rid, samples in per_run.items():
        m = RID_RE.match(rid)
        if not m:
            continue
        g = m.groupdict()
        agg[(g["model"], g["cfg"], g["layout"], g["comm"])].extend(samples)
    return agg


def main() -> None:
    agg = parse()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = sorted({k[0] for k in agg})
    for model in models:
        rows = ["model,cfg,layout,comm,samples,median_step_ms,mean_step_ms,stdev_step_ms,min_step_ms,max_step_ms,median_rank_max_ms"]
        print("=" * 96)
        print(f"{model}: synchronized makespan (median ML step_ms over 3 repeats)")
        print(f"{'cfg':10s} {'layout':8s} {'coll':>10s} {'p2p1':>10s} {'p2p24':>10s}")
        for cfg in sorted({k[1] for k in agg if k[0] == model}):
            for layout in LAYOUTS:
                cells = []
                for comm in COMMS:
                    samples = agg.get((model, cfg, layout, comm), [])
                    steps = [s for s, _ in samples]
                    rmaxs = [r for _, r in samples if r == r]
                    if steps:
                        med = statistics.median(steps)
                        cells.append(f"{med:10.1f}")
                        rows.append(
                            f"{model},{cfg},{layout},{comm},{len(steps)},{med:.4f},"
                            f"{statistics.mean(steps):.4f},{statistics.stdev(steps):.4f},"
                            f"{min(steps):.4f},{max(steps):.4f},"
                            f"{(statistics.median(rmaxs) if rmaxs else float('nan')):.4f}"
                        )
                    else:
                        cells.append(f"{'-':>10s}")
                print(f"{cfg:10s} {layout:8s} " + " ".join(cells))
        out = OUT_DIR / f"summary_{model}.csv"
        out.write_text("\n".join(rows) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
