#!/usr/bin/env python3
"""Aggregate the SmartSim-provider retest logs.

For each case run_id, keeps the sample set from the highest job id seen (the
corrected reruns), so the bugged first-pass `pergpu_c0` runs do not leak in.
Writes results/retest_smartsim/summary_<model>.csv and prints a combined
AIx-vs-SmartSim table.
"""
from __future__ import annotations

import glob
import re
import statistics
from collections import defaultdict
from pathlib import Path

OUT_DIR = Path("results/retest_smartsim")
RID_RE = re.compile(
    r"retest_smartsim_(?P<cfg>\S+?)_(?P<model>watercnn8k|giantmlp288)_(?P<db>shared|pergpu)_c0_r(?P<rep>\d+)$"
)
STEP_RE = re.compile(r"STEP_TIMING step=(\d+) solver=ML step_ms=([0-9.]+)")


def jobid(path: str) -> int:
    m = re.search(r"_(\d+)\.txt$", path)
    return int(m.group(1)) if m else -1


def parse() -> dict[str, list[float]]:
    best: dict[str, tuple[int, list[float]]] = {}
    for log in glob.glob("logs/retest_aix_*.txt"):
        jid = jobid(log)
        text = Path(log).read_text(errors="replace")
        parts = re.split(r"\[.*?\] case run_id=(\S+)", text)
        for i in range(1, len(parts), 2):
            rid, body = parts[i], parts[i + 1]
            vals = [float(m.group(2)) for m in STEP_RE.finditer(body) if int(m.group(1)) > 2]
            if not vals:
                continue
            if rid not in best or jid > best[rid][0]:
                best[rid] = (jid, vals)
    return {rid: v[1] for rid, v in best.items()}


def aix_medians() -> dict:
    out = {}
    for f in glob.glob("results/retest_aix/summary_*.csv"):
        for line in Path(f).read_text().splitlines()[1:]:
            p = line.split(",")
            out[(p[0], p[1], p[2], p[3])] = float(p[5])
    return out


def main() -> None:
    per_rid = parse()
    agg: dict[tuple, list[float]] = defaultdict(list)
    runs: dict[tuple, list[float]] = defaultdict(list)
    for rid, vals in per_rid.items():
        m = RID_RE.match(rid)
        if not m:
            continue
        g = m.groupdict()
        agg[(g["model"], g["cfg"], g["db"])].extend(vals)
        runs[(g["model"], g["cfg"], g["db"])].append(statistics.median(vals))

    aix = aix_medians()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = ["model,cfg,dbmode,repeats,samples,median_step_ms,min_run_median,max_run_median,"
            "aix_default_coll_ms,aix_numa_coll_ms,ratio_vs_aix_default"]
    print(f"{'model':12s} {'cfg':10s} {'aix_dflt':>9s} {'aix_numa':>9s} {'ss_shared':>10s} {'ss_pergpu':>10s} "
          f"{'shared/ax':>10s} {'pg/ax':>7s}")
    for model in sorted({k[0] for k in agg}):
        for cfg in sorted({k[1] for k in agg if k[0] == model}):
            def med(db):
                v = agg.get((model, cfg, db))
                return statistics.median(v) if v else float("nan")
            def runspread(db):
                r = runs.get((model, cfg, db))
                return (min(r), max(r)) if r else (float("nan"), float("nan"))
            a = aix.get((model, cfg, "default", "coll"), float("nan"))
            s, p = med("shared"), med("pergpu")
            print(f"{model:12s} {cfg:10s} {a:9.1f} {aix.get((model,cfg,'numa','coll'),float('nan')):9.1f} "
                  f"{s:10.1f} {p:10.1f} {s/a:10.2f} {p/a:7.2f}")
            for db in ("shared", "pergpu"):
                v = agg.get((model, cfg, db))
                if not v:
                    continue
                lo, hi = runspread(db)
                rows.append(f"{model},{cfg},{db},{len(runs[(model,cfg,db)])},{len(v)},"
                            f"{statistics.median(v):.4f},{lo:.4f},{hi:.4f},{a:.4f},"
                            f"{aix.get((model,cfg,'numa','coll'),float('nan')):.4f},"
                            f"{(statistics.median(v)/a):.4f}")
    out = OUT_DIR / "summary_combined.csv"
    out.write_text("\n".join(rows) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
