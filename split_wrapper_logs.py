#!/usr/bin/env python3
"""Split wrapped campaign logs into per-cell logs.

The 5-repeat campaign wrappers run the frozen job driver 9 times in one
allocation, so all STEP_TIMING output lands in the wrapper's stdout
(``logs/mini_app_output_5r_<cfg>_<jobid>.txt``). The wrapper prints a banner
before every cell:

    [2026-09-18 17:49:03] cell layout=default comm=coll run_id=5r_96c4g_default_coll_r1

This script splits such a log on those banners and writes one
``logs/mini_app_output_<run_id>.txt`` per cell, so the usual analysis tooling
works unchanged.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

BANNER = re.compile(r"\] cell layout=(\S+) comm=(\S+) run_id=(\S+)")


def split_log(log_path: Path, out_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    fh = None
    current = None
    with log_path.open(errors="replace") as src:
        for line in src:
            m = BANNER.search(line)
            if m:
                if fh is not None:
                    fh.close()
                current = m.group(3)
                counts[current] = 0
                fh = (out_dir / f"mini_app_output_{current}.txt").open("w")
                continue
            if fh is not None:
                fh.write(line)
                counts[current] += 1
    if fh is not None:
        fh.close()
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Split wrapper logs into per-cell logs")
    ap.add_argument("--out-dir", type=Path, default=Path("mini_app/logs"))
    ap.add_argument("logs", type=Path, nargs="+")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for log in args.logs:
        counts = split_log(log, args.out_dir)
        total = sum(counts.values())
        print(f"{log.name}: {len(counts)} cells, {total} lines -> {args.out_dir}")
        for run_id, n in counts.items():
            print(f"    {run_id}: {n} lines")


if __name__ == "__main__":
    main()
