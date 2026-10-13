#!/usr/bin/env python3
"""Repair p2p timeline CSVs polluted by append-mode reruns.

When a scenario was re-run under the same RUN_ID, the C++ timeline writer
(std::ios::app) appended the new run's rows to the old run's file. Pollution
signature: the controller file contains a second ml_step_start row for step 0
before the first run's final step. Repair = keep only the tail from the LAST
step-0 ml_step_start row onwards, in every rank file of the timeline dir
(worker files have no ml_step_start; for them the boundary is transferred
from the controller by line arithmetic is NOT possible, so we instead detect
the restart by timestamp: rows belonging to the new run have
time_s < first_run_max_time is ambiguous. Simplification: the two runs are
separated by a constant line offset delta = (total lines - tail lines) of the
controller file; apply the same delta to worker files, which share the exact
same event-row layout per rank.)

Usage: repair_p2p_timelines.py <timeline_dir> [<timeline_dir> ...]
"""

import sys
from pathlib import Path


def repair_dir(tdir: Path) -> str:
    files = sorted(tdir.glob("aix_p2p_timeline_rank_*.csv"))
    if not files:
        return "no files"
    ctrl = None
    for f in files:
        with open(f) as s:
            header = s.readline()
            row = s.readline().rstrip("\n").split(",")
            if len(row) > 4 and row[4] == "1":
                ctrl = f
                break
    if ctrl is None:
        return "no controller file"
    with open(ctrl) as s:
        lines = s.readlines()
    starts = [i for i, l in enumerate(lines) if l.rstrip("\n").endswith(",ml_step_start") and l.startswith("0,")]
    if len(starts) <= 1:
        return "clean"
    cut = starts[-1]  # keep from the last step-0 row onwards
    delta = cut  # rows dropped from controller = run-1 block length
    with open(ctrl, "w") as s:
        s.writelines(lines[cut:])
    repaired = 1
    for f in files:
        if f == ctrl:
            continue
        with open(f) as s:
            fl = s.readlines()
        if len(fl) <= delta:
            continue
        # worker rows: header + own rows; drop the same number of run-1 rows
        data = fl[1:]
        if len(data) > delta:
            with open(f, "w") as s:
                s.write(fl[0])
                s.writelines(data[delta:])
            repaired += 1
    return f"repaired (cut {delta} run-1 rows from {repaired} files)"


def main():
    for arg in sys.argv[1:]:
        tdir = Path(arg)
        print(f"{tdir.name}: {repair_dir(tdir)}")


if __name__ == "__main__":
    main()
