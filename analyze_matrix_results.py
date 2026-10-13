#!/usr/bin/env python3
"""Parse and tabulate a 3x3 AIx layout matrix (3 layouts x 3 comm modes) from job logs.

Two input modes:

* ``--job-map default_coll=4069576,...``  (single run per cell, original mode)
* ``--runs-csv runs_r1.csv runs_r2.csv ...`` where each manifest has rows
  ``run_id_cell,jobid,repeat,layout,comm,rc`` (as written by the 5-repeat
  campaign wrappers). All runs of a cell are concatenated, so e.g. 5 runs x
  100 steps become one 500-sample distribution.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import median, mean, stdev

STEP_RE = re.compile(r"STEP_TIMING step=(\d+) solver=(\S+) step_ms=([0-9.]+)")

LAYOUTS = ["default", "affine", "numa"]
COMMS = ["coll", "p2p1", "p2p23"]
COMM_HEADERS = ["Collective", "P2P (credit=1)", "P2P (full credits)"]


def extract_ml_steps(path: Path, skip_first: int = 1) -> list[float]:
    if not path.exists():
        return []
    steps: list[tuple[int, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        match = STEP_RE.search(line)
        if match and match.group(2) == "ML":
            steps.append((int(match.group(1)), float(match.group(3))))
    return [v for _, v in steps[skip_first:]]


def parse_job_map(raw: str | None) -> dict[str, int]:
    job_map: dict[str, int] = {}
    if raw:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            key, value = item.split("=")
            job_map[key.strip()] = int(value)
    return job_map


def parse_runs_csv(paths: list[Path], include_failed: bool = False) -> dict[tuple[str, str], list[str]]:
    """Return {(layout, comm): [run_id, ...]} from campaign manifest CSVs."""
    cells: dict[tuple[str, str], list[str]] = {}
    for path in paths:
        if not path.exists():
            continue
        lines = path.read_text(errors="replace").splitlines()
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) < 6:
                continue
            run_id, _jobid, _repeat, layout, comm, rc = (p.strip() for p in parts[:6])
            if not include_failed and rc not in ("0", ""):
                continue
            if comm == "p2p24":  # super-high-res alias of full credits
                comm = "p2p23"
            cells.setdefault((layout, comm), []).append(run_id)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description="3x3 AIx layout matrix summary")
    parser.add_argument("--logs-dir", type=Path, default=Path("mini_app/logs"),
                        help="Directory containing mini_app_output_*.txt")
    parser.add_argument("--job-map", type=str, default=None,
                        help="Comma-separated key=jobid pairs, e.g. default_coll=4069576,...")
    parser.add_argument("--runs-csv", type=Path, nargs="*", default=None,
                        help="One or more campaign manifest CSVs (run_id_cell,jobid,repeat,layout,comm,rc)")
    parser.add_argument("--include-failed-runs", action="store_true",
                        help="With --runs-csv, also include runs whose rc != 0")
    parser.add_argument("--config-label", type=str, default="96c4g_c23g",
                        help="Config identifier shown in the header")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="Optional path to write a per-cell statistics CSV")
    parser.add_argument("--skip-warmup", type=int, default=1,
                        help="Number of initial ML steps to skip as warmup per run")
    args = parser.parse_args()

    job_map = parse_job_map(args.job_map)
    runs_cells: dict[tuple[str, str], list[str]] | None = None
    if args.runs_csv:
        runs_cells = parse_runs_csv(args.runs_csv, include_failed=args.include_failed_runs)

    print("=" * 78)
    print(f"3x3 AIx Layout Matrix [{args.config_label}]: Median Steady-State ML Step Time (ms)")
    print("=" * 78)
    print(f"{'Layout':<15s}" + "".join(f"{h:>20s}" for h in COMM_HEADERS))
    print("-" * 78)

    csv_rows = ["layout,comm_mode,job_id,runs,samples,median_ms,mean_ms,stdev_ms,min_ms,p25_ms,p75_ms,max_ms"]
    for layout in LAYOUTS:
        row = [f"{layout:<15s}"]
        for comm in COMMS:
            vals: list[float] = []
            runs_count = 0
            job_label: str
            if runs_cells is not None:
                run_ids = runs_cells.get((layout, comm), [])
                runs_count = len(run_ids)
                for run_id in run_ids:
                    vals.extend(extract_ml_steps(args.logs_dir / f"mini_app_output_{run_id}.txt",
                                                 skip_first=args.skip_warmup))
                job_label = "+".join(run_ids) if run_ids else ""
                if not run_ids:
                    row.append(f"{'N/A':>20s}")
                    continue
            else:
                key = f"{layout}_{comm}"
                jid = job_map.get(key)
                if not jid:
                    row.append(f"{'N/A':>20s}")
                    continue
                runs_count = 1
                job_label = str(jid)
                vals = extract_ml_steps(args.logs_dir / f"mini_app_output_{jid}.txt",
                                        skip_first=args.skip_warmup)
                if not vals:
                    row.append(f"{f'Pending ({jid})':>20s}")
                    continue
            ordered = sorted(vals)
            n = len(ordered)
            p25 = ordered[n // 4]
            p75 = ordered[(3 * n) // 4]
            sd = stdev(ordered) if n > 1 else 0.0
            row.append(f"{median(ordered):15.3f} ms ({n})")
            csv_rows.append(
                f"{layout},{comm},{job_label},{runs_count},{n},{median(ordered):.4f},"
                f"{mean(ordered):.4f},{sd:.4f},{min(ordered):.4f},{p25:.4f},{p75:.4f},{max(ordered):.4f}"
            )
        print("".join(row))
    print("=" * 78)

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        args.out_csv.write_text("\n".join(csv_rows) + "\n")
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
