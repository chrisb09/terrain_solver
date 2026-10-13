#!/usr/bin/env python3
"""Compare ML step times from mini_app logs via STEP_TIMING lines."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import median

STEP_RE = re.compile(r"STEP_TIMING step=(\d+) solver=(\S+) step_ms=([0-9.]+)")


def ml_steps(path: Path) -> list[tuple[int, float]]:
    steps = []
    for line in path.read_text(errors="replace").splitlines():
        match = STEP_RE.search(line)
        if match and match.group(2) == "ML":
            steps.append((int(match.group(1)), float(match.group(3))))
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--skip-first-ml", type=int, default=1,
                        help="number of leading ML steps to discard as warmup")
    args = parser.parse_args()

    print(f"{'log':46s} {'ML steps':>8s} {'median_ms':>10s} {'min_ms':>10s} {'max_ms':>10s}")
    for log in args.logs:
        path = Path(log)
        if not path.exists():
            print(f"{path.name:46s} {'MISSING':>8s}")
            continue
        steps = ml_steps(path)
        measured = [value for _, value in steps[args.skip_first_ml:]]
        if not measured:
            print(f"{path.name:46s} {'no ML steps found':>8s}")
            continue
        print(f"{path.name:46s} {len(measured):8d} {median(measured):10.3f} "
              f"{min(measured):10.3f} {max(measured):10.3f}")


if __name__ == "__main__":
    main()
