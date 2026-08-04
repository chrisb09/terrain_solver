#!/usr/bin/env python3
"""Validate SmartSim balanced-key routing and summarize steady ML step time."""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


ASSIGNMENT = re.compile(r"SMARTSIM_KEY_BALANCE_ASSIGNMENT rank=(\d+) target_shard=(\d+) gpu=(\d+)")
TAG = re.compile(
    r"SMARTSIM_KEY_BALANCE shard=(\d+) tag=\S+ slot=(\d+) expected_slot_range=\[(\d+),(\d+)\]"
)
STEP = re.compile(r"STEP_TIMING step=(\d+) solver=ML step_ms=([0-9.]+)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--ranks", type=int, default=96)
    parser.add_argument("--timing-only", action="store_true")
    args = parser.parse_args()

    lines = args.log.read_text(errors="replace").splitlines()
    tags = [
        tuple(map(int, match.groups()))
        for line in lines
        if (match := TAG.search(line))
    ]
    assignments = [tuple(map(int, match.groups())) for line in lines if (match := ASSIGNMENT.search(line))]
    steps = [(int(match.group(1)), float(match.group(2))) for line in lines if (match := STEP.search(line))]

    failures = []
    if not args.timing_only:
        if len(tags) < args.nodes:
            failures.append(f"found {len(tags)} tag records; expected at least {args.nodes}")
        for shard, slot, first, last in tags:
            expected_first = (shard * 16384) // args.nodes
            expected_last = (((shard + 1) * 16384) // args.nodes) - 1
            if slot < expected_first or slot > expected_last or (first, last) != (expected_first, expected_last):
                failures.append(
                    f"tag for shard {shard} has slot {slot}, expected [{expected_first},{expected_last}]"
                )

        by_rank = {rank: (shard, gpu) for rank, shard, gpu in assignments}
        if len(by_rank) != args.ranks:
            failures.append(f"found assignments for {len(by_rank)} ranks; expected {args.ranks}")
        for rank, (shard, gpu) in by_rank.items():
            expected_shard = (rank // args.gpus) % args.nodes
            if (shard, gpu) != (expected_shard, rank % args.gpus):
                failures.append(
                    f"rank {rank} maps to shard/GPU {shard}/{gpu}; expected {expected_shard}/{rank % args.gpus}"
                )

        counts = Counter(by_rank.values())
        expected_pairs = [(shard, gpu) for shard in range(args.nodes) for gpu in range(args.gpus)]
        if any(pair not in counts for pair in expected_pairs):
            failures.append("at least one shard/GPU pair has no requester")
        if counts:
            minimum, maximum = min(counts.values()), max(counts.values())
            if maximum - minimum > 1:
                failures.append(f"shard/GPU loads range from {minimum} to {maximum}, not balanced")
            print(f"Shard/GPU loads: min={minimum} max={maximum} pairs={len(counts)}")

    steady_steps = [milliseconds for step, milliseconds in steps if step > 2]
    if steady_steps:
        print(
            "Steady ML STEP_TIMING: "
            f"count={len(steady_steps)} mean_ms={sum(steady_steps) / len(steady_steps):.3f} "
            f"min_ms={min(steady_steps):.3f} max_ms={max(steady_steps):.3f}"
        )
    else:
        print("No steady ML STEP_TIMING records found.")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("PASS: timing parsed." if args.timing_only else "PASS: Redis tags and rank-to-shard/GPU assignments are balanced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
