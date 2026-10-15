#!/usr/bin/env python3
"""Export inclusive CPU ENTER/LEAVE intervals without altering the OTF2 archive.

Times are seconds relative to ClockProperties.GlobalOffset; depth is zero-based.
Membership requires full containment. Solver locations use their own measured
wrappers; other locations use the union window across solver locations for each
call, only when exactly one window contains the interval. Blank means unknown,
not steady-state. Thread IDs come from location names, never location-ID bits.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


FIELDS = "run_id provider variant location_id world_rank thread_id solver_step ml_call_index is_warmup region depth start_s end_s duration_s source".split()
WARMUP = "solver_step_ml_warmup"
STEADY = "solver_step_ml_steady"
REF = r"<(\d+)>"
EVENT = re.compile(r"^(\w+)\s+(\d+)\s+(\d+)\s+(.*)$")


def _stream(trace, definitions=False):
    command = [os.environ.get("OTF2_PRINT", "otf2-print"), "-Werror"]
    if definitions:
        command.append("-G")
    command.append(str(trace))
    # A file avoids deadlocking on stderr while stdout is consumed incrementally.
    with tempfile.TemporaryFile(mode="w+t") as errors:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors,
                              text=True) as process:
            try:
                yield from process.stdout
            finally:
                process.stdout.close()
                if process.poll() is None:
                    process.terminate()
                status = process.wait()
            if status:
                errors.seek(0)
                raise ValueError(f"otf2-print exited {status}: {errors.read(4096)}")


def _definitions(lines):
    regions, locations, location_groups, groups, worlds = {}, {}, {}, {}, []
    clock = None
    for line in lines:
        kind = line.split(maxsplit=1)[0] if line.strip() else ""
        if kind == "CLOCK_PROPERTIES":
            match = re.search(r"Ticks per Seconds: (\d+), Global Offset: (\d+), Length: (\d+)", line)
            if not match or clock is not None:
                raise ValueError("bad/duplicate clock definition")
            clock = tuple(map(int, match.groups()))
            continue
        if kind not in {"REGION", "LOCATION", "LOCATION_GROUP", "GROUP", "COMM"}:
            continue
        match = re.match(r'\w+\s+(\d+)\s+Name: "(.*?)" <\d+>', line)
        if not match:
            raise ValueError(f"bad {kind} definition: {line.strip()}")
        ident, name = int(match[1]), match[2]
        if kind == "REGION":
            target, value = regions, name
        elif kind == "LOCATION_GROUP":
            target, value = location_groups, name
        elif kind == "LOCATION":
            details = re.search(r'Type: (\w+), # Events: (\d+), Group: .*?<(\d+)>', line)
            if not details:
                raise ValueError("bad location definition")
            target, value = locations, (name, details[1], int(details[2]), int(details[3]))
        elif kind == "GROUP":
            details = re.search(r'Type: (\w+), Paradigm: (.*?), Flags: .*?, (\d+) Members(?:: (.*))?$', line.strip())
            if not details:
                raise ValueError("bad group definition")
            members = list(map(int, re.findall(REF, details[4] or "")))
            if len(members) != int(details[3]):
                raise ValueError("bad group member count")
            target, value = groups, (details[1], details[2], members)
        else:
            if name == "MPI_COMM_WORLD":
                group = re.search(r'Group: .*?<(\d+)>', line)
                if not group:
                    raise ValueError("bad world communicator")
                worlds.append(int(group[1]))
            continue
        if ident in target:
            raise ValueError(f"duplicate {kind} {ident}")
        target[ident] = value
    if not clock or clock[0] <= 0 or clock[2] <= 0:
        raise ValueError("missing/invalid clock definition")
    if not regions or not locations or not location_groups or not groups or not worlds:
        raise ValueError("empty/missing global definitions or MPI world")
    rank_by_group = {}
    for world in worlds:
        if world not in groups or groups[world][0] != "COMM_GROUP":
            raise ValueError("invalid MPI world group")
        for rank, loc in enumerate(groups[world][2]):
            if loc not in locations:
                raise ValueError("world references undefined location")
            group = locations[loc][3]
            if group in rank_by_group and rank_by_group[group] != rank:
                raise ValueError("ambiguous world rank")
            rank_by_group[group] = rank
    mapping = {}
    for loc, (name, typ, count, group) in locations.items():
        if group not in location_groups:
            raise ValueError("undefined location group")
        thread = re.fullmatch(r"(?:Worker )?[Tt]hread\s+(\d+)", name)
        thread_id = 0 if name == "Master thread" else int(thread[1]) if thread else ""
        mapping[loc] = (rank_by_group.get(group, ""), thread_id, typ, count)
    return clock, regions, mapping


def _intervals(lines, clock, regions, locations, spool):
    stacks = defaultdict(list)
    seen, previous = Counter(), {}
    windows = defaultdict(list)
    rows = 0
    for line in lines:
        match = EVENT.match(line)
        if not match:
            if re.match(r"^(ENTER|LEAVE)\b", line):
                raise ValueError("malformed region event")
            continue
        event, loc, tick, detail = match.groups()
        loc, tick = int(loc), int(tick)
        if loc not in locations:
            raise ValueError(f"undefined event location {loc}")
        if tick < previous.get(loc, tick) or not clock[1] <= tick <= clock[1] + clock[2]:
            raise ValueError("nonmonotonic/out-of-clock event")
        previous[loc] = tick
        seen[loc] += 1
        if event not in {"ENTER", "LEAVE"}:
            continue
        region = re.fullmatch(r'Region: ".*" <(\d+)>\s*', detail)
        if not region or int(region[1]) not in regions:
            raise ValueError("malformed/undefined event region")
        region = int(region[1])
        stack = stacks[loc]
        if event == "ENTER":
            stack.append((region, tick))
            continue
        if not stack or stack[-1][0] != region:
            raise ValueError(f"unbalanced LEAVE at location {loc}")
        _, start = stack.pop()
        name = regions[region]
        if name in {WARMUP, STEADY}:
            windows[loc].append((start, tick, name))
        if locations[loc][2] == "CPU_THREAD":
            spool.write(json.dumps((loc, name, len(stack), start, tick)) + "\n")
            rows += 1
    if any(stacks.values()):
        raise ValueError("truncated stream: unclosed ENTER stacks")
    if not rows:
        raise ValueError("empty region event stream")
    for loc, (_, _, _, expected) in locations.items():
        if seen[loc] != expected:
            raise ValueError(f"truncated event stream at {loc}: {seen[loc]} != {expected}")
    return windows


def export_trace(trace, csv_path, run_id, provider, variant, expected_steady=5):
    """Return row/location/call counts; raise ValueError on invalid input.

    ``expected_steady=None`` disables only the steady-count check. Output is
    atomically replaced after validation. ``OTF2_PRINT`` selects the executable.
    ``source`` is the absolute OTF2 anchor path. No source files are written.
    """
    if expected_steady is not None and expected_steady < 0:
        raise ValueError("expected_steady must be nonnegative or None")
    trace = Path(trace).resolve()
    if trace.is_dir():
        trace = trace / "traces.otf2"
    csv_path = Path(csv_path)
    if csv_path.resolve() == trace or csv_path.resolve().is_relative_to(trace.parent):
        raise ValueError("CSV must be outside the raw OTF2 archive directory")
    clock, regions, locations = _definitions(_stream(trace, definitions=True))
    with tempfile.TemporaryFile(mode="w+t") as spool:
        windows = _intervals(_stream(trace), clock, regions, locations, spool)
        if not windows:
            raise ValueError("no solver ML wrapper locations")
        local, global_windows = {}, {}
        counts = Counter()
        for loc, calls in windows.items():
            calls.sort()
            names = [call[2] for call in calls]
            if names.count(WARMUP) != 1 or names[0] != WARMUP:
                raise ValueError(f"location {loc}: expected one leading warmup")
            if expected_steady is not None and names.count(STEADY) != expected_steady:
                raise ValueError(f"location {loc}: expected {expected_steady} steady calls, got {names.count(STEADY)}")
            if any(a[1] > b[0] for a, b in zip(calls, calls[1:])):
                raise ValueError("overlapping solver wrappers")
            local[loc] = []
            for index, (start, end, name) in enumerate(calls, 1):
                label = (index * 2, index, int(name == WARMUP))
                local[loc].append((start, end, label))
                if index in global_windows:
                    a, b, _ = global_windows[index]
                    start, end = min(a, start), max(b, end)
                global_windows[index] = (start, end, label)
                counts["warmup_calls" if name == WARMUP else "steady_calls"] += 1
        output = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", newline="", dir=csv_path.parent,
                                             prefix=".intervals-", delete=False) as stream:
                output = Path(stream.name)
                writer = csv.writer(stream)
                writer.writerow(FIELDS)
                spool.seek(0)
                used_locations = set()
                for line in spool:
                    loc, region, depth, start, end = json.loads(line)
                    candidates = local.get(loc, global_windows.values())
                    # Overlap with another call makes global membership ambiguous,
                    # even if only one of those calls fully contains this interval.
                    overlapping = [(a, b, label) for a, b, label in candidates
                                   if start < b and end > a]
                    label = ("", "", "")
                    if len(overlapping) == 1:
                        a, b, candidate = overlapping[0]
                        if a <= start and end <= b:
                            label = candidate
                    rank, thread, _, _ = locations[loc]
                    writer.writerow([run_id, provider, variant, loc, rank, thread,
                                     *label, region, depth, (start-clock[1])/clock[0],
                                     (end-clock[1])/clock[0], (end-start)/clock[0], str(trace)])
                    counts["rows"] += 1
                    counts["labeled_rows" if label[0] != "" else "unlabeled_rows"] += 1
                    used_locations.add(loc)
            os.replace(output, csv_path)
        finally:
            if output is not None:
                output.unlink(missing_ok=True)
    counts["locations"] = len(used_locations)
    counts["solver_locations"] = len(local)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("csv", type=Path)
    for name in ("run-id", "provider", "variant"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-steady", type=int, default=5)
    parser.add_argument("--no-steady-count-check", action="store_true")
    args = parser.parse_args()
    counts = export_trace(args.trace, args.csv, args.run_id, args.provider, args.variant,
                          None if args.no_steady_count_check else args.expected_steady)
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
