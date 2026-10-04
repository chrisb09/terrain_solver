#!/usr/bin/env python3
"""External, single-node, configuration-scoped /proc memory sampling."""
import argparse
import csv
import errno
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import time

ROLES = ("solver", "dl", "db", "support")
LIFECYCLE_ERRORS = ("exited", "pid_reused", "zombie")


def identity(directory):
    # comm can contain spaces and parentheses; fields after its final ')' are fixed.
    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[1]), int(fields[19]), fields[0]  # ppid, starttime, state


def snapshot(proc=Path("/proc")):
    records = {}
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            ppid, start, state = identity(directory)
            if state in ("Z", "X", "x"):
                continue
            try:
                env = dict(item.split(b"=", 1) for item in (directory / "environ").read_bytes().split(b"\0") if b"=" in item)
            except OSError:
                env = {}  # Descendant/cgroup discovery can still identify this PID.
            groups = (directory / "cgroup").read_text()
            unified = next((line[3:] for line in groups.splitlines() if line.startswith("0::")), "")
            records[int(directory.name)] = dict(
                ppid=ppid, start=start, env=env, groups=groups, cgroup=unified,
                exe=os.path.realpath(directory / "exe"),
                cmd=(directory / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace"))
        except (OSError, ValueError, IndexError):
            continue
    return records


def in_job(record, job):
    if not job:
        return False
    # New Slurm installations need not include job IDs in cgroup paths.
    env_job = record["env"].get(b"SLURM_JOB_ID", b"").decode()
    return env_job == job or bool(re.search(r"(?:job_|job-)%s(?:[/.\s]|$)" % re.escape(job), record["groups"]))


def select(records, tracked, baseline, token, job, solver, dl, port, exclude):
    chosen = {}
    for pid, record in records.items():
        if pid == exclude:
            continue
        key = (pid, record["start"])
        tagged = record["env"].get(b"TERRAIN_MEMORY_TOKEN") == token.encode()
        redis = Path(record["exe"]).name in ("redis-server", "redis-stack-server", "valkey-server")
        port_match = port and re.search(r"(?:[: ]|--port[ =])%d(?:\s|$)" % port, record["cmd"])
        db_fallback = redis and port_match and in_job(record, job) and key not in baseline
        if not (tagged or key in tracked or db_fallback):
            continue
        role = "support"
        if record["exe"] == solver:
            role = "solver"
        elif record["exe"] == dl:
            role = "dl"
        elif redis:
            role = "db"
        elif key in tracked:
            role = tracked[key]
        chosen[pid] = (role, "token" if tagged else "tracked" if key in tracked else "job+port")
    # MPI/Slurm workers may be reparented; keep previously seen PID/start pairs.
    # Children inherit their nearest application role, not the launcher's role.
    changed = True
    while changed:
        changed = False
        for pid, record in records.items():
            parent = chosen.get(record["ppid"])
            if pid == exclude or not parent:
                continue
            if pid not in chosen:
                role = parent[0]
                if record["exe"] == solver:
                    role = "solver"
                elif record["exe"] == dl:
                    role = "dl"
                elif Path(record["exe"]).name in ("redis-server", "redis-stack-server", "valkey-server"):
                    role = "db"
                chosen[pid] = (role, "descendant")
                changed = True
            elif chosen[pid][0] == "support" and parent[0] != "support":
                chosen[pid] = (parent[0], chosen[pid][1])
                changed = True
    for pid, (role, _) in chosen.items():
        tracked[(pid, records[pid]["start"])] = role
    return chosen


def memory(directory, start):
    values = {}
    error = ""
    try:
        text = (directory / "smaps_rollup").read_text()
        for line in text.splitlines():
            if line.startswith(("Rss:", "Pss:")):
                values[line.split(":")[0].lower()] = int(line.split()[1]) * 1024
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return {}, "exited"
        error = f"live_read_failure:{type(exc).__name__}:errno={exc.errno}"
        try:
            for line in (directory / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    values["rss"] = int(line.split()[1]) * 1024
        except OSError:
            pass
    try:
        _, current_start, state = identity(directory)
        if current_start != start:
            return {}, "pid_reused"
        if state in ("Z", "X", "x"):
            return {}, "zombie"
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ESRCH):
            return {}, "exited"
        return {}, f"live_read_failure:{type(exc).__name__}:errno={exc.errno}"
    except (ValueError, IndexError):
        return {}, "live_read_failure:invalid_stat"
    if not error and any(key not in values for key in ("rss", "pss")):
        error = "live_read_failure:missing_rss_or_pss"
    return values, error


def cgroup_mount():
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        if after.split()[0] == "cgroup2" and before.split()[3] == "/":
            return Path(before.split()[4])
    return None


def scoped_group(paths, selected_pids, mount, existing):
    if not mount or existing is None or not paths or any(not p for p in paths):
        return None
    common = mount / os.path.commonpath(paths).lstrip("/")
    # A reused batch/job group cannot provide a configuration lifetime peak.
    if common == mount or str(common) in existing or not common.is_dir():
        return None
    try:
        members = set()
        for directory, _, _ in os.walk(common, onerror=raise_walk_error):
            members.update(int(pid) for pid in (Path(directory) / "cgroup.procs").read_text().split())
        if not selected_pids.issubset(members) or not members.issubset(selected_pids):
            return None
        return common
    except OSError:
        return None


def raise_walk_error(error):
    raise error


def counter(path):
    try:
        return int(path.read_text())
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--solver", required=True)
    parser.add_argument("--dl", required=True)
    parser.add_argument("--db-port", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.5)
    for role in ("solver", "dl", "db"):
        parser.add_argument("--expected-" + role, type=int, default=0)
    parser.add_argument("--cleanup", action="store_true", help="Terminate only tracked surviving PID/start pairs on stop")
    args = parser.parse_args()
    if not 0.2 <= args.interval <= 1.0:
        parser.error("interval must be between 0.2 and 1 seconds")
    if not args.output.is_dir():
        parser.error("output directory must already exist")
    solver, dl = os.path.realpath(args.solver), os.path.realpath(args.dl)
    initial = snapshot()
    baseline = {(pid, rec["start"]) for pid, rec in initial.items()}
    mount = cgroup_mount()
    try:
        existing = {directory for directory, _, _ in os.walk(mount, onerror=raise_walk_error)} if mount else None
    except OSError:
        existing = None  # An incomplete baseline cannot prove a cgroup is fresh.
    tracked = {}
    peaks = {role: {"rss_bytes": None, "pss_bytes": None, "max_processes": 0} for role in (*ROLES, "total")}
    seen = {role: set() for role in ROLES}
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    samples = 0
    errors = 0
    lifecycle_races = 0
    live_errors = 0
    error_counts = {}
    scope_path = None
    scope_invalid = False
    scope_peak = None
    max_sweep = 0.0
    max_direct = {role: 0 for role in ("solver", "dl", "db")}
    simultaneous_coverage_samples = 0
    actual_intervals = []
    previous_begin = None
    overrun_samples = 0
    max_cycle = 0.0
    with (args.output / "memory_samples.csv").open("x", newline="") as sf, \
            (args.output / "memory_processes.csv").open("x", newline="") as pf, \
            (args.output / "memory_cgroups.csv").open("x", newline="") as cf:
        sw, pw, cw = csv.writer(sf), csv.writer(pf), csv.writer(cf)
        sw.writerow(["sample", "unix_s", "elapsed_s", "sweep_s", "role", "processes", "rss_read", "pss_read", "rss_bytes", "pss_bytes", "direct_solver", "direct_dl", "direct_db", "sample_interval_s"])
        pw.writerow(["sample", "pid", "start_ticks", "ppid", "role", "attribution", "exe", "cgroup", "rss_bytes", "pss_bytes", "error", "error_category", "cmd"])
        cw.writerow(["sample", "cgroup", "accounting_scope", "current_bytes", "lifetime_peak_bytes"])
        (args.output / "memory.ready").touch()
        while True:
            begin = time.monotonic()
            actual_interval = None if previous_begin is None else begin - previous_begin
            if actual_interval is not None:
                actual_intervals.append(actual_interval)
            previous_begin = begin
            unix = time.time()
            records = snapshot()
            chosen = select(records, tracked, baseline, args.token, args.job_id, solver, dl, args.db_port, os.getpid())
            direct = dict(solver=sum(records[pid]["exe"] == solver for pid in chosen),
                          dl=sum(records[pid]["exe"] == dl for pid in chosen),
                          db=sum(Path(records[pid]["exe"]).name in ("redis-server", "redis-stack-server", "valkey-server") for pid in chosen))
            for role, count in direct.items():
                max_direct[role] = max(max_direct[role], count)
            simultaneous_coverage_samples += all(direct[role] >= getattr(args, "expected_" + role) for role in direct)
            totals = {role: [] for role in (*ROLES, "total")}
            for pid, (role, attribution) in chosen.items():
                rec = records[pid]
                values, error = memory(Path("/proc") / str(pid), rec["start"])
                errors += bool(error)
                category = ""
                if error:
                    category = "lifecycle" if error in LIFECYCLE_ERRORS else "live_read_failure"
                    lifecycle_races += category == "lifecycle"
                    live_errors += category == "live_read_failure"
                    error_counts[error] = error_counts.get(error, 0) + 1
                totals[role].append(values)
                totals["total"].append(values)
                seen[role].add((pid, rec["start"]))
                pw.writerow([samples, pid, rec["start"], rec["ppid"], role, attribution, rec["exe"], rec["cgroup"], values.get("rss"), values.get("pss"), error, category, rec["cmd"]])
            paths = {records[pid]["cgroup"] for pid in chosen}
            scope = scoped_group(list(paths), set(chosen), mount, existing) if chosen else None
            if chosen:
                if scope is None or (scope_path is not None and str(scope) != scope_path):
                    scope_invalid = True
                elif scope_path is None:
                    scope_path = str(scope)
            for path in sorted(paths - {""}):
                group = mount / path.lstrip("/") if mount else None
                if group:
                    cw.writerow([samples, path, "diagnostic_only_not_config", counter(group / "memory.current"), counter(group / "memory.peak")])
            if scope and not scope_invalid:
                peak = counter(scope / "memory.peak")
                if peak is not None:
                    scope_peak = max(scope_peak or 0, peak)
                cw.writerow([samples, str(scope), "fresh_exclusive_config", counter(scope / "memory.current"), peak])
            sweep = time.monotonic() - begin
            max_sweep = max(max_sweep, sweep)
            for role, readings in totals.items():
                counts = [sum(key in value for value in readings) for key in ("rss", "pss")]
                sums = [sum(value[key] for value in readings) if count == len(readings) else None for key, count in zip(("rss", "pss"), counts)]
                sw.writerow([samples, unix, begin - started, sweep, role, len(readings), *counts, *sums, direct["solver"], direct["dl"], direct["db"], actual_interval])
                peaks[role]["max_processes"] = max(peaks[role]["max_processes"], len(readings))
                for key, value in zip(("rss_bytes", "pss_bytes"), sums):
                    if value is not None and readings:
                        peaks[role][key] = max(peaks[role][key] or 0, value)
            sf.flush(); pf.flush(); cf.flush()
            cycle = time.monotonic() - begin
            max_cycle = max(max_cycle, cycle)
            overrun_samples += cycle > args.interval
            samples += 1
            if stopped or (args.output / "memory.stop").exists():
                break
            time.sleep(max(0, args.interval - cycle))
    cleanup_survivors = []
    if args.cleanup:
        # No process-name-wide kills. Recheck identity immediately before signalling.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for (pid, start) in tracked:
                try:
                    _, current_start, state = identity(Path("/proc") / str(pid))
                    if current_start == start and state not in ("Z", "X", "x"):
                        os.kill(pid, sig)
                        if sig == signal.SIGKILL:
                            cleanup_survivors.append(pid)
                except (OSError, ValueError, IndexError):
                    pass
            if sig == signal.SIGTERM:
                time.sleep(5)
    ordered_intervals = sorted(actual_intervals)
    interval_stats = dict(count=len(actual_intervals),
                          min=min(actual_intervals) if actual_intervals else None,
                          mean=statistics.mean(actual_intervals) if actual_intervals else None,
                          median=statistics.median(actual_intervals) if actual_intervals else None,
                          p95=ordered_intervals[math.ceil(0.95 * len(ordered_intervals)) - 1] if ordered_intervals else None,
                          max=max(actual_intervals) if actual_intervals else None)
    summary = dict(host=socket.gethostname(), job_id=args.job_id, token=args.token,
                   interval_s=args.interval, samples=samples, max_sweep_s=max_sweep,
                   process_read_errors=errors, sampled_simultaneous_peaks=peaks,
                   lifecycle_read_races=lifecycle_races, live_read_errors=live_errors,
                   read_error_counts=error_counts, actual_intervals_s=interval_stats,
                   overrun_samples=overrun_samples, max_cycle_s=max_cycle,
                   unique_processes={role: len(ids) for role, ids in seen.items()},
                   max_direct_processes=max_direct,
                   expected_processes={role: getattr(args, "expected_" + role) for role in max_direct},
                   simultaneous_coverage_samples=simultaneous_coverage_samples,
                   coverage_ok=simultaneous_coverage_samples > 0,
                   whole_config_cgroup=scope_path if not scope_invalid else None,
                   whole_config_cgroup_peak_bytes=scope_peak if not scope_invalid else None,
                   cgroup_status="fresh_exclusive" if scope_path and not scope_invalid else "unavailable: no fresh exclusive common config cgroup",
                   cleanup_sigkill_pids=cleanup_survivors,
                   caveats="Same sweep, not atomic. Sampled peaks are lower bounds, not summed per-PID HWMs. RSS double-counts shared pages; PSS apportions pages across all sharers, including untracked processes. Missing reads make the affected total null. Cgroup diagnostic lifetime peaks include earlier configurations; cgroups charge cache/kernel memory differently from RSS/PSS. Single node only; support includes controller/launchers, excludes this sampler and batch shell.")
    (args.output / "memory_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
