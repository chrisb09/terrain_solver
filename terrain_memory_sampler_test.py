#!/usr/bin/env python3
"""Fixtures plus a real /proc test; no Slurm allocation or MPI required."""
import csv
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import terrain_memory_sampler as sampler


def record(exe="/bin/launcher", start=100, ppid=1, token=None, job="42", cmd="", group="/slurm/job_42/step_0"):
    env = {b"SLURM_JOB_ID": job.encode()}
    if token:
        env[b"TERRAIN_MEMORY_TOKEN"] = token.encode()
    return dict(exe=exe, start=start, ppid=ppid, env=env, cmd=cmd, cgroup=group, groups="0::" + group)


def stat_text(start=123, state="S"):
    fields = [state, "1"] + ["0"] * 17 + [str(start)]
    return "10 (name with ) spaces) " + " ".join(fields)


class MemoryTests(unittest.TestCase):
    def test_selection_roles_reparenting_and_pid_reuse(self):
        records = {10: record(token="cfg"), 11: record("/solver", ppid=10),
                   12: record("/dl", token="cfg"), 13: record(ppid=12),
                   20: record("/bin/redis-server", cmd="redis-server *:7777"),
                   21: record("/bin/redis-server", job="99", group="/slurm/job_99", cmd="redis-server *:7777"),
                   22: record("/solver", token="other"),
                   23: record("/bin/redis-server", cmd="redis-server *:7777")}
        tracked = {}
        selected = sampler.select(records, tracked, {(23, 100)}, "cfg", "42", "/solver", "/dl", 7777, 999)
        self.assertEqual({pid: role for pid, (role, _) in selected.items()}, {10: "support", 11: "solver", 12: "dl", 13: "dl", 20: "db"})
        records = {11: record("/solver", ppid=1), 13: record(start=101)}
        selected = sampler.select(records, tracked, set(), "cfg", "42", "/solver", "/dl", 7777, 999)
        self.assertEqual(set(selected), {11})

    def test_job_cgroup_without_environment_and_port_boundary(self):
        rec = record("/bin/redis-server", cmd="redis-server *:77770")
        rec["env"] = {}
        self.assertTrue(sampler.in_job(rec, "42"))
        self.assertFalse(sampler.in_job(rec, "4"))
        self.assertFalse(sampler.select({10: rec}, {}, set(), "cfg", "42", "/solver", "/dl", 7777, 999))

    def test_cgroup_scope_rejects_batch_and_unrelated_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = root / "batch"
            batch.mkdir()
            (batch / "cgroup.procs").write_text("10\n")
            self.assertIsNone(sampler.scoped_group(["/batch"], {10}, root, {str(batch)}))
            fresh = root / "fresh"
            fresh.mkdir()
            (fresh / "cgroup.procs").write_text("10\n11\n")
            self.assertIsNone(sampler.scoped_group(["/fresh"], {10}, root, {str(batch)}))
            self.assertEqual(sampler.scoped_group(["/fresh"], {10, 11}, root, {str(batch)}), fresh)
            self.assertIsNone(sampler.scoped_group(["/fresh"], {10, 11}, root, None))
            self.assertIsNone(sampler.scoped_group(["/fresh", "/batch"], {10, 11}, root, set()))

    def test_memory_fixture_missing_pss_and_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stat").write_text(stat_text())
            (root / "status").write_text("VmRSS:\t2048 kB\n")
            values, error = sampler.memory(root, 123)
            self.assertEqual(values, {"rss": 2048 * 1024})
            self.assertEqual(error, "live_read_failure:FileNotFoundError:errno=2")
            (root / "smaps_rollup").write_text("Rss: 4096 kB\nPss: 3072 kB\n")
            self.assertEqual(sampler.memory(root, 123), ({"rss": 4096 * 1024, "pss": 3072 * 1024}, ""))
            self.assertEqual(sampler.memory(root, 124), ({}, "pid_reused"))
            (root / "stat").write_text(stat_text(state="Z"))
            self.assertEqual(sampler.memory(root, 123), ({}, "zombie"))
            (root / "stat").unlink()
            self.assertEqual(sampler.memory(root, 123), ({}, "exited"))

    def test_live_read_permission_failure_is_not_an_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stat").write_text(stat_text())
            (root / "status").write_text("VmRSS:\t2048 kB\n")
            original = Path.read_text

            def denied(path, *args, **kwargs):
                if path.name == "smaps_rollup":
                    raise PermissionError(errno.EACCES, "denied")
                return original(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", denied):
                self.assertEqual(sampler.memory(root, 123),
                                 ({"rss": 2048 * 1024}, "live_read_failure:PermissionError:errno=13"))
            def dying(path, *args, **kwargs):
                if path.name == "smaps_rollup":
                    raise ProcessLookupError(errno.ESRCH, "process exiting")
                return original(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", dying):
                self.assertEqual(sampler.memory(root, 123), ({}, "exited"))
            (root / "smaps_rollup").write_text("Rss: 4096 kB\n")
            self.assertEqual(sampler.memory(root, 123),
                             ({"rss": 4096 * 1024}, "live_read_failure:missing_rss_or_pss"))

            def stat_denied(path, *args, **kwargs):
                if path.name == "stat":
                    raise PermissionError(errno.EACCES, "denied")
                return original(path, *args, **kwargs)

            with mock.patch.object(Path, "read_text", stat_denied):
                self.assertEqual(sampler.memory(root, 123),
                                 ({}, "live_read_failure:PermissionError:errno=13"))

    def test_snapshot_skips_zombies_and_dead_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for pid, state in ((10, "S"), (11, "Z"), (12, "X"), (13, "x")):
                directory = root / str(pid)
                directory.mkdir()
                (directory / "stat").write_text(stat_text(state=state))
                (directory / "environ").write_bytes(b"TERRAIN_MEMORY_TOKEN=cfg\0")
                (directory / "cgroup").write_text("0::/test\n")
                (directory / "cmdline").write_bytes(b"solver\0")
                (directory / "exe").symlink_to("/solver")
            self.assertEqual(set(sampler.snapshot(root)), {10})

    def test_sample_schema_errors_peaks_cadence_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = {10: record("/solver", token="cfg"), 11: record("/solver", token="cfg"),
                       12: record(token="cfg"), 13: record(token="cfg")}
            calls = 0

            def snapshots():
                nonlocal calls
                calls += 1
                if calls == 1:
                    return {}
                if calls == 4:
                    (root / "memory.stop").touch()
                return records

            readings = [({}, "zombie"), ({}, "live_read_failure:PermissionError:errno=13"),
                        ({}, "exited"), ({}, "pid_reused")]
            readings += [({"rss": 100, "pss": 50}, "")] * 4
            readings += [({}, "exited")] + [({"rss": 10000, "pss": 5000}, "")] * 3

            def identities(directory):
                pid = int(directory.name)
                return 1, 101 if pid == 11 else 100, {10: "Z", 11: "S", 12: "S", 13: "X"}[pid]

            argv = ["sampler", "--output", tmp, "--token", "cfg", "--solver", "/solver",
                    "--dl", "/dl", "--interval", "0.2", "--expected-solver", "2", "--cleanup"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(sampler, "snapshot", side_effect=snapshots), \
                    mock.patch.object(sampler, "memory", side_effect=readings), \
                    mock.patch.object(sampler, "cgroup_mount", return_value=None), \
                    mock.patch.object(sampler, "identity", side_effect=identities), \
                    mock.patch.object(sampler.signal, "signal"), \
                    mock.patch.object(sampler.os, "kill") as kill, \
                    mock.patch.object(sampler.time, "sleep"), \
                    mock.patch.object(sampler.time, "monotonic", side_effect=[0, 1, 1.1, 1.4, 1.6, 1.65, 1.7, 1.8, 1.85, 1.9]):
                sampler.main()
            self.assertEqual(kill.call_args_list, [mock.call(12, sampler.signal.SIGTERM), mock.call(12, sampler.signal.SIGKILL)])
            summary = json.loads((root / "memory_summary.json").read_text())
            self.assertEqual(summary["cleanup_sigkill_pids"], [12])
            self.assertEqual(summary["process_read_errors"], 5)
            self.assertEqual(summary["lifecycle_read_races"], 4)
            self.assertEqual(summary["live_read_errors"], 1)
            self.assertEqual(summary["read_error_counts"]["exited"], 2)
            self.assertEqual(summary["sampled_simultaneous_peaks"]["total"]["rss_bytes"], 400)
            self.assertEqual(summary["sampled_simultaneous_peaks"]["total"]["pss_bytes"], 200)
            self.assertEqual(summary["overrun_samples"], 1)
            self.assertAlmostEqual(summary["max_cycle_s"], 0.4)
            self.assertEqual(summary["actual_intervals_s"]["count"], 2)
            self.assertAlmostEqual(summary["actual_intervals_s"]["mean"], 0.4)
            self.assertAlmostEqual(summary["actual_intervals_s"]["p95"], 0.6)
            with (root / "memory_samples.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 15)
            for row in rows:
                self.assertEqual([row[key] for key in ("direct_solver", "direct_dl", "direct_db")], ["2", "0", "0"])
                if row["sample"] == "0":
                    self.assertEqual(row["sample_interval_s"], "")
                elif row["sample"] == "1":
                    self.assertAlmostEqual(float(row["sample_interval_s"]), 0.6)
                if row["role"] == "total" and row["sample"] != "1":
                    self.assertEqual(row["rss_bytes"], "")
                    self.assertEqual(row["pss_bytes"], "")
            with (root / "memory_processes.csv").open() as file:
                process_rows = list(csv.DictReader(file))
            self.assertEqual([row["error_category"] for row in process_rows[:4]],
                             ["lifecycle", "live_read_failure", "lifecycle", "lifecycle"])

    def test_live_proc_isolation_and_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            token = "terrain_test_%d" % os.getpid()
            monitor = subprocess.Popen([sys.executable, str(Path(sampler.__file__)), "--output", tmp,
                                        "--token", token, "--solver", os.path.realpath(sys.executable),
                                        "--dl", "/not/a/dl", "--interval", "0.2", "--expected-solver", "2", "--cleanup"])
            children = []
            try:
                deadline = time.monotonic() + 15
                while not (root / "memory.ready").exists():
                    self.assertIsNone(monitor.poll())
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.05)
                for tag in (token, token, "unrelated"):
                    children.append(subprocess.Popen([sys.executable, "-c", "import time; data=bytearray(16*1024*1024); time.sleep(20)"], env={**os.environ, "TERRAIN_MEMORY_TOKEN": tag}))
                time.sleep(1.2)
                (root / "memory.stop").touch()
                self.assertEqual(monitor.wait(timeout=10), 0)
                self.assertIsNotNone(children[0].poll())
                self.assertIsNotNone(children[1].poll())
                self.assertIsNone(children[2].poll())
                summary = json.loads((root / "memory_summary.json").read_text())
                self.assertTrue(summary["coverage_ok"])
                self.assertEqual(summary["max_direct_processes"]["solver"], 2)
                self.assertIsNone(summary["whole_config_cgroup_peak_bytes"])
                self.assertFalse(summary["cleanup_sigkill_pids"])
                self.assertEqual(summary["process_read_errors"], summary["lifecycle_read_races"] + summary["live_read_errors"])
                self.assertEqual(summary["actual_intervals_s"]["count"], summary["samples"] - 1)
                with (root / "memory_processes.csv").open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual({int(row["pid"]) for row in rows}, {p.pid for p in children[:2]})
                with (root / "memory_samples.csv").open() as file:
                    totals = {row["sample"]: row for row in csv.DictReader(file) if row["role"] == "total"}
                for sample, total in totals.items():
                    readings = [row for row in rows if row["sample"] == sample]
                    if total["rss_bytes"]:
                        self.assertEqual(int(total["rss_bytes"]), sum(int(row["rss_bytes"]) for row in readings))
                    self.assertEqual(int(total["direct_solver"]), len(readings))
            finally:
                if monitor.poll() is None:
                    monitor.terminate()
                    monitor.wait(timeout=10)
                for child in children:
                    child.terminate()
                    child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
