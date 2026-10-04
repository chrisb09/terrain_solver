#!/usr/bin/env python3
"""Local fixtures only: no MPI, Slurm submissions, model execution, or builds."""
import csv
import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import terrain_cpu_diagnostics as diagnostics


def log_text(values=(10, 20, 30, 40)):
    lines = ["STEP_TIMING step=2 solver=ML step_ms=999 rank_min_ms=999 rank_max_ms=999",
             "STEP_TIMING step=3 solver=Regular step_ms=1"]
    lines.extend(f"STEP_TIMING step={step} solver=ML step_ms={value} rank_min_ms={value / 2} rank_max_ms={value}"
                 for step, value in zip((4, 6, 8, 10), values))
    return "\n".join(lines) + "\n"


class DiagnosticsTest(unittest.TestCase):
    def test_abba_pairs_and_monitor_scope(self):
        for suite, cycles in (("aix", 2), ("phydll", 1)):
            rows = diagnostics.plan(suite)
            self.assertEqual(len(rows), cycles * 4)
            self.assertEqual([r[3] for r in rows[:4]], ["off", "on", "on", "off"])
            for rep in range(1, cycles * 2 + 1):
                self.assertEqual({r[3] for r in rows if r[1] == rep}, {"off", "on"})
            for row in rows:
                self.assertEqual(row[6], int(suite == "aix" and row[3] == "on"))
                self.assertEqual(row[7], int(suite == "phydll" and row[3] == "on"))

    def test_smartsim_reverse_and_batch_targets(self):
        rows = diagnostics.plan("smartsim")
        self.assertEqual(len(rows), 12)
        self.assertEqual([r[2:] for r in rows[:6]], [r[2:] for r in rows[6:]][::-1])
        self.assertTrue(all(r[6:] == (0, 0) for r in rows))
        self.assertEqual([8294400 // tpq for tpq in (1, 12, 96)], [8294400, 691200, 86400])
        self.assertEqual([46080 // tpq for tpq in (1, 12, 96)], [46080, 3840, 480])
        with self.assertRaises(ValueError):
            diagnostics.plan("smartsim", repeats=1)

    def test_steady_validation(self):
        with tempfile.TemporaryDirectory(prefix="terrain_cpu_diagnostics_") as tmp:
            path = Path(tmp) / "fixture.log"
            path.write_text(log_text())
            self.assertEqual([r["step"] for r in diagnostics.timings(path, 10)], [4, 6, 8, 10])
            for text in (log_text() + log_text(), log_text((10, 20, 30)),
                         log_text().replace("step_ms=10 ", "step_ms=nan "),
                         log_text().replace("rank_min_ms=5.0", "rank_min_ms=500")):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    diagnostics.timings(path, 10)

    def fixture(self, root):
        rows = diagnostics.plan("phydll")
        (root / "plan.tsv").write_text("".join("\t".join(map(str, row)) + "\n" for row in rows))
        manifest = []
        for run, rep, cfg, condition, tpq, intra, sampler, barriers in rows:
            path = root / f"{run}.log"
            # A different repeat baseline, but the same known treatment delta.
            base = rep * 100 + (10 if condition == "on" else 0)
            path.write_text(log_text(tuple(base + v for v in (10, 20, 30, 40))))
            manifest.append(dict(model="giant", suite="phydll", run=run, rep=rep, config=cfg,
                                 condition=condition, tpq=tpq, intra=intra, sampler=sampler,
                                 barriers=barriers, steps=10, wall_s=20, status="SUCCESS", log=str(path),
                                 cubex=str(root / f"{run}.cubex")))
        diagnostics.write_csv(root / "results.csv", manifest)
        return manifest

    def test_rep_aggregation_and_complete_plan(self):
        with tempfile.TemporaryDirectory(prefix="terrain_cpu_diagnostics_") as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            diagnostics.aggregate(root)
            with (root / "analysis/per_run.csv").open() as stream:
                per_run = list(csv.DictReader(stream))
            self.assertEqual(len(per_run), 4)
            self.assertEqual(float(per_run[0]["mean_step_ms"]), 125)
            self.assertEqual(float(per_run[0]["median_step_ms"]), 125)
            with (root / "analysis/paired.csv").open() as stream:
                pairs = list(csv.DictReader(stream))
            self.assertEqual(len(pairs), 2)
            self.assertTrue(all(float(r["on_minus_off_mean_ms"]) == 10 for r in pairs))
            with (root / "analysis/conditions.csv").open() as stream:
                conditions = list(csv.DictReader(stream))
            self.assertTrue(all(r["replicates"] == "2" for r in conditions))
            diagnostics.write_csv(root / "results.csv", manifest[:-1])
            with self.assertRaisesRegex(ValueError, "ordered plan"):
                diagnostics.aggregate(root)
            manifest[0]["status"] = "FAILED_rc_1"
            diagnostics.write_csv(root / "results.csv", manifest)
            with self.assertRaisesRegex(ValueError, "failed suite"):
                diagnostics.aggregate(root)

    def test_reused_scorep_rank_scope_and_warmup_rejection(self):
        from extract_scorep_tables import extract

        def node(name, parent, incl, excl, visits=4):
            return SimpleNamespace(name=name, parent_id=parent, incl_time=incl, excl_time=excl, visits=visits)

        solver = {0: node("terrain_solver", None, 8, 0),
                  1: node("solver_step_ml_steady", 0, 8, 0),
                  2: node("app_provider_inference", 1, 8, 6),
                  3: node("MPI_Barrier", 2, 2, 2)}
        dl = {0: node("phydll_dl_client", None, 4, 0),
              1: node("dl_torch_forward", 0, 4, 4)}
        trees = [solver] * 96 + [dl] * 96
        raw, _ = extract(trees, {"cubex": "fixture", "config": "PHYDLL_CPP"}, steady_steps=4)
        selected = [r for r in raw if r["selected_steady"] and r["region"] in ("MPI_Barrier", "dl_torch_forward")]
        self.assertEqual({r["scope"] for r in selected}, {"solver_steady", "dl_steady_count_validated"})
        self.assertEqual({r["rank_count"] for r in selected}, {96})
        self.assertEqual({r["denominator"] for r in selected}, {384})
        self.assertEqual(next(r["inclusive_ms_per_step"] for r in selected if r["region"] == "dl_torch_forward"), 1000)
        with tempfile.TemporaryDirectory(prefix="terrain_cpu_diagnostics_") as tmp:
            root = Path(tmp)
            manifest = self.fixture(root)
            with patch("extract_scorep_tables.parse_cubex_file", return_value=trees) as parse:
                diagnostics.aggregate(root, scorep=True)
            self.assertEqual([call.args[0] for call in parse.call_args_list], [Path(r["cubex"]) for r in manifest])
            with (root / "analysis/scorep_forward_barrier.csv").open() as stream:
                outputs = list(csv.DictReader(stream))
            self.assertEqual(len(outputs), 8)
            for row in outputs:
                source = next(r for r in manifest if r["run"] == row["run"])
                self.assertEqual(row["cubex"], source["cubex"])
                self.assertEqual(row["condition"], source["condition"])
                self.assertEqual(row["rank_count"], "96")
                self.assertEqual(row["denominator"], "384")
                self.assertEqual(float(row["inclusive_ms_per_step"]),
                                 1000 if row["region"] == "dl_torch_forward" else 500)
        dl[1].visits = 5
        with self.assertRaisesRegex(ValueError, "cannot remove warmup"):
            extract(trees, {"cubex": "fixture", "config": "PHYDLL_CPP"}, steady_steps=4)

    def test_runner_contract(self):
        text = Path(__file__).with_name("terrain_cpu_diagnostics.sbatch").read_text()
        self.assertIn('command+=(: "${exports[@]}" "${extra_exports[@]}" -n 96', text)
        self.assertIn("-x PHYDLL_IO_LOG_BARRIERS -x PHYDLL_TELEMETRY", text)
        self.assertIn("export PHYDLL_IO_LOG=1", text)
        self.assertIn("-x PHYDLL_IO_LOG -x PHYDLL_IO_LOG_BARRIERS", text)
        self.assertIn("mpirun --bind-to none --report-bindings --oversubscribe", text)
        self.assertNotIn("python3", text)
        self.assertLess(text.index("HELPER_PYTHON="), text.index('"$HELPER_PYTHON" "$HELPER" --plan'))
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_DYNAMIC"):
            self.assertNotIn(variable, text)
        self.assertIn("OMP_*|MKL_*|OPENBLAS_*", text)
        self.assertIn("SMARTSIM_FLAG_TELEMETRY=0 PHYDLL_TELEMETRY=0", text)
        self.assertIn("--ml-db-min-batch-size 0 --ml-db-min-batch-timeout 0", text)
        self.assertNotIn("pkill", text)
        self.assertNotIn("cmake", text)

    def test_loaded_phydll_hashes(self):
        with tempfile.TemporaryDirectory(prefix="terrain_cpu_diagnostics_") as tmp:
            root = Path(tmp)
            library = root / "libphydll.so.1"
            library.write_bytes(b"fixture library build")
            alias = root / "libphydll.so"
            alias.symlink_to(library)
            solver, dl = root / "solver_libraries.txt", root / "dl_libraries.txt"
            solver.write_text(f"\tlibphydll.so => {alias} (0x123)\n")
            dl.write_text(f"\tlibphydll.so.1 => {library} (0x456)\n")
            output = io.StringIO()
            with redirect_stdout(output):
                diagnostics.hash_phydll_libraries([solver, dl])
            self.assertEqual(output.getvalue(), f"{hashlib.sha256(library.read_bytes()).hexdigest()}  {library.resolve()}\n")
            second = root / "other/libphydll.so"
            second.parent.mkdir()
            second.write_bytes(b"different DL library build")
            dl.write_text(f"\tlibphydll.so => {second} (0x456)\n")
            output = io.StringIO()
            with redirect_stdout(output):
                diagnostics.hash_phydll_libraries([solver, dl])
            expected = {f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.resolve()}" for p in (library, second)}
            self.assertEqual(set(output.getvalue().splitlines()), expected)
            dl.write_text("\tlibphydll.so => not found\n")
            with self.assertRaisesRegex(ValueError, "resolved readable"):
                diagnostics.hash_phydll_libraries([solver, dl])

    def test_startup_failure_manifest_paths(self):
        text = Path(__file__).with_name("terrain_cpu_diagnostics.sbatch").read_text()
        record = text[text.index("record_run() {"):text.index("trap 'exit_rc=")]
        finish = text[text.index("    t0=$SECONDS\n    if (( rc == 0 ));"):text.index('done < "$ROOT/plan.tsv"')]
        with tempfile.TemporaryDirectory(prefix="terrain_cpu_diagnostics_") as tmp:
            root = Path(tmp)
            fields = "model,suite,run,rep,config,condition,tpq,intra,db_batch,sampler,barriers,steps,wall_s,status,log,cubex"
            (root / "results.csv").write_text(fields + "\n")
            script = record + f'\nROOT="{root}"; dir="$ROOT/run01"; mkdir "$dir";\n' + '''
MODEL=giant; SUITE=smartsim; run=run01; rep=1; config=TPQ1_I96; condition=off
tpq=1; intra=96; db_batch=0; sampler=0; barriers=0; STEPS=10; rc=1
active_run=1; finished_wall=
cleanup() { return 0; }
command=(false)
''' + finish
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            with (root / "results.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "FAILED_rc_1")
            self.assertEqual(rows[0]["cubex"], "")
            self.assertEqual(Path(rows[0]["log"]), root / "run01/solver.log")
            self.assertIn("Solver not launched", Path(rows[0]["log"]).read_text())
            with self.assertRaisesRegex(ValueError, "failed suite"):
                diagnostics.aggregate(root, scorep=True)

    def test_shell_child_stdin_preserves_complete_ordered_plan(self):
        text = Path(__file__).with_name("terrain_cpu_diagnostics.sbatch").read_text()
        record = text[text.index("record_run() {"):text.index("trap 'exit_rc=")]
        header = next(line for line in text.splitlines() if line.startswith("while IFS=$'\\t' read"))
        finish = text[text.index("    t0=$SECONDS\n    if (( rc == 0 ));"):text.index('\n"$HELPER_PYTHON" "$HELPER" --root')]
        helper = Path(diagnostics.__file__).resolve()
        with tempfile.TemporaryDirectory(prefix="terrain_cpu_diagnostics_") as tmp:
            for suite in ("aix", "phydll", "smartsim"):
                for isolated in (True, False):
                    with self.subTest(suite=suite, isolated=isolated):
                        root = Path(tmp) / f"{suite}_{isolated}"
                        root.mkdir()
                        plan = diagnostics.plan(suite)
                        (root / "plan.tsv").write_text("".join("\t".join(map(str, row)) + "\n" for row in plan))
                        fields = "model,suite,run,rep,config,condition,tpq,intra,db_batch,sampler,barriers,steps,wall_s,status,log,cubex"
                        (root / "results.csv").write_text(fields + "\n")
                        prefix = f'''set -eo pipefail
ROOT="{root}"; MODEL=giant; SUITE={suite}; STEPS=10
HELPER_PYTHON="{sys.executable}"; HELPER="{helper}"
cleanup() {{ return 0; }}
mock_child() {{
    printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' "$run" "$rep" "$config" "$condition" "$tpq" "$intra" "$sampler" "$barriers" >> "$ROOT/executed.tsv"
    bash -c 'while IFS= read -r line; do printf "%s\\n" "$line"; done' > "$dir/consumed_stdin.tsv"
    mkdir "$dir/scorep"
    : > "$dir/scorep/profile.cubex"
    printf '%s\\n' '{{"coverage_ok":true,"live_read_errors":0,"cleanup_sigkill_pids":[]}}' > "$dir/memory_summary.json"
    printf '%s' '{log_text()}'
}}
'''
                        body = '''
    dir=$ROOT/$run
    mkdir "$dir"
    rc=0; db_batch=0; finished_wall=; active_run=1
    command=(mock_child)
'''
                        # Negative control recreates the reported bug using the same loop.
                        invocation = finish if isolated else finish.replace('"${command[@]}" < /dev/null', '"${command[@]}"')
                        result = subprocess.run(["bash", "-c", record + prefix + header + "\n" + body + invocation],
                                                capture_output=True, text=True, timeout=30)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        executed = (root / "executed.tsv").read_text().splitlines()
                        expected = ["\t".join(map(str, row)) for row in plan]
                        self.assertEqual(executed, expected if isolated else expected[:1])
                        with (root / "results.csv").open() as stream:
                            rows = list(csv.DictReader(stream))
                        self.assertEqual([r["run"] for r in rows], [r[0] for r in plan] if isolated else ["run01"])
                        self.assertTrue(all(r["status"] == "SUCCESS" for r in rows))
                        if isolated:
                            self.assertTrue(all(not p.read_text() for p in root.glob("run*/consumed_stdin.tsv")))
                            diagnostics.aggregate(root)
                        else:
                            self.assertEqual((root / "run01/consumed_stdin.tsv").read_text().splitlines(), expected[1:])
                            with self.assertRaisesRegex(ValueError, "ordered plan"):
                                diagnostics.aggregate(root)

    def test_shell_cleanup_scope_and_sampler_wait(self):
        text = Path(__file__).with_name("terrain_cpu_diagnostics.sbatch").read_text()
        functions = text[text.index("monitor_pid= ctrl_pid="):text.index("record_run()")]
        with tempfile.TemporaryDirectory(prefix="terrain_cpu_diagnostics_") as tmp:
            prefix = f'SLURM_JOB_ID=123; dir="{tmp}"; done_file="$dir/done"; '
            mocks = '''
kill() { return 1; }
wait() { return 0; }
sleep() { :; }
scancel() { printf 'cancelled:%s\n' "$1"; }
'''
            script = functions + "\n" + prefix + mocks + '''
squeue() { printf '123.batch\n123.extern\n123.4\n999.2\n'; }
baseline_steps[123.4]=1; controller_active=1; ctrl_pid=777
cleanup
'''
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("cancelled:", result.stdout)
            script = functions + "\n" + prefix + mocks + '''
squeue() { printf '123.batch\n123.4\n123.5\n999.2\n'; }
baseline_steps[123.4]=1; controller_active=1; ctrl_pid=777
cleanup
'''
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(result.stdout.strip(), "cancelled:123.5")
            script = functions + "\n" + prefix + mocks + '''
squeue() { return 1; }
controller_active=1; ctrl_pid=777
cleanup
'''
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            script = functions + "\n" + prefix + mocks + '''
wait() { printf 'waited:%s\n' "$1"; return 1; }
monitor_pid=888
cleanup
'''
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(result.stdout.strip(), "waited:888")
            self.assertTrue((Path(tmp) / "memory.stop").is_file())
            script = functions + "\n" + prefix + '''
ctrl_pid=$BASHPID
read -r stat < /proc/self/stat
read -ra fields <<< "${stat##*) }"
printf '%s\n' "${fields[19]}" > "$dir/controller_start"
controller_alive || exit 2
printf '0\n' > "$dir/controller_start"
if controller_alive; then exit 3; fi
'''
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
