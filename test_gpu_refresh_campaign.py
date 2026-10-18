import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("refresh", HERE / "gpu_refresh_campaign.py")
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)


class CampaignTest(unittest.TestCase):
    def test_original_solver_exact_patch_and_fail_closed_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            mini = self.fixture(Path(tmp), reviewed_solver=True)
            source = mini / "solver_cpp/terrain_solver.cpp"
            current = source.read_bytes()
            solver = mini / "solver_cpp/build_gpu_refresh_plain"
            prov = solver / "provenance-123"
            protected = [prov / "source.sha256", prov / "binaries.sha256", solver / "BUILD_READY",
                         solver / "terrain_solver",
                         mini.parent / "CPP-ML-Interface/dl_clients/build-gpu-refresh-plain/phydll_dl_client"]
            originals = {p: p.read_bytes() for p in protected}
            with patch.object(refresh, "MINI", mini), patch.dict(os.environ, GPU_REFRESH_BUILD_JOB_ID_ENV="123"):
                for flag in ("", "0", "true"):
                    with patch.dict(os.environ, {refresh.REUSE_ORIGINAL_SOLVER_ENV: flag}):
                        with self.assertRaises(subprocess.CalledProcessError):
                            refresh.ready("plain")
                with patch.dict(os.environ, {refresh.REUSE_ORIGINAL_SOLVER_ENV: "1"}):
                    self.assertEqual(refresh.ready("plain"), originals[solver / "BUILD_READY"].decode())
                    review, original = refresh.check_sources(prov)
                    self.assertEqual(original, b"// fixture\n" + refresh.ORIGINAL_MODEL_PATH_CHECK)
                    self.assertEqual(review["original_sha256"], hashlib.sha256(original).hexdigest())
                    self.assertEqual(review["current_sha256"], hashlib.sha256(current).hexdigest())
                    for changed in (current + b"// unexpected\n", current * 2,
                                    current.replace(b'!= "none"', b'== "none"')):
                        source.write_bytes(changed)
                        with self.assertRaises(RuntimeError):
                            refresh.ready("plain")
                    source.write_bytes(current)
                    other = mini / "5r_driver_single.zsh"
                    other_original = other.read_bytes()
                    other.write_bytes(other_original + b"# unexpected\n")
                    with self.assertRaises(subprocess.CalledProcessError):
                        refresh.ready("plain")
                    other.write_bytes(other_original)
                    for path in protected:
                        path.write_bytes(originals[path] + b"corrupt\n")
                        with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                            refresh.ready("plain")
                        path.write_bytes(originals[path])
                    with self.assertRaisesRegex(RuntimeError, "plain timing"):
                        refresh.ready("scorep")
                    refresh.ready("plain")
            self.assertEqual(source.read_bytes(), current)
            self.assertTrue(all(p.read_bytes() == data for p, data in originals.items()))

    def test_original_solver_targeted_scope_and_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            mini = self.fixture(Path(tmp), reviewed_solver=True)
            groups = refresh.targeted_retest_groups()
            env = {"GPU_REFRESH_BUILD_JOB_ID_ENV": "123", refresh.REUSE_ORIGINAL_SOLVER_ENV: "1"}
            with patch.object(refresh, "MINI", mini), patch.dict(os.environ, env), \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                rejected = [refresh.groups()[0], refresh.missing65_profile_groups()[0],
                            dict(groups[0], mode="profile"), dict(groups[0], mode="trace")]
                for group in rejected:
                    for selected in ([group], [groups[0], group]):
                        with self.assertRaisesRegex(RuntimeError, "restricted"):
                            refresh.submit("--dry-run", "rejected", selected)
                    with self.assertRaisesRegex(RuntimeError, "restricted"):
                        refresh.run(group, "rejected")
                self.assertFalse((mini / "results").exists())
                refresh.submit("--dry-run", "reviewed", groups, smoke_first=True)
                root = mini / "results/gpu_refresh/reviewed"
                review = json.loads((root / "original-solver-review.json").read_text())
                original = (root / review["source_original_snapshot"]).read_bytes()
                self.assertEqual(hashlib.sha256(original).hexdigest(), review["original_sha256"])
                self.assertEqual((review["build_identity"], review["original_binary_profile"]), ("123", "plain"))
                self.assertEqual(review["approved_patch"]["replacements"], 1)
                commands = [shlex.split(line) for line in output.getvalue().splitlines() if line.startswith("sbatch ")]
                self.assertEqual(len(commands), 9)
                self.assertTrue(all(any(a.startswith("--export=ALL,") for a in cmd) for cmd in commands))

    def test_original_solver_runtime_review_before_cases(self):
        from unittest.mock import MagicMock
        group = refresh.targeted_retest_groups()[3]
        with tempfile.TemporaryDirectory() as tmp:
            mini = self.fixture(Path(tmp), reviewed_solver=True)
            shutil.copy(HERE / "config.toml", mini)
            root = mini / "results/gpu_refresh/reviewed" / f"{group['id']}_456"

            def fake_popen(*args, **kwargs):
                review = json.loads((root / "original-solver-review.json").read_text())
                self.assertTrue((root / review["source_original_snapshot"]).exists())
                self.assertEqual(kwargs["env"][refresh.REUSE_ORIGINAL_SOLVER_ENV], "1")
                for step in range(2, 13, 2):
                    kwargs["stdout"].write(f"STEP_TIMING step={step} solver=ML step_ms=1\n")
                return MagicMock(pid=99999999, wait=MagicMock(return_value=0))

            env = {"GPU_REFRESH_BUILD_JOB_ID_ENV": "123", refresh.REUSE_ORIGINAL_SOLVER_ENV: "1",
                   "SLURM_JOB_ID": "456", "SLURM_NTASKS": "96", "SLURM_JOB_NUM_NODES": "1"}
            with patch.object(refresh, "MINI", mini), patch.dict(os.environ, env), \
                 patch.object(refresh, "ready") as ready, patch.object(refresh, "clean_steps"), \
                 patch.object(refresh.subprocess, "run"), \
                 patch.object(refresh.subprocess, "Popen", side_effect=fake_popen) as popen, \
                 patch.object(refresh.os, "killpg", side_effect=ProcessLookupError):
                self.assertEqual(refresh.run(group, "reviewed"), 0)
                self.assertEqual([c.args for c in ready.call_args_list], [("plain",)] * 2)
                popen.assert_called_once()

    def test_targeted_retest_matrix_and_environment(self):
        groups = refresh.targeted_retest_groups()
        expected = {
            ("96p96c4g", "watercnn8k", "PHYDLL", "default"): (25, 8.5, 101),
            ("96c4g", "giantmlp288", "SMARTSIM", "shared"): (8, 2, 6),
            ("96c4g", "watercnn8k", "SMARTSIM", "shared_c1"): (25, 10.7, 101),
        }
        self.assertEqual(len(groups), 9)
        self.assertEqual(len({g["id"] for g in groups}), 9)
        self.assertFalse({g["id"] for g in groups} & {g["id"] for g in refresh.groups()})
        for identity, (limit, bound, calls) in expected.items():
            selected = [g for g in groups if (g["hw"], *g["cases"][0]) == identity]
            self.assertEqual([g["repeat"] for g in selected], [1, 2, 3])
            for group in selected:
                self.assertEqual(len(group["cases"]), 1)
                self.assertEqual(group["mode"], "timing")
                self.assertEqual(refresh.minutes(group), limit)
                budget = group["budget"]
                self.assertEqual(budget["purpose"], "diagnostic-retest")
                self.assertEqual(budget["telemetry"], "not-integrated")
                self.assertEqual(len(budget["original_job_ids"]), 3)
                self.assertEqual(budget["native_call_bound_seconds"], bound)
                self.assertEqual(budget["expected_calls_per_case"], calls)
                self.assertAlmostEqual(budget["inference_seconds"] + budget["startup_cleanup_margin_seconds"], limit * 60)
                env = refresh.case_env(group, group["cases"][0])
                water = identity[1] == "watercnn8k"
                self.assertEqual(env["TOTAL_STEPS_ENV"], "202" if water else "12")
                self.assertEqual(env["ML_BATCH_SIZE_ENV"], "20000000" if water else "500000")
                for key in ("USE_SCOREP", "USE_SCOREP_ENV", "NIC_SAMPLING_ENV", "GPU_MEM_SAMPLING_ENV", "OVERWRITE_OUTPUT_ENV"):
                    self.assertEqual(env[key], "0")
                self.assertEqual(env["SCOREP_ENABLE_PROFILING"], "false")
                self.assertEqual(env["SCOREP_ENABLE_TRACING"], "false")
                self.assertNotIn("CUDA_LAUNCH_BLOCKING", env)
                self.assertEqual(env["PHYDLL_PIN_CPU_BUDGET_ENV"], "96")
                self.assertEqual(env["PHYDLL_NP_DL_ENV"], "4")
                self.assertEqual(env["SMARTSIM_THREADS_PER_QUEUE"], "1")
                self.assertEqual(env["SMARTSIM_INTRA_OP_THREADS"], "8")
                self.assertEqual(env["SMARTSIM_INTER_OP_THREADS"], "1")
                self.assertEqual(env["SMARTSIM_MPI_SEQUENTIAL_PUT"], "1" if identity[3] == "shared_c1" else "0")
                with patch.object(refresh, "run", return_value=0) as run, \
                     patch.object(refresh.os.sys, "argv", ["campaign", "--run", group["id"]]), \
                     patch.dict(os.environ, GPU_REFRESH_CAMPAIGN_ENV="targeted_test"):
                    self.assertEqual(refresh.main(), 0)
                    run.assert_called_once_with(group, "targeted_test")

    def test_targeted_retest_dry_run_gate_and_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            mini = self.fixture(Path(tmp))
            groups = refresh.targeted_retest_groups()
            with patch.object(refresh, "MINI", mini), patch.dict(os.environ, GPU_REFRESH_BUILD_JOB_ID_ENV="123"), \
                 patch.object(refresh, "ready", wraps=refresh.ready) as ready, \
                 patch.object(refresh.subprocess, "check_output", wraps=subprocess.check_output) as calls, \
                 contextlib.redirect_stdout(io.StringIO()) as output:
                refresh.submit("--dry-run", "targeted", groups, notifier="snotify", smoke_first=True)
                ready.assert_called_once_with("plain")
                self.assertFalse(any(call.args[0][0] in ("sbatch", "snotify") for call in calls.call_args_list))
                commands = [shlex.split(line) for line in output.getvalue().splitlines() if line.startswith("sbatch ")]
                self.assertEqual(len(commands), 9)
                for index, (group, command) in enumerate(zip(groups, commands)):
                    self.assertEqual(command[-1], group["id"])
                    self.assertEqual(command.count(":"), int("p" in group["hw"]))
                    self.assertEqual([a for a in command if a.startswith("--time=")],
                                     [f"--time=00:{refresh.minutes(group):02d}:00"] * (2 if "p" in group["hw"] else 1))
                    self.assertEqual([a for a in command if a.startswith("--dependency=")],
                                     ["--dependency=afterok:DRY_RUN"] if index else [])
                    if index:
                        self.assertIn("--kill-on-invalid-dep=yes", command)
                root = mini / "results/gpu_refresh/targeted"
                self.assertEqual(set(json.loads((root / "ready.json").read_text())), {"plain"})
                self.assertEqual(len(json.loads((root / "budget.json").read_text())), 9)
                with (root / "cases.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 9)
                self.assertTrue(all((r["warm"], r["steady"]) == ("1", "100" if r["model"] == "watercnn8k" else "5") for r in rows))
                original = (root / "cases.csv").read_bytes()
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    refresh.submit("--dry-run", "targeted", groups)
                self.assertEqual((root / "cases.csv").read_bytes(), original)
                (mini / "solver_cpp/build_gpu_refresh_plain/BUILD_READY").unlink()
                with self.assertRaises(OSError):
                    refresh.submit("--dry-run", "not_ready", groups)
                self.assertFalse((mini / "results/gpu_refresh/not_ready").exists())

    def test_targeted_retest_all_nine_complete_without_profiles(self):
        from unittest.mock import MagicMock
        for group in refresh.targeted_retest_groups():
            with self.subTest(group=group["id"]), tempfile.TemporaryDirectory() as tmp:
                mini = self.fixture(Path(tmp))
                for name in ("config.toml", "config_phydll_gpu_refresh_default.toml"):
                    shutil.copy(HERE / name, mini)

                def fake_popen(*args, **kwargs):
                    env = kwargs["env"]
                    self.assertEqual(env["CUDA_LAUNCH_BLOCKING"], "1")
                    for step in range(2, int(env["TOTAL_STEPS_ENV"]) + 1, 2):
                        kwargs["stdout"].write(f"STEP_TIMING step={step} solver=ML step_ms=1\n")
                    return MagicMock(pid=99999999, wait=MagicMock(return_value=0))

                env = dict(SLURM_JOB_ID="456", SLURM_NTASKS="96", SLURM_JOB_NUM_NODES="1",
                           SLURM_HET_SIZE="2", SLURM_NTASKS_HET_GROUP_0="96", SLURM_NTASKS_HET_GROUP_1="96",
                           SLURM_JOB_NUM_NODES_HET_GROUP_0="1", SLURM_JOB_NUM_NODES_HET_GROUP_1="1",
                           CUDA_LAUNCH_BLOCKING="1")
                with patch.object(refresh, "MINI", mini), patch.dict(os.environ, env), \
                     patch.object(refresh, "ready") as ready, patch.object(refresh, "clean_steps"), \
                     patch.object(refresh.subprocess, "Popen", side_effect=fake_popen) as popen, \
                     patch.object(refresh.os, "killpg", side_effect=ProcessLookupError):
                    self.assertEqual(refresh.run(group, "targeted"), 0)
                popen.assert_called_once()
                self.assertEqual([call.args for call in ready.call_args_list], [("plain",)] * 2)
                root = mini / "results/gpu_refresh/targeted" / f"{group['id']}_456"
                with (root / "manifest.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual((row["rc"], row["validation_error"], row["cleanup_safe"]), ("0", "", "True"))
                self.assertEqual(int(row["observed_ml_calls"]), 101 if row["model"] == "watercnn8k" else 6)
                self.assertEqual((root / row["run_id"] / "profiles.txt").read_text(), "")

    def test_thesis_aix_trace_matrix_environment_and_authorization(self):
        groups = refresh.thesis_aix_trace_groups()
        self.assertEqual(len(groups), 4)
        self.assertEqual(len({g["id"] for g in groups}), 4)
        self.assertEqual({g["hw"] for g in groups}, set(refresh.HARDWARE))
        self.assertEqual([refresh.minutes(g) for g in groups], [6, 8, 6, 10])
        seen = set()
        for group in groups:
            self.assertEqual((group["mode"], group["repeat"]), ("trace", 1))
            budget = group["budget"]
            self.assertEqual(budget["expected_calls_per_case"], 6)
            seconds = 60 + 1.5 * (budget["driver_seconds"] + budget["cleanup_seconds"])
            self.assertEqual(budget["estimated_seconds"], seconds)
            self.assertEqual(refresh.minutes(group), max(6, refresh.math.ceil(seconds / 60)))
            for case in group["cases"]:
                self.assertIsInstance(case, dict)
                self.assertEqual(case, dict(model="watercnn8k", provider="AIX",
                                            variant=case["variant"], layout="default"))
                identity = (group["hw"], case["variant"])
                self.assertNotIn(identity, seen)
                seen.add(identity)
                env = refresh.case_env(group, case)
                for key in ("SCOREP_ENABLE_TRACING", "SCOREP_ENABLE_PROFILING"):
                    self.assertEqual(env[key], "true")
                    self.assertEqual(env[key + "_ENV"], "true")
                self.assertEqual(env["TOTAL_STEPS_ENV"], "12")
                self.assertEqual(int(env["TOTAL_STEPS_ENV"]) // 2 - 1, 5)
                self.assertEqual(env["USE_SCOREP_ENV"], "1")
                self.assertEqual(env["SKIP_COMPILE_ENV"], "1")
                prefix = HERE / "solver_cpp/build_thesis_aix_variants/default_scorep/aix-install"
                self.assertEqual(env["AIX_INSTALL_PREFIX_ENV"], str(prefix))
                self.assertEqual(env["GPU_REFRESH_AIX_LIBRARY_DIR_ENV"], str(prefix / "lib"))
                pipelined = case["variant"] == "p2p1"
                self.assertEqual(env["AIX_COMMUNICATION_MODE_ENV"], "pipelined" if pipelined else "collective")
                self.assertEqual(env["AIX_P2P_TIMELINE_ENV"], str(int(pipelined)))
                self.assertEqual(env["CPP_ML_CONFIG_ENV"], "config_aix_pipelined.toml" if pipelined else "config_aix.toml")
                if pipelined:
                    self.assertEqual(env["AIX_P2P_INITIAL_CREDITS_ENV"], "1")
                    self.assertEqual(env["AIX_P2P_ASYNC_CONTROLLER"], "0")
                else:
                    self.assertNotIn("AIX_P2P_INITIAL_CREDITS_ENV", env)
            with patch.object(refresh, "run", return_value=0) as run:
                with patch.object(refresh.os.sys, "argv", ["campaign", "--run", group["id"]]), patch.dict(os.environ, GPU_REFRESH_CAMPAIGN_ENV="aix_trace_test"):
                    self.assertEqual(refresh.main(), 0)
                run.assert_called_once_with(group, "aix_trace_test")
        self.assertEqual(seen, {(hw, "default") for hw in refresh.HARDWARE}
                         | {("96c4g", "p2p1"), ("96p96c4g", "p2p1")})
        with patch.object(refresh, "run") as run, contextlib.redirect_stderr(io.StringIO()):
            with patch.object(refresh.os.sys, "argv", ["campaign", "--run", "thesis_aix_trace_unlisted_r1"]):
                with self.assertRaises(SystemExit):
                    refresh.main()
            run.assert_not_called()

    def test_thesis_aix_trace_dry_run_readiness_and_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            mini = self.fixture(Path(tmp))
            variant = self.variant_fixture(mini, "default")
            groups = refresh.thesis_aix_trace_groups()
            with patch.object(refresh, "MINI", mini), patch.dict(os.environ, GPU_REFRESH_BUILD_JOB_ID_ENV="123"), contextlib.redirect_stdout(io.StringIO()) as output:
                with patch.object(refresh, "ready", wraps=refresh.ready) as ready:
                    refresh.submit("--dry-run", "aix_trace", groups)
                ready.assert_called_once_with("scorep", ["default"])
                root = mini / "results/gpu_refresh/aix_trace"
                with (root / "cases.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 6)
                self.assertTrue(all((r["warm"], r["steady"], r["steps"]) == ("1", "5", "12") for r in rows))
                self.assertEqual(set(json.loads((root / "aix-ready.json").read_text())), {"default_scorep"})
                self.assertEqual(len(json.loads((root / "budget.json").read_text())), 4)
                commands = [shlex.split(line) for line in output.getvalue().splitlines() if line.startswith("sbatch ")]
                self.assertEqual(len(commands), 4)
                for group, command in zip(groups, commands):
                    self.assertEqual([a for a in command if a.startswith("--time=")],
                                     [f"--time=00:{refresh.minutes(group):02d}:00"] * (2 if "p" in group["hw"] else 1))
                (variant / "READY").unlink()
                with self.assertRaises(OSError):
                    refresh.submit("--dry-run", "aix_trace_bad", groups)
                self.assertFalse((mini / "results/gpu_refresh/aix_trace_bad").exists())

    def test_thesis_aix_trace_runtime_checks_and_p2p_csv(self):
        from unittest.mock import MagicMock
        group = next(g for g in refresh.thesis_aix_trace_groups() if g["hw"] == "96c4g")
        for timeline in (True, False):
            with self.subTest(timeline=timeline), tempfile.TemporaryDirectory() as tmp:
                mini = self.fixture(Path(tmp))
                for name in ("config_aix.toml", "config_aix_pipelined.toml"):
                    shutil.copy(HERE / name, mini)

                def fake_popen(*args, **kwargs):
                    env = kwargs["env"]
                    run_id = env["RUN_ID_ENV"]
                    for step in range(2, 13, 2):
                        kwargs["stdout"].write(f"STEP_TIMING step={step} solver=ML step_ms=1\n")
                    scorep = mini / "scorep_runs" / f"gpu_refresh_{run_id}"
                    scorep.mkdir(parents=True)
                    (scorep / "profile.cubex").write_text("profile")
                    (scorep / "traces.otf2").write_text("trace")
                    if timeline and env["AIX_P2P_TIMELINE_ENV"] == "1":
                        csv_dir = mini / "logs" / f"aix_p2p_timeline_{run_id}"
                        csv_dir.mkdir(parents=True)
                        (csv_dir / "aix_p2p_timeline_rank_0.csv").write_text("step,time\n2,1\n")
                    return MagicMock(pid=99999999, wait=MagicMock(return_value=0))

                env = dict(SLURM_JOB_ID="456", SLURM_NTASKS="96", SLURM_JOB_NUM_NODES="1")
                with patch.object(refresh, "MINI", mini), patch.dict(os.environ, env), \
                     patch.object(refresh, "ready") as ready, \
                     patch.object(refresh, "aix_ready", return_value="verified READY") as aix_ready, \
                     patch.object(refresh, "clean_steps"), \
                     patch.object(refresh.subprocess, "Popen", side_effect=fake_popen), \
                     patch.object(refresh.os, "killpg", side_effect=ProcessLookupError):
                    self.assertEqual(refresh.run(group, "trace_test"), int(not timeline))
                self.assertEqual(ready.call_count, 3)
                self.assertTrue(all(call.args == ("scorep", ["default"]) for call in ready.call_args_list))
                self.assertEqual(aix_ready.call_count, 2)
                self.assertTrue(all(call.args == ("default", "scorep") for call in aix_ready.call_args_list))
                root = mini / "results/gpu_refresh/trace_test" / f"{group['id']}_456"
                with (root / "manifest.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([r["observed_ml_calls"] for r in rows], ["6", "6"])
                self.assertEqual(rows[1]["validation_error"], "" if timeline else "; Missing AIx P2P timeline CSV")
                self.assertTrue(all((root / r["run_id"] / "aix.READY").read_text() == "verified READY" for r in rows))

    def test_missing65_matrix_environment_and_budgets(self):
        groups = refresh.missing65_profile_groups()
        self.assertEqual(len(groups), 8)
        self.assertEqual(sum(len(g["cases"]) for g in groups), 65)
        self.assertEqual(len({g["id"] for g in groups}), 8)
        self.assertEqual([len(g["cases"]) for g in groups], [7, 9, 8, 8, 8, 9, 8, 8])
        self.assertGreater(len({refresh.minutes(g) for g in groups}), 2)
        seen = set()
        for group in groups:
            budget = group["budget"]
            self.assertEqual(budget["expected_calls_per_case"], 6)
            seconds = 60 + 1.5 * (budget["inference_seconds"] + budget["startup_seconds"] + budget["cleanup_seconds"])
            self.assertAlmostEqual(budget["estimated_seconds"], seconds)
            self.assertEqual(refresh.minutes(group), max(5, refresh.math.ceil(seconds / 60)))
            for case in group["cases"]:
                model, provider, variant, layout = refresh.case_fields(case)
                identity = (group["hw"], model, provider, variant, layout)
                self.assertNotIn(identity, seen)
                seen.add(identity)
                env = refresh.case_env(group, case)
                self.assertEqual(env["TOTAL_STEPS_ENV"], "12")
                self.assertEqual(env["USE_SCOREP_ENV"], "1")
                self.assertEqual(env["SCOREP_ENABLE_TRACING_ENV"], "false")
                self.assertEqual(env["SKIP_COMPILE_ENV"], "1")
                self.assertEqual(env["PHYDLL_REBUILD_ENV"], "0")
                if provider == "AIX":
                    self.assertFalse(layout == "default" and variant == "default")
                    self.assertFalse(model == "watercnn8k" and group["hw"] in ("24c1g", "24p24c1g")
                                     and layout == "default" and variant == "p2p1")
                    prefix = HERE / "solver_cpp/build_thesis_aix_variants" / f"{layout}_scorep/aix-install"
                    self.assertEqual(env["AIX_INSTALL_PREFIX_ENV"], str(prefix))
                    self.assertEqual(env["GPU_REFRESH_AIX_LIBRARY_DIR_ENV"], str(prefix / "lib"))
                    self.assertEqual(env["AIX_COMMUNICATION_MODE_ENV"], "collective" if variant == "default" else "pipelined")
                    if variant != "default":
                        self.assertEqual(env["AIX_P2P_INITIAL_CREDITS_ENV"], "1" if variant == "p2p1" else "24")
                else:
                    self.assertEqual((provider, variant), ("SMARTSIM", "pergpu"))
                    self.assertEqual(env["DB_LAYOUT_ENV"], "per-ml-node")
                    self.assertNotIn("GPU_REFRESH_AIX_LIBRARY_DIR_ENV", env)
            with patch.object(refresh, "run", return_value=0) as run:
                with patch.object(refresh.os.sys, "argv", ["campaign", "--run", group["id"]]), patch.dict(os.environ, GPU_REFRESH_CAMPAIGN_ENV="missing_test"):
                    self.assertEqual(refresh.main(), 0)
                self.assertEqual(run.call_args.args, (group, "missing_test"))
        self.assertEqual(sum(c[2] == "AIX" for c in seen), 62)
        self.assertEqual({(c[0], c[1]) for c in seen if c[2] == "SMARTSIM"},
                         {("24c1g", "giantmlp288"), ("24p24c1g", "watercnn8k"), ("24p24c1g", "giantmlp288")})

    def variant_fixture(self, mini, layout):
        variant = mini / "solver_cpp/build_thesis_aix_variants" / f"{layout}_scorep"
        library = variant / "aix-install/lib/libAIxeleratorService.so"
        library.parent.mkdir(parents=True)
        library.write_text("isolated library")
        source = variant / "source.sha256"
        solver = mini / "solver_cpp/build_gpu_refresh_scorep/terrain_solver"
        dl = mini.parent / "CPP-ML-Interface/dl_clients/build-gpu-refresh-scorep/phydll_dl_client"
        source.write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}\n" for p in (solver, dl)))
        linkage = variant / "library.ldd.txt"
        linkage.write_text("libscorep_mpp_mpi libscorep_adapter_user libtorch_cuda\n")
        checksums = variant / "checksums.sha256"
        checksums.write_text(source.read_text() + "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}\n" for p in (library, source, linkage)))
        entries = dict(format="THESIS_AIX_READY_V1", layout=layout, mode="scorep", job="999", gitsha=refresh.AIX_VARIANT_SHA,
                       gpu_refresh_job="123", prefix=str(library.parent.parent), library=str(library),
                       cache=str(variant / "aix-build/CMakeCache.txt"), solver=str(solver), dl_client=str(dl),
                       checksums=str(checksums), checksums_sha256=hashlib.sha256(checksums.read_bytes()).hexdigest(),
                       library_size=str(library.stat().st_size), library_mtime=str(int(library.stat().st_mtime)),
                       validation="cpu-build-only", runtime_validation="pending")
        (variant / "READY").write_text("".join(f"{k}={v}\n" for k, v in entries.items()))
        return variant

    def test_variant_readiness_and_missing65_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            mini = self.fixture(Path(tmp))
            variants = [self.variant_fixture(mini, layout) for layout in refresh.AIX_LAYOUTS]
            with patch.object(refresh, "MINI", mini), patch.dict(os.environ, GPU_REFRESH_BUILD_JOB_ID_ENV="123"):
                # Validate both the reused solver and the selected layout builds.
                with patch.object(refresh.subprocess, "run", wraps=subprocess.run) as checks:
                    refresh.ready("scorep", refresh.AIX_LAYOUTS)
                self.assertTrue(any("source.sha256" in str(call.args[0]) for call in checks.call_args_list))
                self.assertEqual(sum("checksums.sha256" in str(call.args[0]) for call in checks.call_args_list), 3)
                for variant in variants:
                    marker = variant / "READY"
                    original = marker.read_text()
                    for before, after in (("mode=scorep", "mode=plain"), ("gpu_refresh_job=123", "gpu_refresh_job=124"),
                                          (refresh.AIX_VARIANT_SHA, "badsha"), ("checksums_sha256=", "wrong_digest=")):
                        marker.write_text(original.replace(before, after))
                        with self.assertRaises(RuntimeError):
                            refresh.aix_ready(variant.name.removesuffix("_scorep"), "scorep")
                    marker.write_text(original)
                    library = variant / "aix-install/lib/libAIxeleratorService.so"
                    original_library = library.read_bytes()
                    library.write_bytes(b"corrupt")
                    with self.assertRaises(RuntimeError):
                        refresh.aix_ready(variant.name.removesuffix("_scorep"), "scorep")
                    library.write_bytes(original_library)
                    # Restore metadata after the deliberate corruption.
                    entries = dict(line.split("=", 1) for line in original.splitlines())
                    os.utime(library, (int(entries["library_mtime"]), int(entries["library_mtime"])))
                with contextlib.redirect_stdout(io.StringIO()) as output, patch.object(refresh.subprocess, "check_output", wraps=subprocess.check_output) as calls:
                    refresh.submit("--dry-run", "missing", refresh.missing65_profile_groups())
                    self.assertFalse(any(call.args[0][0] == "sbatch" for call in calls.call_args_list))
                self.assertEqual(output.getvalue().count("sbatch --parsable"), 8)
                root = mini / "results/gpu_refresh/missing"
                self.assertEqual(len(json.loads((root / "budget.json").read_text())), 8)
                self.assertEqual(set(json.loads((root / "ready.json").read_text())), {"scorep"})
                self.assertEqual(len(json.loads((root / "aix-ready.json").read_text())), 3)
                with (root / "cases.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 65)
                self.assertEqual({r["layout"] for r in rows}, set(refresh.AIX_LAYOUTS))
                self.assertTrue(all(r["aix_library_dir"].endswith(f"{r['layout']}_scorep/aix-install/lib") for r in rows if r["provider"] == "AIX"))

    def test_matrix_and_caps(self):
        groups = refresh.groups()
        self.assertEqual(len(groups), 28)
        self.assertEqual(sum(len(g["cases"]) for g in groups), 88)
        self.assertEqual(len({g["id"] for g in groups}), 28)
        self.assertEqual(sum(g["mode"] == "timing" for g in groups), 24)
        for g in groups:
            if g["mode"] == "timing":
                self.assertEqual([c[2] for c in g["cases"]], ["default", "readiness"] if g["repeat"] % 2 else ["readiness", "default"])
            for case in g["cases"]:
                e = refresh.case_env(g, case)
                water = case[0] == "watercnn8k"
                self.assertEqual(e["ML_BATCH_SIZE_ENV"], "20000000" if water else "500000")
                self.assertEqual(e["TOTAL_STEPS_ENV"], "202" if water and g["mode"] == "timing" else "12")
                self.assertEqual(e["SAVE_EVERY_ENV"], e["TOTAL_STEPS_ENV"])
                self.assertEqual((e["TARGET_WIDTH_ENV"], e["TARGET_HEIGHT_ENV"], e["CHUNK_SIZE_ENV"]), ("7680", "4320", "60") if water else ("480", "288", "16"))
                self.assertEqual(e["SCOREP_ENABLE_TRACING_ENV"], "false")
                self.assertEqual(e["USE_SCOREP_ENV"], str(int(g["mode"] == "profile")))
                self.assertEqual(e["SMARTSIM_THREADS_PER_QUEUE_ENV"], "1")
                if case[1] == "SMARTSIM":
                    self.assertEqual(e["SMARTSIM_MPI_SEQUENTIAL_PUT"],
                                     {"shared_c1": "1", "shared_c3": "3"}.get(case[2], "0"))
                if "p" in g["hw"]:
                    self.assertEqual((e["RANK_GRID_X_ENV"], e["RANK_GRID_Z_ENV"]), ("", ""))
                else:
                    self.assertEqual(int(e["RANK_GRID_X_ENV"]) * int(e["RANK_GRID_Z_ENV"]), int(e["MPI_RANKS_ENV"]))
        missing = [(g["hw"], c) for g in groups if g["mode"] == "profile" for c in g["cases"] if c[1] != "PHYDLL"]
        self.assertEqual(len(missing), 24)
        self.assertEqual(sum(c[1] == "AIX" for _, c in missing), 6)
        self.assertEqual(sum(c[1] == "SMARTSIM" and c[2] == "shared" for _, c in missing), 2)
        chain_cases = [(hw, c) for hw, c in missing
                       if c[1] == "SMARTSIM" and c[2] in ("shared_c1", "shared_c3")]
        self.assertEqual(len(chain_cases), 16)
        self.assertEqual({c[2] for _, c in chain_cases}, {"shared_c1", "shared_c3"})
        for variant in ("default", "readiness"):
            library = (HERE / f"config_phydll_gpu_refresh_{variant}.toml").read_text()
            self.assertIn('transport_layout = "packed"', library)
            self.assertIn(f'solver_readiness_wait = {str(variant == "readiness").lower()}', library)
            self.assertNotIn("wire_dtype", library)
            self.assertNotIn("float_codec", library)

    def test_profile_retry_excludes_valid_cases(self):
        groups = refresh.profile_retry_groups()
        self.assertEqual([len(g["cases"]) for g in groups], [6, 11, 10, 4])
        self.assertEqual([refresh.minutes(g) for g in groups], [35, 45, 55, 25])
        self.assertEqual(len({g["id"] for g in groups}), 4)
        self.assertFalse({g["id"] for g in groups} & {g["id"] for g in refresh.groups()})
        for g in groups:
            self.assertEqual(g["mode"], "profile")
            for case in g["cases"]:
                e = refresh.case_env(g, case)
                self.assertEqual(e["TOTAL_STEPS_ENV"], "12")
                self.assertEqual(e["USE_SCOREP_ENV"], "1")
                if g["hw"] == "24c1g":
                    self.assertNotIn(case[2], ("shared_c1", "shared_c3"))
                if g["hw"] == "96p96c4g":
                    self.assertEqual(case[1], "SMARTSIM")
        with patch.object(refresh, "run", return_value=0) as run:
            with patch.object(refresh.os.sys, "argv", ["campaign", "--run", groups[0]["id"]]):
                with patch.dict(os.environ, GPU_REFRESH_CAMPAIGN_ENV="profile_retry_test"):
                    self.assertEqual(refresh.main(), 0)
            self.assertEqual(run.call_args.args, (groups[0], "profile_retry_test"))

    def test_times_templates_and_syntax(self):
        for group in refresh.groups():
            expected = (65 if "p" in group["hw"] else 55) if group["mode"] == "profile" else (20 if group["cases"][0][0] == "giantmlp288" else 75 if "p" in group["hw"] else 60 if group["hw"] == "24c1g" else 40)
            self.assertEqual(refresh.minutes(group), expected)
        for hw in refresh.HARDWARE:
            file = HERE / f"gpu_refresh_{hw}.sbatch"
            subprocess.run(["bash", "-n", str(file)], check=True)
            sections = file.read_text().split("#SBATCH hetjob")
            self.assertEqual(len(sections), 2 if "p" in hw else 1)
            for section in sections:
                for directive in ("--account=rwth0792", "--partition=c23g", "--exclusive", "--mem=0"):
                    self.assertIn(directive, section)
            for section in sections:
                self.assertIn(f"--gres=gpu:{4 if hw.startswith('96') else 1}", section)
        subprocess.run(["zsh", "-n", str(HERE / "gpu_refresh_case.zsh")], check=True)

    def test_single_case_profile_recovery(self):
        group = refresh.profile_recovery_group()
        self.assertEqual(group["hw"], "24p24c1g")
        self.assertEqual(group["cases"], [("watercnn8k", "SMARTSIM", "shared_c1")])
        self.assertEqual(refresh.minutes(group), 20)
        self.assertEqual(refresh.case_env(group, group["cases"][0])["TOTAL_STEPS_ENV"], "12")
        with patch.object(refresh, "run", return_value=0) as run:
            with patch.object(refresh.os.sys, "argv", ["campaign", "--run", group["id"]]):
                with patch.dict(os.environ, GPU_REFRESH_CAMPAIGN_ENV="recovery_test"):
                    self.assertEqual(refresh.main(), 0)
            self.assertEqual(run.call_args.args, (group, "recovery_test"))

    def test_thesis_native_and_trace_groups(self):
        native = refresh.thesis_native_groups()
        self.assertEqual(len(native), 24)
        self.assertEqual(sum(len(g["cases"]) for g in native), 66)
        self.assertEqual({g["mode"] for g in native}, {"timing"})
        for hw in refresh.HARDWARE:
            per_model = refresh.smartsim_gap_cases(hw)
            for model in refresh.MODELS:
                self.assertEqual(sum(c[0] == model for c in per_model), {"24c1g": 3, "96c4g": 2, "24p24c1g": 4, "96p96c4g": 2}[hw])
        for g in native:
            for case in g["cases"]:
                e = refresh.case_env(g, case)
                self.assertEqual(e["TOTAL_STEPS_ENV"], "202" if case[0] == "watercnn8k" else "12")
                self.assertEqual(e["USE_SCOREP_ENV"], "0")
                if case[2] == "pergpu":
                    self.assertEqual(e["DB_LAYOUT_ENV"], "per-ml-node")
                    self.assertEqual(e["DB_NODES_ENV"], str(int(e["GPUS_PER_NODE_ENV"])))
                else:
                    self.assertEqual(e["DB_LAYOUT_ENV"], "shared")
                    self.assertEqual(e["DB_NODES_ENV"], "1")
        trace = refresh.thesis_trace_groups()
        self.assertEqual(len(trace), 2)
        self.assertEqual([g["mode"] for g in trace], ["trace", "trace"])
        for g in trace:
            for case in g["cases"]:
                e = refresh.case_env(g, case)
                self.assertEqual(e["USE_SCOREP_ENV"], "1")
                self.assertEqual(e["SCOREP_ENABLE_TRACING_ENV"], "true")
                self.assertEqual(e["SCOREP_ENABLE_PROFILING_ENV"], "true")
                self.assertEqual(e["TOTAL_STEPS_ENV"], "12")
        aix_p2p = refresh.case_env(trace[0], ("watercnn8k", "AIX", "p2p1"))
        self.assertEqual(aix_p2p["CPP_ML_CONFIG_ENV"], "config_aix_pipelined.toml")
        self.assertEqual(aix_p2p["AIX_P2P_INITIAL_CREDITS_ENV"], "1")
        self.assertEqual(aix_p2p["AIX_P2P_TIMELINE_ENV"], "1")
        self.assertEqual(aix_p2p["AIX_COMMUNICATION_MODE_ENV"], "pipelined")
        aix_coll = refresh.case_env(refresh.groups()[0], ("watercnn8k", "AIX", "default"))
        self.assertEqual(aix_coll["CPP_ML_CONFIG_ENV"], "config_aix.toml")
        self.assertNotIn("AIX_P2P_INITIAL_CREDITS_ENV", aix_coll)
        for g in native + trace:
            with patch.object(refresh, "run", return_value=0) as run:
                with patch.object(refresh.os.sys, "argv", ["campaign", "--run", g["id"]]):
                    with patch.dict(os.environ, GPU_REFRESH_CAMPAIGN_ENV="thesis_test"):
                        self.assertEqual(refresh.main(), 0)
                self.assertEqual(run.call_args.args, (g, "thesis_test"))

    def fixture(self, base, reviewed_solver=False):
        mini = base / "mini_app"
        mini.mkdir()
        for driver in ("5r_driver_single.zsh", "5r_driver_hetsurrogate.zsh"):
            shutil.copy(HERE / driver, mini / driver)
        for mode in ("plain", "scorep"):
            solver = mini / "solver_cpp" / f"build_gpu_refresh_{mode}"
            dl = base / "CPP-ML-Interface/dl_clients" / f"build-gpu-refresh-{mode}"
            solver.mkdir(parents=True)
            dl.mkdir(parents=True)
            prov = solver / "provenance-123"
            prov.mkdir()
            entries = []
            for path in (solver / "terrain_solver", dl / "phydll_dl_client"):
                path.write_text("#!/bin/sh\nexit 0\n")
                path.chmod(0o755)
                entries.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n")
                (prov / f"{path.name}.ldd.txt").write_text("libscorep_mpp_mpi\n" if mode == "scorep" else "libmpi\n")
            (prov / "binaries.sha256").write_text("".join(entries))
            (prov / "source.sha256").write_text(entries[0])
            if reviewed_solver:
                source = mini / "solver_cpp/terrain_solver.cpp"
                original = b"// fixture\n" + refresh.ORIGINAL_MODEL_PATH_CHECK
                source.write_bytes(b"// fixture\n" + refresh.APPROVED_MODEL_PATH_CHECK)
                other = mini / "5r_driver_single.zsh"
                (prov / "source.sha256").write_text(
                    f"{hashlib.sha256(original).hexdigest()}  {source}\n"
                    f"{hashlib.sha256(other.read_bytes()).hexdigest()}  {other}\n")
            cache = f"CMAKE_BUILD_TYPE:STRING=Release\nWITH_SCOREP:BOOL={'ON' if mode == 'scorep' else 'OFF'}\nPHYDLL_DL_WITH_TORCH:BOOL=ON\nCPPML_SCOREP_MPP:STRING=mpi\n"
            (dl / "CMakeCache.txt").write_text(cache)
            (solver / "CMakeCache.txt").write_text(cache + "".join(f"{f}:BOOL=ON\n" for f in ("USE_CPP_ML_INTERFACE", "WITH_PHYDLL", "WITH_SMARTSIM", "WITH_AIX", "WITH_TORCH")))
            (solver / "generated_registry.hpp").write_text("solver_readiness_wait")
            (solver / "cpp-ml-interface-build").mkdir()
            shutil.copy(solver / "generated_registry.hpp", solver / "cpp-ml-interface-build/generated_registry.hpp")
            phy = base / "CPP-ML-Interface/extern/phydll/build-gpu-refresh"
            (solver / "build.env").write_text(f"SOLVER_BIN={solver}/terrain_solver\nPHYDLL_DL_CLIENT={dl}/phydll_dl_client\nUSE_SCOREP={int(mode == 'scorep')}\nSCOREP_MPP=mpi\nPROVENANCE={prov}\nPHYDLL_BUILD_DIR={phy}\nGPU_REFRESH_LIBRARY_PATH={phy}/lib\n")
            marker = f"BUILD_READY job=123 mode={mode} completed=2026-10-11T00:00:00Z validation=cpu-build-only runtime_validation=pending\n"
            for file in (prov / "binaries.sha256", prov / "source.sha256"):
                marker += f"{hashlib.sha256(file.read_bytes()).hexdigest()}  {file}\n"
            for directory in (solver, dl):
                (directory / "BUILD_READY").write_text(marker)
        return mini

    def test_ready_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            mini = self.fixture(Path(tmp))
            with patch.object(refresh, "MINI", mini), patch.dict(os.environ, GPU_REFRESH_BUILD_JOB_ID_ENV="123"):
                refresh.ready("plain")
                refresh.ready("scorep")
                dl_marker = Path(tmp) / "CPP-ML-Interface/dl_clients/build-gpu-refresh-plain/BUILD_READY"
                original_marker = dl_marker.read_text()
                dl_marker.write_text(original_marker.replace("job=123", "job=124"))
                with self.assertRaises(RuntimeError):
                    refresh.ready("plain")
                dl_marker.write_text(original_marker)
                ldd = mini / "solver_cpp/build_gpu_refresh_plain/provenance-123/terrain_solver.ldd.txt"
                ldd.write_text("libscorep_mpp_mpi")
                with self.assertRaises(RuntimeError):
                    refresh.ready("plain")
                ldd.write_text("libmpi")
                with patch.dict(os.environ, GPU_REFRESH_BUILD_JOB_ID_ENV="456"):
                    with self.assertRaises(RuntimeError):
                        refresh.ready("plain")
                cache = mini / "solver_cpp/build_gpu_refresh_plain/CMakeCache.txt"
                cache.write_text(cache.read_text().replace("WITH_PHYDLL:BOOL=ON", "WITH_PHYDLL:BOOL=OFF"))
                with self.assertRaises(RuntimeError):
                    refresh.ready("plain")

    def test_fake_submit_and_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mini = self.fixture(base)
            tools = base / "tools"
            tools.mkdir()
            calls = base / "calls"
            sbatch = tools / "sbatch"
            sbatch.write_text('#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\nassert not any(k.startswith(("SLURM_", "SBATCH_", "SRUN_")) for k in os.environ)\nassert os.environ["GPU_REFRESH_BUILD_JOB_ID_ENV"] == "123"\nassert os.environ["OTHER_ENV"] == "preserved"\np=Path(os.environ["CALLS"])\na=json.loads(p.read_text()) if p.exists() else []\na.append(sys.argv[1:]);p.write_text(json.dumps(a));print(1000+len(a))\n')
            sbatch.chmod(0o755)
            watch = tools / "swatch"
            watch.write_text('#!/usr/bin/env python3\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["WATCH"]).write_text(json.dumps(sys.argv[1:]))\n')
            watch.chmod(0o755)
            env = dict(GPU_REFRESH_BUILD_JOB_ID_ENV="123", PATH=f"{tools}:{os.environ['PATH']}", GPU_REFRESH_SWATCH=str(watch), CALLS=str(calls), WATCH=str(base / "watch"))
            env.update(SLURM_MEM_PER_NODE="2048", SLURM_CPUS_PER_TASK="1", SLURM_JOB_ID="5027177",
                       SBATCH_ACCOUNT="thes2181", SRUN_MEM="2G", OTHER_ENV="preserved")
            with patch.object(refresh, "MINI", mini), patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()) as output:
                refresh.submit("--dry-run", "dry")
                self.assertFalse(calls.exists())
                self.assertFalse((base / "watch").exists())
                self.assertEqual(output.getvalue().count("sbatch --parsable"), 28)
                refresh.submit("--dry-run", "subset", [refresh.groups()[0]])
                with (mini / "results/gpu_refresh/subset/groups.csv").open() as f:
                    self.assertEqual(len(list(csv.DictReader(f))), 1)
                with (mini / "results/gpu_refresh/subset/cases.csv").open() as f:
                    self.assertEqual(len(list(csv.DictReader(f))), 2)
                refresh.submit("--submit", "actual")
                commands = json.loads(calls.read_text())
                self.assertEqual(len(commands), 28)
                self.assertTrue(all(not any("dependency" in a for a in cmd) for cmd in commands))
                for command in commands:
                    if "p" in Path(command[-2]).stem.removeprefix("gpu_refresh_"):
                        self.assertEqual(command.count(":"), 1)
                        times = [a for a in command if a.startswith("--time=")]
                        self.assertEqual(len(times), 2)
                        self.assertEqual(times[0], times[1])
                self.assertEqual(len(json.loads((base / "watch").read_text())), 29)
                with (mini / "results/gpu_refresh/actual/cases.csv").open() as f:
                    self.assertEqual(len(list(csv.DictReader(f))), 88)
                with self.assertRaises(RuntimeError):
                    refresh.submit("--submit", "actual")
                with patch.dict(os.environ, GPU_REFRESH_SNOTIFY=str(watch)):
                    refresh.submit("--submit", "recovery", [refresh.profile_recovery_group()], notifier="snotify")
                notify_args = json.loads((base / "watch").read_text())
                self.assertEqual(notify_args[:2], ["-N", "GPU Refresh Profiles"])
                self.assertEqual(notify_args[-1], "1029")
                with (mini / "results/gpu_refresh/recovery/cases.csv").open() as f:
                    self.assertEqual(len(list(csv.DictReader(f))), 1)
                for layout in refresh.AIX_LAYOUTS:
                    self.variant_fixture(mini, layout)
                with patch.dict(os.environ, GPU_REFRESH_SNOTIFY=str(watch)):
                    refresh.submit("--submit", "missing65", refresh.missing65_profile_groups(), notifier="snotify")
                commands = json.loads(calls.read_text())[-8:]
                self.assertEqual(len(commands), 8)
                expected = refresh.missing65_profile_groups()
                for command, group in zip(commands, expected):
                    self.assertEqual(command[-1], group["id"])
                    self.assertIn(f"--time=00:{refresh.minutes(group):02d}:00", command)
                    self.assertEqual(command.count(":"), int("p" in group["hw"]))
                self.assertEqual(json.loads((base / "watch").read_text())[-8:], [str(i) for i in range(1030, 1038)])
                self.assertTrue(all(not any(a.startswith("--dependency=") for a in cmd) for cmd in commands))
                self.assertEqual({refresh.case_fields(c)[3] for c in expected[0]["cases"]}, set(refresh.AIX_LAYOUTS))
                with patch.dict(os.environ, GPU_REFRESH_SNOTIFY=str(watch)):
                    refresh.submit("--submit", "missing65_smoke", expected, notifier="snotify", smoke_first=True)
                gated_commands = json.loads(calls.read_text())[-8:]
                for index, (command, original, group) in enumerate(zip(gated_commands, commands, expected)):
                    dependencies = [a for a in command if a.startswith("--dependency=")]
                    self.assertEqual(dependencies, ["--dependency=afterok:1038"] if index else [])
                    self.assertEqual(command.count(":"), int("p" in group["hw"]))
                    self.assertEqual([a for a in command if a.startswith("--time=")],
                                     [a for a in original if a.startswith("--time=")])
                    self.assertEqual(command[-2:], original[-2:])
                    if dependencies and ":" in command:
                        self.assertLess(command.index(dependencies[0]), command.index(":"))
                self.assertEqual(json.loads((base / "watch").read_text())[-8:], [str(i) for i in range(1038, 1046)])
                with (mini / "results/gpu_refresh/missing65_smoke/groups.csv").open() as f:
                    self.assertEqual([r["job_id"] for r in csv.DictReader(f)], [str(i) for i in range(1038, 1046)])
                with contextlib.redirect_stdout(io.StringIO()) as dry_output:
                    refresh.submit("--dry-run", "missing65_smoke_dry", expected, smoke_first=True)
                # The placeholder documents the gate without contacting Slurm.
                dry_commands = [shlex.split(line) for line in dry_output.getvalue().splitlines() if line.startswith("sbatch ")]
                self.assertEqual(len(dry_commands), 8)
                for index, command in enumerate(dry_commands):
                    self.assertEqual([a for a in command if a.startswith("--dependency=")],
                                     ["--dependency=afterok:DRY_RUN"] if index else [])
                self.assertEqual(len(json.loads(calls.read_text())), 45)
                self.assertTrue(all(os.environ[k] == v for k, v in env.items()))

    def test_shell_adapter_and_rebuild_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build = root / "build.env"
            build.write_text(f"SOLVER_BIN={root}/build_gpu_refresh_plain/terrain_solver\nPHYDLL_DL_CLIENT={root}/build-gpu-refresh-plain/phydll_dl_client\nPHYDLL_BUILD_DIR={root}/fresh\nGPU_REFRESH_LIBRARY_PATH={root}/libs\n")
            driver = root / "5r_driver_single.zsh"
            driver.write_text('EXTERNAL_DIR=/legacy\nPREP_H5="${EXTERNAL_DIR}/test.h5"\nprint -r -- "$PREP_H5"\nrealpath /old/extern/phydll/build/lib\n')
            env = dict(os.environ, GPU_REFRESH_BUILD_ENV=str(build), GPU_REFRESH_OUTPUT_ROOT=str(root / "output"))
            reply = subprocess.check_output(["zsh", str(HERE / "gpu_refresh_case.zsh"), str(driver)], env=env, text=True)
            self.assertEqual(reply.splitlines(), [f"{root}/output/test.h5", f"{root}/fresh/lib"])
            for skip in (1, 0):
                driver.write_text(f'SKIP_COMPILE={skip}\nRUN_SOLVER=1\nif [[ "$RUN_SOLVER" -eq 1 ]]; then\n'
                                  'if [[ "$SKIP_COMPILE" -eq 1 ]]; then\n'
                                  'print -r -- "Skipping compilation; tars=$SMARTSIM_CREATE_RUNTIME_TARS"\n'
                                  'else\necho "Compiling solver..."\ncmake -S /never-run\nfi\nfi\n')
                reply = subprocess.run(["zsh", str(HERE / "gpu_refresh_case.zsh"), str(driver)],
                                       env=env, capture_output=True, text=True)
                self.assertEqual(reply.returncode, 0 if skip else 90)
                if skip:
                    self.assertEqual(reply.stdout.strip(), "Skipping compilation; tars=0")
            for command in ('echo "Compiling solver..."', 'cmake -S /never-run',
                            'cmake --build /never-run', 'rm -rf "solver_cpp/never-run"'):
                with self.subTest(command=command):
                    driver.write_text(command + '\nexit 0\n')
                    reply = subprocess.run(["zsh", str(HERE / "gpu_refresh_case.zsh"), str(driver)],
                                           env=env, capture_output=True)
                    self.assertEqual(reply.returncode, 90)

    def test_group_failure_continues_only_after_safe_cleanup(self):
        from unittest.mock import MagicMock
        for unsafe in (False, True):
            with self.subTest(unsafe=unsafe), tempfile.TemporaryDirectory() as tmp:
                mini = self.fixture(Path(tmp))
                for variant in ("default", "readiness"):
                    shutil.copy(HERE / f"config_phydll_gpu_refresh_{variant}.toml", mini)
                group = refresh.groups()[0]
                processes = []

                def fake_popen(*args, **kwargs):
                    proc = MagicMock()
                    proc.pid = 99999999
                    proc.wait.return_value = 7 if not processes else 0
                    for step in range(2, int(kwargs["env"]["TOTAL_STEPS_ENV"]) + 1, 2):
                        kwargs["stdout"].write(f"STEP_TIMING step={step} solver=ML step_ms=1\n")
                    processes.append(kwargs["env"])
                    return proc

                cleanup = MagicMock(side_effect=[None, RuntimeError("unsafe")] if unsafe else None)
                env = dict(GPU_REFRESH_BUILD_JOB_ID_ENV="123", SLURM_JOB_ID="456", SLURM_NTASKS="24", SLURM_JOB_NUM_NODES="1", SSDB="stale",
                           SLURM_MEM_PER_NODE="2048", SLURM_MEM_PER_CPU="128", SLURM_MEM_PER_GPU="2048")
                with patch.object(refresh, "MINI", mini), patch.dict(os.environ, env), patch.object(refresh, "clean_steps", cleanup), patch.object(refresh.subprocess, "Popen", fake_popen), patch.object(refresh.os, "killpg", side_effect=ProcessLookupError):
                    # Mocking Popen also affects subprocess.run in ready().
                    with patch.object(refresh, "ready", return_value="validated"):
                        if unsafe:
                            with self.assertRaises(RuntimeError):
                                refresh.run(group, "test")
                        else:
                            self.assertEqual(refresh.run(group, "test"), 1)
                self.assertEqual(len(processes), 1 if unsafe else 2)
                manifest = mini / "results/gpu_refresh/test" / f"{group['id']}_456/manifest.csv"
                with manifest.open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), len(processes))
                self.assertEqual(rows[0]["rc"], "7")
                for e in processes:
                    self.assertNotIn("SSDB", e)
                    self.assertEqual(e["SLURM_MEM_PER_NODE"], "0")
                    self.assertNotIn("SLURM_MEM_PER_CPU", e)
                    self.assertNotIn("SLURM_MEM_PER_GPU", e)
                    self.assertIn("batch_size = 20000000", Path(e["CPP_ML_CONFIG_ENV"]).read_text())
                self.assertEqual(len({e["RUN_ID_ENV"] for e in processes}), len(processes))

    def test_missing65_failure_stops_after_manifest(self):
        from unittest.mock import MagicMock
        for group in refresh.missing65_profile_groups():
            # Zero driver status must also stop if output validation fails.
            for rc in (7, 137, 0):
                with self.subTest(group=group["id"], rc=rc), tempfile.TemporaryDirectory() as tmp:
                    mini = self.fixture(Path(tmp))
                    shutil.copy(HERE / "config_aix_pipelined.toml", mini)
                    ranks = "96" if group["hw"].startswith("96") else "24"
                    env = dict(SLURM_JOB_ID="456", SLURM_NTASKS=ranks, SLURM_JOB_NUM_NODES="1",
                               SLURM_HET_SIZE="2", SLURM_NTASKS_HET_GROUP_0=ranks,
                               SLURM_NTASKS_HET_GROUP_1=ranks, SLURM_JOB_NUM_NODES_HET_GROUP_0="1",
                               SLURM_JOB_NUM_NODES_HET_GROUP_1="1", SLURM_MEM_PER_NODE="2048",
                               SLURM_MEM_PER_CPU="128", SLURM_MEM_PER_GPU="2048")
                    proc = MagicMock(pid=99999999)
                    proc.wait.return_value = rc
                    with patch.object(refresh, "MINI", mini), patch.dict(os.environ, env), \
                         patch.object(refresh, "ready"), patch.object(refresh, "aix_ready", return_value="validated"), \
                         patch.object(refresh, "clean_steps") as cleanup, \
                         patch.object(refresh.subprocess, "Popen", return_value=proc) as popen, \
                         patch.object(refresh.os, "killpg", side_effect=ProcessLookupError):
                        self.assertEqual(refresh.run(group, "test"), 1)
                    popen.assert_called_once()
                    self.assertEqual(cleanup.call_count, 2)
                    childenv = popen.call_args.kwargs["env"]
                    self.assertEqual(childenv["SLURM_MEM_PER_NODE"], "0")
                    self.assertNotIn("SLURM_MEM_PER_CPU", childenv)
                    self.assertNotIn("SLURM_MEM_PER_GPU", childenv)
                    root = mini / "results/gpu_refresh/test" / f"{group['id']}_456"
                    with (root / "manifest.csv").open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(int(rows[0]["rc"]), rc or 1)
                    self.assertEqual(rows[0]["cleanup_safe"], "True")
                    out = root / rows[0]["run_id"]
                    self.assertEqual((out / "manifest.csv").read_text(), (root / "manifest.csv").read_text())
                    self.assertEqual(json.loads((out / "runtime.env.json").read_text()), childenv)


if __name__ == "__main__":
    unittest.main()
