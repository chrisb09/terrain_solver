"""Native selection, independent-repeat statistics and real PGF/PDF contracts.

Run the renderer first. These tests read its owned artifacts, never change raw
inputs, submit jobs or write temporary files in another agent's output paths.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import unittest
from collections import Counter, defaultdict

import render_terrain_gpu_native as native


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = native.read_csv(native.TABLES / "runs.csv")
        cls.calls = native.read_csv(native.TABLES / "calls.csv")
        cls.values = native.read_csv(native.TABLES / "values.csv")
        cls.report = json.loads((native.TABLES / "report.json").read_text())

    def test_exact_native_matrix(self):
        self.assertEqual(self.report["status"], "ok")
        self.assertEqual(self.report["errors"], [])
        self.assertEqual(len(self.runs), 360)
        identities = [(r["model"], r["hardware"], (r["provider"], r["variant"]), int(r["rep"]))
                      for r in self.runs]
        self.assertEqual(set(identities), native.expected_matrix())
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(Counter(r["provider"] for r in self.runs),
                         {"aix": 216, "phydll": 48, "smartsim": 96})
        self.assertEqual(len(self.values), 120)
        self.assertTrue(all(r["selected_repetitions"] == "3" for r in native.read_csv(native.TABLES/"coverage.csv")))

    def test_student_t_not_calls(self):
        result = native.mean_ci([1, 2, 3])
        self.assertEqual(result["mean_s"], 2)
        self.assertEqual(result["sd_s"], 1)
        self.assertAlmostEqual(result["ci_half_s"], 4.302652729911275/math.sqrt(3))
        self.assertLess(result["ci_low_s"], 0)
        for bad in ([1]*15, [1, 2], [1, 2, float("nan")], [-1, 2, 3]):
            with self.assertRaises(ValueError):
                native.mean_ci(bad)

    def test_all_calls_and_repetition_means_independently(self):
        by_run = defaultdict(list)
        for call in self.calls:
            by_run[call["run_id"]].append(call)
        means = {}
        for run in self.runs:
            calls = by_run[run["run_id"]]
            n = 100 if run["model"] == "watercnn8k" else 5
            self.assertEqual(len(calls), n+1)
            self.assertEqual([int(c["step"]) for c in calls], list(range(2, 2*(n+1)+1, 2)))
            self.assertEqual([c["phase"] for c in calls], ["warmup"]+["measured"]*n)
            self.assertTrue(all(c["job_id"] == run["job_id"] for c in calls))
            mean = sum(float(c["latency_ms"])/1000 for c in calls[1:])/n
            self.assertAlmostEqual(mean, float(run["mean_s"]), places=12)
            means[run["run_id"]] = mean
        self.assertEqual(len(self.calls), 19260)
        for row in self.values:
            reps = [means[rid] for rid in row["run_ids"].split(";")]
            mean = sum(reps)/3
            sd = math.sqrt(sum((v-mean)**2 for v in reps)/2)
            half = 4.302652729911275*sd/math.sqrt(3)
            for key, expected in (("mean_s", mean), ("sd_s", sd), ("ci_low_s", mean-half),
                                  ("ci_high_s", mean+half)):
                self.assertAlmostEqual(float(row[key]), expected, places=11)

    def test_bad_schedule_rejected(self):
        good = "\n".join(f"STEP_TIMING step={i*2} solver=ML step_ms={i}" for i in range(1, 7))
        self.assertEqual(len(native.parse_calls(good, "giantmlp288")), 6)
        for bad in (good+"\n"+good, good.replace("step=4", "step=3"),
                    good.replace("step_ms=2", "step_ms=-2"), good.split("\n", 1)[1]):
            with self.assertRaises(ValueError):
                native.parse_calls(bad, "giantmlp288")

    def test_explicit_replacements_and_log_receipts(self):
        ledger = native.read_csv(native.TABLES / "selection_ledger.csv")
        selected = [r for r in ledger if r["decision"] == "selected"]
        self.assertEqual(len(selected), 360)
        aix = [r for r in self.runs if r["provider"] == "aix" and r["hardware"] == "24p24c1g"]
        self.assertEqual({r["job_id"] for r in aix}, set(native.AIX_REPLACEMENTS.values()))
        for row in selected:
            self.assertTrue(Path(row["source_path"]).exists())
            if row["nominal_solver_log"] and row["source_path"] != row["nominal_solver_log"]:
                self.assertFalse(Path(row["nominal_solver_log"]).exists())
                self.assertEqual(Path(row["source_path"]).name, "driver.log")
                self.assertEqual(Path(row["source_path"]).parent.name, row["run_id"])
                self.assertIn("exact run-directory", row["reason"])
        self.assertEqual(native.read_csv(native.TABLES/"selection_errors.csv"), [])

    def test_hashes_and_native_environment(self):
        for path, receipt in self.report["provenance"].items():
            data = Path(path).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), receipt["sha256"], path)
            self.assertEqual(len(data), receipt["bytes"])
        for run in self.runs:
            if run["environment_path"]:
                env = json.loads(Path(run["environment_path"]).read_text())
                self.assertEqual(env["USE_SCOREP_ENV"], "0")
                self.assertEqual(env["SCOREP_ENABLE_PROFILING_ENV"], "false")
                self.assertEqual(env["SCOREP_ENABLE_TRACING_ENV"], "false")

    def test_one_gpu_per_db_only_audited_not_baseline(self):
        baseline = native.read_csv(native.OUT/"ch6_baselines/latency_baselines_values.csv")
        self.assertEqual(len(baseline), 28)
        self.assertEqual(sum(r["variant"] == "pergpu_c0" for r in baseline), 4)
        self.assertFalse(any(r["variant"] == "pergpu_c0" and "1g" in r["hardware"] for r in baseline))
        separate = [r for r in self.runs if r["variant"] == "pergpu_c0" and "1g" in r["hardware"]]
        self.assertEqual(len(separate), 12)
        self.assertEqual(len({r["run_id"] for r in separate}), 12)
        simple = native.read_csv(native.OUT/"ch6_baselines/latency_baselines_without_pergpu_values.csv")
        self.assertEqual(len(simple), 24)
        self.assertEqual(simple, [r for r in baseline if r["variant"] != "pergpu_c0"])

    def test_layout_only_data_hashes(self):
        hashes = json.loads((native.TABLES/"layout_data_hashes.json").read_text())
        for relative, expected in hashes.items():
            self.assertEqual(hashlib.sha256((native.OUT/relative).read_bytes()).hexdigest(), expected)

    def test_readiness_upload_scheduling_and_layout_coverage(self):
        for folder, name, count in (("ch7_readiness", "readiness_latency", 24),
                                    ("ch7_smartsim_chaining", "chaining_latency", 24),
                                    *(("ch7_layout_pipeline_native", f"{layout}_native_latency", 24)
                                      for layout in native.LAYOUTS)):
            rows = native.read_csv(native.OUT/folder/f"{name}_values.csv")
            self.assertEqual(len(rows), count)
            self.assertEqual({r["hardware"] for r in rows}, set(native.HARDWARE))
            self.assertTrue(all(r["n_reps"] == "3" for r in rows))

    def test_caption_free_artist_contract(self):
        sys.path.insert(0, str(Path.home()/"Master-Thesis/gitlab/sourcecode"))
        import thesis_style as ts
        ts.apply()
        import matplotlib.pyplot as plt
        values = [{k: float(v) if k in ("mean_s", "ci_half_s", "ci_low_s", "ci_high_s") else v
                   for k, v in row.items()} for row in self.values]
        variants = [("aix", "default_coll"), ("phydll", "default"),
                    ("smartsim", "shared_c0"), ("smartsim", "pergpu_c0")]
        for models in (native.MODELS, (native.MODELS[0],)):
            fig = native.figure(values, variants, ["AIx", "PhyDLL", "SmartSim shared", "SmartSim per-GPU"],
                                ["#577787", "#C97AA7", "#8064A2", "#B89B47"], models, True)
            self.assertFalse(fig.texts)
            self.assertAlmostEqual(fig.get_figwidth()*25.4, 150 if len(models) == 2 else 73.5)
            for ax in fig.axes:
                self.assertFalse(ax.texts)
                self.assertEqual(len(ax.patches), 14)
                self.assertFalse(any(type(c).__name__ == "PathCollection" for c in ax.collections))
                if len(models) == 1:
                    self.assertEqual(ax.get_title(loc="left"), "")
            plt.close(fig)

        for axis_height in (32, 40, 41.6052):
            for models in (native.MODELS, (native.MODELS[0],), (native.MODELS[1],)):
                for per_gpu in (True, False):
                    labels = (["AIx", "PhyDLL", "SmartSim shared", "SmartSim per-GPU"]
                              if per_gpu else ["AIx", "PhyDLL", "SmartSim"])
                    fig = native.figure(values, variants if per_gpu else variants[:3], labels,
                                        ["#577787", "#C97AA7", "#8064A2", "#B89B47"],
                                        models, per_gpu, axis_height)
                    self.assertEqual([t.get_text() for t in fig.legends[0].texts], labels)
                    for ax, model in zip(fig.axes, models):
                        box = ax.get_position()
                        self.assertAlmostEqual(box.height*fig.get_figheight()*25.4, axis_height)
                        self.assertAlmostEqual(box.width*fig.get_figwidth()*25.4, 61)
                        expected = [r["mean_s"] for p, v in (variants if per_gpu else variants[:3])
                                    for hw in native.HARDWARE for r in values
                                    if (r["model"], r["hardware"], r["provider"], r["variant"])
                                    == (model, hw, p, v)
                                    and not (per_gpu and v == "pergpu_c0" and "1g" in hw)]
                        self.assertEqual([p.get_height() for p in ax.patches], expected)
                    plt.close(fig)

    def test_real_exports_dimensions_fonts_and_bounds(self):
        geometry = native.read_csv(native.TABLES/"figure_geometry.csv")
        self.assertEqual(len(geometry), 30)
        for row in geometry:
            path = native.OUT/row["figure"]
            self.assertEqual(row["text_bounds"], "inside canvas")
            self.assertTrue(0 < float(row["legend_header_gap_mm"]) <= 3.1)
            pgf = path.with_suffix(".pgf").read_text()
            self.assertNotIn("SmartSim SS", pgf)
            expected_height = (40 if row["figure"].startswith("ch7_readiness/") else
                               41.6052 if row["figure"].startswith(("ch7_layout_pipeline_native/", "ch7_layout_comparison/")) else 32)
            self.assertAlmostEqual(float(row["axes_height_mm"]), expected_height)
            self.assertEqual(float(row["pair_gap_mm"]), 3)
            self.assertIn(f"{float(row['width_mm'])/25.4:.6f}in", pgf)
            self.assertNotIn("\\caption", pgf)
            self.assertNotIn("Mean of", pgf)
            self.assertNotIn("(a)", pgf)
            info = subprocess.check_output(["pdfinfo", str(path.with_suffix(".pdf"))], text=True)
            size = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", info)
            self.assertIsNotNone(size)
            self.assertAlmostEqual(float(size[1])*25.4/72, float(row["width_mm"]), places=2)
            self.assertAlmostEqual(float(size[2])*25.4/72, float(row["height_mm"]), places=2)
            fonts = subprocess.check_output(["pdffonts", str(path.with_suffix(".pdf"))], text=True).splitlines()[2:]
            self.assertTrue(fonts)
            self.assertTrue(all("Type 1" in line and "yes yes" in line and "LMRoman" in line for line in fonts))
            self.assertGreater(path.with_suffix(".png").stat().st_size, 1000)

    def test_independent_pgf_examples(self):
        examples = [p for folder in ("ch6_baselines", "ch7_readiness", "ch7_smartsim_chaining",
                                     "ch7_layout_pipeline_native", "ch7_layout_comparison")
                    for p in (native.OUT/folder).glob("*_example.tex")]
        self.assertEqual(len(examples), 10)
        for path in examples:
            text = path.read_text()
            self.assertEqual(text.count(r"\begin{minipage}[t]{73.5mm}"), 2)
            self.assertIn(r"\hspace{3mm}", text)
            self.assertNotIn("resizebox", text)
            self.assertNotIn("scalebox", text)
            for filename in re.findall(r"\\input\{([^}]+)\}", text):
                self.assertTrue((path.parent/filename).is_file())
            result = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", path.name],
                                    cwd=path.parent, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertTrue(path.with_suffix(".pdf").is_file())

    def test_cross_layout_bars_against_raw_logs(self):
        sys.path.insert(0, str(Path.home()/"Master-Thesis/gitlab/sourcecode"))
        import thesis_style as ts
        ts.apply()
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex
        directory = native.OUT/"ch7_layout_comparison"
        self.assertEqual(len(list(directory.glob("*.pgf"))), 9)
        checked = set()
        for comm in native.COMMS:
            variants = [("aix", f"{layout}_{comm}") for layout in native.LAYOUTS]
            rows = native.read_csv(directory/f"{comm}_layout_latency_values.csv")
            runs = native.read_csv(directory/f"{comm}_layout_latency_runs.csv")
            self.assertEqual(len(rows), 24)
            self.assertEqual(len(runs), 72)
            expected_rows = [r for r in self.values if (r["provider"], r["variant"]) in variants]
            self.assertEqual(rows, expected_rows)
            numeric = []
            for row in rows:
                selected = sorted([r for r in runs if all(r[k] == row[k] for k in
                                  ("model", "hardware", "provider", "variant"))], key=lambda r: int(r["rep"]))
                self.assertEqual([r["rep"] for r in selected], ["1", "2", "3"])
                reps = []
                for run in selected:
                    text = Path(run["source_path"]).read_text()
                    parts = re.split(r"\[[^\n]*\] cell run_id=(\S+)", text)
                    bodies = [parts[i+1] for i in range(1, len(parts), 2) if parts[i] == run["run_id"]]
                    self.assertEqual(len(bodies), 1)
                    calls = [(int(step), float(ms)) for step, ms in re.findall(
                        r"STEP_TIMING step=(\d+) solver=ML step_ms=([0-9.eE+-]+)", bodies[0])]
                    n = 100 if row["model"] == native.MODELS[0] else 5
                    self.assertEqual([s for s, _ in calls], list(range(2, 2*(n+1)+1, 2)))
                    reps.append(sum(ms for _, ms in calls[1:])/n/1000)
                mean = sum(reps)/3
                half = 4.302652729911275*math.sqrt(sum((v-mean)**2 for v in reps)/2)/math.sqrt(3)
                for key, expected in (("mean_s", mean), ("ci_half_s", half),
                                      ("ci_low_s", mean-half), ("ci_high_s", mean+half),
                                      *((f"rep{i}_s", v) for i, v in enumerate(reps, 1))):
                    self.assertAlmostEqual(float(row[key]), expected, places=11)
                numeric.append(dict(row, mean_s=mean, ci_half_s=half,
                                    ci_low_s=mean-half, ci_high_s=mean+half))
                checked.add((row["model"], row["hardware"], row["variant"]))
            for models in (native.MODELS, (native.MODELS[0],), (native.MODELS[1],)):
                fig = native.figure(numeric, variants, ["Default", "Affine", "NUMA"],
                                    ["#577787", "#9784B4", "#568775"], models,
                                    axis_height_mm=41.6052)
                self.assertEqual([t.get_text() for t in fig.legends[0].texts], ["Default", "Affine", "NUMA"])
                self.assertFalse(fig.texts)
                for ax, model in zip(fig.axes, models):
                    expected = [r for _, variant in variants for hw in native.HARDWARE
                                for r in numeric if (r["model"], r["hardware"], r["variant"]) == (model, hw, variant)]
                    self.assertEqual([p.get_height() for p in ax.patches], [r["mean_s"] for r in expected])
                    self.assertEqual([to_hex(p.get_facecolor()) for p in ax.patches],
                                     [c for c in ("#577787", "#9784b4", "#568775") for _ in native.HARDWARE])
                    # Errorbar line segments retain both signed confidence endpoints.
                    intervals = [segment for collection in ax.collections for segment in collection.get_segments()]
                    for segment, row in zip(intervals, expected):
                        self.assertAlmostEqual(segment[0][1], row["ci_low_s"])
                        self.assertAlmostEqual(segment[1][1], row["ci_high_s"])
                    self.assertEqual(len(intervals), 12)
                    self.assertFalse(ax.texts)
                plt.close(fig)
        self.assertEqual(len(checked), 72)

    def test_gallery_inventory_and_links(self):
        stems = list(native.OUT.glob("*/*.pgf"))
        self.assertEqual(len(stems), 60)
        for path in stems:
            for ext in (".png", ".pdf"):
                self.assertTrue(path.with_suffix(ext).is_file())
        html = (native.OUT/"index.html").read_text()
        inventory = html.split('<!-- figure inventory -->', 1)[1].split('<!-- /figure inventory -->', 1)[0]
        self.assertEqual(len(re.findall(r'<img ', inventory)), 60)
        for path in stems:
            relative = path.relative_to(native.OUT).with_suffix("").as_posix()
            self.assertIn(f'src="{relative}.png"', inventory)
            for ext in ("png", "pdf", "pgf"):
                self.assertIn(f'href="{relative}.{ext}"', inventory)
        self.assertTrue(html.split('<nav>', 1)[1].startswith('<a href="#affinity-numa">AIx Affinity And NUMA</a>'))
        self.assertLess(html.index('id="affinity-numa"'), html.index('id="native"'))
        self.assertIn("WITH_HWLOC=ON", html)
        self.assertIn("NUMA-local rank groups", html)
        for link in re.findall(r'(?:href|src)="([^"]+)"', html):
            if not link.startswith("#"):
                self.assertTrue((native.OUT/link).is_file(), link)
        for comm in native.COMMS:
            self.assertIn(f"ch7_layout_comparison/{comm}_layout_latency_1x2.png", html)
        self.assertIn("ch6_baselines/latency_baselines_without_pergpu_1x2.png", html)
        for model in native.MODELS:
            for hw in native.HARDWARE:
                self.assertIn(f"ch6_phases/{model}_{hw}_phases_without_pergpu.png", html)

    def test_semantic_display_labels_and_preserved_keys(self):
        directory = native.OUT/"ch7_smartsim_chaining"
        rows = native.read_csv(directory/"chaining_latency_values.csv")
        self.assertEqual({r["variant"] for r in rows}, {"shared_c0", "shared_c1", "shared_c3"})
        for path in directory.glob("*.pgf"):
            text = path.read_text()
            for label in ("Default", "c=1", "c=3"):
                self.assertIn(label, text)
            self.assertNotRegex(text.lower(), r"chaining|\bc[013]\b")
        for path in (native.OUT/"ch7_layout_comparison").glob("*.pgf"):
            for label in ("Default", "Affine", "NUMA"):
                self.assertIn(label, path.read_text())
        for path in [native.OUT/"index.html", native.OUT/"README.md", native.TABLES/"README.md",
                     *(native.OUT/folder/"README.md" for folder in (
                         "ch6_baselines", "ch7_readiness", "ch7_smartsim_chaining",
                         "ch7_layout_pipeline_native", "ch7_layout_comparison"))]:
            text = path.read_text()
            # Persisted paths/keys may retain their spelling; visible prose must not.
            visible = re.sub(r'\([^\n)]*\)|<[^>]+>|`[^`]+`', '', text)
            self.assertNotIn("chaining", visible.lower(), str(path))
        self.assertIn("SmartSim upload scheduling", native.NOTES)


if __name__ == "__main__":
    unittest.main()
