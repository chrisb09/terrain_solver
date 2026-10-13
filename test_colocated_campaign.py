import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from summarize_colocated_campaign import summarize


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("direct_nopreload", "direct_preload", "cmi_baseline", "cmi_profiling", "cmi_tracing")


class CampaignTests(unittest.TestCase):
    def test_shell_syntax(self):
        for name in ("run_colocated_campaign.sh", "submit_colocated_campaign.sh", "colocated_campaign_env.sh"):
            subprocess.run(["bash", "-n", str(ROOT / name)], check=True)

    def test_cyclic_balance(self):
        orders = [[CONDITIONS[(rep + position) % 5] for position in range(5)] for rep in range(5)]
        for position in range(5):
            self.assertEqual({order[position] for order in orders}, set(CONDITIONS))

    def test_paired_summary_and_warmup(self):
        manifests = [Path(f"/campaign/cpu_production_r{rep}_{rep}/runs.tsv") for rep in range(1, 6)]
        text = "condition\trun_id\tlog\tstatus\twarmup\tsteps\n"
        text += "".join(f"{name}\tid\t/{name}.log\t0\t3\t20\n" for name in CONDITIONS)

        def log(path, **kwargs):
            offset = CONDITIONS.index(path.stem) * 10
            lines = []
            for step in range(1, 21):
                kind = "ML" if step % 2 == 0 else "Regular"
                value = 1000 if step <= 6 else 100 + offset
                lines.append(f"Step {step}, {kind}, local moved: 0, time: {value} ms")
                lines.append(f"STEP_TIMING step={step} solver={kind} step_ms={value} local_moved=0")
            return "\n".join(lines)

        with patch.object(Path, "glob", return_value=manifests), \
             patch.object(Path, "open", side_effect=lambda *a, **kw: io.StringIO(text)), \
             patch.object(Path, "read_text", log):
            result = summarize(Path("/campaign"), "cpu", "production")
        self.assertEqual(len(result["jobs"]), 5)
        self.assertEqual(result["excluded_jobs"], [])
        self.assertEqual(result["jobs"][0]["measured_calls"]["cmi_baseline"], 7)
        self.assertEqual(result["paired_comparisons"][0]["mean_difference_ms"], 10)
        self.assertEqual(result["paired_comparisons"][0]["ci95_difference_ms"], [10, 10])
        self.assertEqual(result["paired_comparisons"][0]["paired_jobs"], 5)

    def test_incomplete_job_excluded(self):
        with patch.object(Path, "glob", return_value=[Path("/campaign/job/runs.tsv")]), \
             patch.object(Path, "open", return_value=io.StringIO("condition\n")):
            result = summarize(Path("/campaign"), "cpu", "smoke")
        self.assertFalse(result["jobs"])
        self.assertEqual(len(result["excluded_jobs"]), 1)

    def test_guard_fails_wrong_environment(self):
        result = subprocess.run(["bash", "-c", f'source "{ROOT}/colocated_campaign_env.sh"',
                                 "solver_wrapper.sh"], env={"SKIP_PAPI_METRICS": "0"}, capture_output=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
