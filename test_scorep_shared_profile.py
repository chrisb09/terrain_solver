import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).parent
DRIVERS = ("5r_driver_single.zsh", "5r_driver_hetsurrogate.zsh")
TMP_ROOT = "/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode"


class SharedProfileTests(unittest.TestCase):
    def test_wrappers_preserve_shared_dir_and_historical_suffixes(self):
        for wrapper in ("solver_wrapper.sh", "5r_solver_wrapper.sh"):
            for shared in (None, "0", "1"):
                for tracing in ("false", "true", "1"):
                    with self.subTest(wrapper=wrapper, shared=shared, tracing=tracing), \
                            tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
                        base = str(Path(tmp) / "experiment")
                        env = dict(os.environ, SCOREP_EXPERIMENT_DIRECTORY=base,
                                   SCOREP_ENABLE_TRACING=tracing, OMPI_COMM_WORLD_RANK="7",
                                   _SOLVER_BINARY="/bin/bash", _CUDA_STUBS="",
                                   HET_SURROGATE_HIDE_GPU="1")
                        env.pop("SCOREP_SHARED_PROFILE_ENV", None)
                        if shared is not None:
                            env["SCOREP_SHARED_PROFILE_ENV"] = shared
                        result = subprocess.run(["bash", str(ROOT / wrapper), "-c",
                                                 'printf "RESULT=%s GPU=%s\\n" "$SCOREP_EXPERIMENT_DIRECTORY" "$CUDA_VISIBLE_DEVICES"'],
                                                env=env, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        expected = base + ("_rank_7" if shared != "1" and tracing == "false" else "")
                        self.assertIn(f"RESULT={expected} GPU=NoDevFiles", result.stdout)

    def test_shared_metadata_is_written_only_at_exit(self):
        for driver in DRIVERS:
            text = (ROOT / driver).read_text()
            start = text.index("    # Shared MPI profiles keep the base empty")
            end = text.index("\n  fi\nfi\n", start)
            block = text[start:end]
            self.assertIn('export SCOREP_SHARED_PROFILE_ENV="${SCOREP_SHARED_PROFILE_ENV:-0}"', text)
            for shared in ("0", "1"):
                with self.subTest(driver=driver, shared=shared), tempfile.TemporaryDirectory(dir=TMP_ROOT) as tmp:
                    env = dict(os.environ, MINI_APP_DIR=tmp, _scorep_dir_name="shared_test",
                               SCOREP_SHARED_PROFILE_ENV=shared, SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY="false",
                               MODEL_NAME="test", TARGET_WIDTH="24", TARGET_HEIGHT="24",
                               ML_BATCH_SIZE="1", MPI_RANKS="24", TOTAL_STEPS="6")
                    check = ('\nprint -r -- "overwrite=$SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY"; '
                             'if [[ -f "$SCOREP_EXPERIMENT_DIRECTORY/run_metadata.json" ]]; then '
                             'print metadata_before_exit; fi; '
                             ': > "$SCOREP_EXPERIMENT_DIRECTORY/scorep.cfg"; exit 7')
                    result = subprocess.run(["zsh", "-c", "set -eu;\n" + block + check],
                                            env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 7, result.stderr)
                    self.assertIn("overwrite=" + ("true" if shared == "1" else "false"), result.stdout)
                    self.assertEqual("metadata_before_exit" in result.stdout, shared == "0")
                    metadata = json.loads((Path(tmp) / "scorep_runs/shared_test/run_metadata.json").read_text())
                    self.assertEqual(metadata["ranks"], 24)

    def test_native_client_override_guards(self):
        found = 0
        for driver in DRIVERS:
            text = (ROOT / driver).read_text()
            blocks = re.findall(r'(?ms)^([ ]*)_phydll_scorep_opts=""\n(.*?)^\1fi$', text)
            for indent, body in blocks:
                found += 1
                for shared in (None, "0", "1"):
                    for tracing in ("false", "true", "1"):
                        env = dict(os.environ, USE_SCOREP="1", SCOREP_ENABLE_TRACING=tracing,
                                   SCOREP_EXPERIMENT_DIRECTORY="experiment")
                        env.pop("SCOREP_SHARED_PROFILE_ENV", None)
                        if shared is not None:
                            env["SCOREP_SHARED_PROFILE_ENV"] = shared
                        result = subprocess.run(["zsh", "-c", 'set -eu; _phydll_scorep_opts="";\n'
                                                 + body + indent + 'fi\nprint -r -- "$_phydll_scorep_opts"'],
                                                env=env, capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual("_cpp_client" in result.stdout,
                                         shared != "1" and tracing == "false")
        self.assertEqual(found, 3)

    def test_gpu_sampling_defaults_on_and_can_be_disabled(self):
        for driver in DRIVERS:
            text = (ROOT / driver).read_text()
            guard = text[:text.index("      nvidia-smi dmon")].splitlines()[-2]
            self.assertIn("GPU_MEM_SAMPLING_ENV:-1", guard)
            for sampling in (None, "0", "1"):
                env = dict(os.environ, USE_SCOREP="1")
                env.pop("GPU_MEM_SAMPLING_ENV", None)
                if sampling is not None:
                    env["GPU_MEM_SAMPLING_ENV"] = sampling
                result = subprocess.run(["zsh", "-c", guard + '\nprint enabled; fi'],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual("enabled" in result.stdout, sampling != "0")


if __name__ == "__main__":
    unittest.main()
