"""Exercise the real zsh adapter without builds, libraries, or cluster jobs."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ADAPTER = Path(__file__).resolve().parent / "gpu_refresh_case.zsh"
ZSH = shutil.which("zsh")


@unittest.skipUnless(ZSH, "zsh is required to exercise the adapter")
class GPURefreshCaseTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gpu refresh case ")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.selected = self.root / "selected/lib"
        self.selected.mkdir(parents=True)
        self.library = self.selected / "libAIxeleratorService.so"
        self.library.write_text("fake selected DSO\n")
        self.output = self.root / "output"
        self.output.mkdir()
        self.solver = self.root / "build_gpu_refresh_test/terrain_solver"
        dl_client = self.root / "build-gpu-refresh-test/phydll_dl_client"
        for binary in (self.solver, dl_client):
            binary.parent.mkdir()
            binary.write_text("fake binary, never executed\n")

        build_env = self.root / "build.env"
        build_env.write_text("".join(
            f"export {key}={shlex.quote(str(value))}\n"
            for key, value in {
                "SOLVER_BIN": self.solver,
                "PHYDLL_DL_CLIENT": dl_client,
                "PHYDLL_BUILD_DIR": self.root / "phydll",
                "GPU_REFRESH_LIBRARY_PATH": "/fresh/runtime/lib",
            }.items()
        ))
        setup = self.root / "setup.zsh"
        setup.write_text("export LD_LIBRARY_PATH=/setup/replaced/lib\n")
        self.driver = self.root / "5r_driver_single.zsh"
        self.driver.write_text(
            'source "$FAKE_SETUP"\n'
            '[[ $LD_LIBRARY_PATH == "$EXPECTED_PREFIX:/setup/replaced/lib" ]] || exit 73\n'
            '[[ $SMARTSIM_CREATE_RUNTIME_TARS == 0 ]] || exit 74\n'
            'print -r -- "$LD_LIBRARY_PATH" > "$GPU_REFRESH_OUTPUT_ROOT/driver.path.txt"\n'
        )
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        ldd = bin_dir / "ldd"
        ldd.write_text(
            "#!/bin/sh\n"
            '[ "$#" = 1 ] && [ "$1" = "$EXPECTED_SOLVER" ] || exit 75\n'
            'printf "%s\\n" "$1" > "$GPU_REFRESH_OUTPUT_ROOT/ldd.called.txt"\n'
            'case "$LD_LIBRARY_PATH" in "$EXPECTED_PREFIX":*) ;; *) exit 76 ;; esac\n'
            'printf "%s\\n" "$FAKE_LDD_OUTPUT"\n'
        )
        ldd.chmod(0o755)
        self.resolution = f"\tlibAIxeleratorService.so => {self.library} (0x1234)"
        self.env = {
            "PATH": str(bin_dir) + os.pathsep + os.defpath,
            "HOME": str(self.root),
            "LD_LIBRARY_PATH": "/inherited/lib",
            "GPU_REFRESH_BUILD_ENV": str(build_env),
            "GPU_REFRESH_OUTPUT_ROOT": str(self.output),
            "GPU_REFRESH_AIX_LIBRARY_DIR_ENV": str(self.selected),
            "FAKE_SETUP": str(setup),
            "EXPECTED_SOLVER": str(self.solver),
            "EXPECTED_PREFIX": f"{self.selected}:/fresh/runtime/lib",
            "FAKE_LDD_OUTPUT": self.resolution,
        }

    def run_adapter(self):
        return subprocess.run(
            [ZSH, "-f", str(ADAPTER), str(self.driver)],
            cwd=self.root, env=self.env, capture_output=True, text=True, timeout=10,
        )

    def test_selected_resolution_and_wrapped_source_preserve_prefix(self):
        result = self.run_adapter()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.output / "solver.ldd.txt").read_text(), self.resolution + "\n")
        self.assertEqual((self.output / "ldd.called.txt").read_text(), str(self.solver) + "\n")
        self.assertEqual(
            (self.output / "driver.path.txt").read_text(),
            f"{self.selected}:/fresh/runtime/lib:/setup/replaced/lib\n",
        )

    def test_mismatched_resolution_exits_91_before_driver(self):
        mismatch = "\tlibAIxeleratorService.so => /other/lib/libAIxeleratorService.so (0x1234)"
        self.env["FAKE_LDD_OUTPUT"] = mismatch
        result = self.run_adapter()
        self.assertEqual(result.returncode, 91, result.stdout + result.stderr)
        self.assertIn("solver did not resolve the selected AIx layout library", result.stderr)
        self.assertEqual((self.output / "solver.ldd.txt").read_text(), mismatch + "\n")
        self.assertFalse((self.output / "driver.path.txt").exists())

    def test_missing_dependency_exits_91_before_driver(self):
        self.env["FAKE_LDD_OUTPUT"] = self.resolution + "\n\tlibnvidia-ml.so.1 => not found"
        result = self.run_adapter()
        self.assertEqual(result.returncode, 91, result.stdout + result.stderr)
        self.assertIn("GPU solver has missing dependencies or CPU validation NVML", result.stderr)
        self.assertEqual(
            (self.output / "solver.ldd.txt").read_text(), self.env["FAKE_LDD_OUTPUT"] + "\n",
        )
        self.assertFalse((self.output / "driver.path.txt").exists())

    def test_cpu_validation_stub_dependency_exits_91_before_driver(self):
        self.env["FAKE_LDD_OUTPUT"] = (
            self.resolution + "\n\tlibnvidia-ml.so.1 => "
            "/variant/cpu-validation-nvml-stub/libnvidia-ml.so.1 (0x5678)"
        )
        result = self.run_adapter()
        self.assertEqual(result.returncode, 91, result.stdout + result.stderr)
        self.assertIn("GPU solver has missing dependencies or CPU validation NVML", result.stderr)
        self.assertEqual(
            (self.output / "solver.ldd.txt").read_text(), self.env["FAKE_LDD_OUTPUT"] + "\n",
        )
        self.assertFalse((self.output / "driver.path.txt").exists())

    def test_missing_selected_dso_exits_2_before_ldd_or_driver(self):
        self.library.unlink()
        result = self.run_adapter()
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        for name in ("ldd.called.txt", "solver.ldd.txt", "driver.path.txt"):
            self.assertFalse((self.output / name).exists(), name)

    def test_no_selection_skips_ldd_and_preserves_runtime_prefix(self):
        del self.env["GPU_REFRESH_AIX_LIBRARY_DIR_ENV"]
        self.env["EXPECTED_PREFIX"] = "/fresh/runtime/lib"
        self.library.unlink()
        result = self.run_adapter()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            (self.output / "driver.path.txt").read_text(),
            "/fresh/runtime/lib:/setup/replaced/lib\n",
        )
        self.assertFalse((self.output / "ldd.called.txt").exists())
        self.assertFalse((self.output / "solver.ldd.txt").exists())


if __name__ == "__main__":
    unittest.main()
