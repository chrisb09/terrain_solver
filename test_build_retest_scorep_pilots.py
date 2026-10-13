"""Login-safe regression tests: mock linkage tools, never build or submit jobs."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("build_retest_scorep_pilots.sbatch").read_text()


def section(start, end):
    return SCRIPT[SCRIPT.index(start):SCRIPT.index(end, SCRIPT.index(start))]


class BuildChecks(unittest.TestCase):
    def run_shell(self, code, directory, **environment):
        return subprocess.run(
            ["bash", "-c", "set -Eeuo pipefail\n" + code],
            cwd=directory, env={**os.environ, **environment},
            text=True, capture_output=True,
        )

    def test_linkage_policy(self):
        checks = section(
            'for binary in "$solver/terrain_solver" "$dl/phydll_dl_client" \\\n',
            'for binary in "$solver/terrain_solver" "$solver/cpp-ml-interface-build/libcpp_ml_interface_library.so"',
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("solver/terrain_solver", "dl/phydll_dl_client",
                         "prefix/lib/libAIxeleratorService.so",
                         "solver/cpp-ml-interface-build/libcpp_ml_interface_library.so",
                         "dl/libphydll_dl_runtime.so"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                path.chmod(0o755)
            (root / "provenance").mkdir()
            mock = r'''
solver="$PWD/solver"; dl="$PWD/dl"; prefix="$PWD/prefix"; provenance="$PWD/provenance"
fail() { printf '%s\n' "$*" >&2; exit 1; }
ldd() {
    printf '%s\n' "$*" >> "$PWD/calls"
    printf 'libscorep libmpi\n'
    if [[ "${*: -1}" != */libphydll_dl_runtime.so ]]; then printf 'libtorch\n'; fi
    if [[ "$1" == -r ]]; then
        if [[ "$MODE" == unresolved ]]; then printf 'undefined symbol: unrelated_symbol\n'; fi
        if [[ "$MODE" == scorep ]]; then printf 'undefined symbol: scorep_subsystems\n'; fi
        if [[ "$MODE" == exit ]]; then return 1; fi
    elif [[ "$MODE" == missing ]]; then
        printf 'libdependency.so => not found\n'
    fi
}
readelf() { printf 'SCOREP_User_RegionBegin\n'; }
'''
            result = self.run_shell(mock + checks, directory, MODE="clean")
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = (root / "calls").read_text().splitlines()
            self.assertEqual(len(calls), 5)
            self.assertTrue(all(call.startswith("-r ") for call in calls[:2]))
            self.assertTrue(all(not call.startswith("-r ") for call in calls[2:]))
            for mode in ("unresolved", "scorep", "exit", "missing"):
                with self.subTest(mode=mode):
                    result = self.run_shell(mock + checks, directory, MODE=mode)
                    self.assertNotEqual(result.returncode, 0)

    def test_provenance_archive_never_overwrites(self):
        archive = section('if [[ -d "$provenance" ]]; then', 'exec > >(tee')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "provenance").mkdir()
            (root / "provenance" / "old").write_text("original provenance")
            code = 'solver="$PWD"; provenance="$solver/provenance"; SLURM_JOB_ID=4859141\n' + archive
            for _ in range(2):
                result = self.run_shell(code, directory)
                self.assertEqual(result.returncode, 0, result.stderr)
            archives = list(root.glob("provenance_previous_4859141_*"))
            self.assertEqual(len(archives), 2)
            originals = list(root.glob("provenance_previous_*/provenance/old"))
            self.assertEqual(len(originals), 1)
            self.assertEqual(originals[0].read_text(), "original provenance")
            self.assertTrue((root / "provenance").is_dir())

    def test_resume_guards_and_cache_validation(self):
        checks = section('check_pilot_caches() {', 'for dir in "$mini/logs"')
        guards = section('if (( resume )); then', 'if (( preflight )); then')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            settings = {
                "WITH_SCOREP": "ON", "CPPML_SCOREP_MPP": "mpi",
                "SCOREP_MPP_SYSTEM": "mpi", "WITH_TORCH": "ON",
                "PHYDLL_DL_WITH_TORCH": "ON", "CMAKE_BUILD_TYPE": "Release",
                "SCOREP_ENABLE_COMPILER": "OFF", "SCOREP_ENABLE_USER": "ON",
                "CMAKE_CXX_COMPILER": f"{root}/scorep/bin/scorep-mpicxx",
                "CMAKE_C_COMPILER": f"{root}/scorep/bin/scorep-mpicc",
                "LIBTORCH_DIR": f"{root}/torch", "Torch_DIR": f"{root}/torch/share/cmake/Torch",
                "PHYDLL_BUILD_DIR": f"{root}/phy",
            }
            for target, source in (("solver", "mini/solver_cpp"), ("dl", "cmi/dl_clients")):
                (root / target).mkdir()
                text = "".join(f"{key}:STRING={value}\n" for key, value in settings.items())
                text += f"CMAKE_HOME_DIRECTORY:INTERNAL={root}/{source}\n"
                if target == "solver":
                    text += "USE_CPP_ML_INTERFACE:BOOL=ON\nWITH_PHYDLL:BOOL=ON\nWITH_AIX:BOOL=ON\n"
                (root / target / "CMakeCache.txt").write_text(text)
            setup = '''
solver="$PWD/solver"; dl="$PWD/dl"; mini="$PWD/mini"; cmi="$PWD/cmi"
scorep="$PWD/scorep"; torch="$PWD/torch"; phy="$PWD/phy"; aix="$PWD/aix"; resume=1
fail() { printf '%s\\n' "$*" >&2; exit 1; }
'''
            code = setup + checks + guards
            result = self.run_shell(code, directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            for key in ("WITH_SCOREP", "CPPML_SCOREP_MPP", "PHYDLL_DL_WITH_TORCH"):
                cache = root / "dl/CMakeCache.txt"
                original = cache.read_text()
                cache.write_text(original.replace(f"{key}:STRING={settings[key]}", f"{key}:STRING=wrong"))
                result = self.run_shell(code, directory)
                self.assertNotEqual(result.returncode, 0, key)
                cache.write_text(original)
            (root / "solver/BUILD_READY").touch()
            result = self.run_shell(code, directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ready build", result.stderr)

    def test_inherited_error_trap_redacts_arguments(self):
        handler = section('build_error() {', 'resume=0')
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_shell(handler + '\n(false secret-token)\n', directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("line=", result.stderr)
            self.assertIn("command=false", result.stderr)
            self.assertNotIn("secret-token", result.stderr)


if __name__ == "__main__":
    unittest.main()
