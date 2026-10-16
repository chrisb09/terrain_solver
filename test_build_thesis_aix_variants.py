"""Exercise CPU-only loader validation without building or submitting jobs."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


BUILDER = Path(__file__).resolve().parent / "build_thesis_aix_variants.sbatch"
BASH = shutil.which("bash")


@unittest.skipUnless(BASH, "bash is required to exercise builder validation")
class BuildThesisAIxVariantsTest(unittest.TestCase):
    def test_cpu_validation_is_command_local_and_scorep_only(self):
        source = BUILDER.read_text()
        start = '        validation_ld="$runtime_ld"\n'
        end = '        LD_LIBRARY_PATH="$validation_ld" ldd -r "$library" > "$variant/library.ldd.txt" 2>&1\n'
        block = start + source.split(start, 1)[1].split(end, 1)[0] + end
        checksum_line = '            sha256sum "$variant/cpu-validation-nvml-stub/libnvidia-ml.so.1" >> "$variant/checksums.sha256"\n'
        checksum_block = '        if [[ "$mode" == scorep ]]; then\n' + checksum_line + '        fi\n'
        self.assertIn(checksum_block, source)

        for mode in ("plain", "scorep"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="aix validation ") as temporary:
                root = Path(temporary)
                variant = root / mode
                variant.mkdir()
                stubs = root / "stubs"
                stubs.mkdir()
                stub_bytes = b"fake NVML stub\x00\xff\n"
                (stubs / "libnvidia-ml.so").write_bytes(stub_bytes)
                library = variant / "libAIxeleratorService.so"
                library.touch()
                bin_dir = root / "bin"
                bin_dir.mkdir()
                ldd = bin_dir / "ldd"
                ldd.write_text(
                    '#!/bin/sh\n'
                    '[ "$#" = 2 ] && [ "$1" = -r ] && [ "$2" = "$library" ] || exit 75\n'
                    'printf "%s\\n" "$LD_LIBRARY_PATH"\n'
                )
                ldd.chmod(0o755)
                runtime_ld = "/runtime/aix/lib:/runtime/torch/lib"
                original_ld = "/original/runtime/lib"
                result = subprocess.run(
                    [BASH, "-eu", "-c", block + checksum_block
                     + 'printf "runtime_ld=%s\\nLD_LIBRARY_PATH=%s\\n" "$runtime_ld" "$LD_LIBRARY_PATH"'],
                    env={**os.environ, "PATH": str(bin_dir) + os.pathsep + os.defpath,
                         "mode": mode, "variant": str(variant), "stubs": str(stubs),
                         "library": str(library), "runtime_ld": runtime_ld,
                         "LD_LIBRARY_PATH": original_ld},
                    capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue(result.stdout.endswith(
                    f"runtime_ld={runtime_ld}\nLD_LIBRARY_PATH={original_ld}\n"
                ), result.stdout)
                validation_dir = variant / "cpu-validation-nvml-stub"
                expected_ld = runtime_ld
                if mode == "scorep":
                    expected_ld = f"{validation_dir}:{runtime_ld}"
                    copied = validation_dir / "libnvidia-ml.so.1"
                    self.assertTrue(copied.is_file())
                    self.assertFalse(copied.is_symlink())
                    self.assertEqual(copied.read_bytes(), stub_bytes)
                    self.assertEqual(
                        (variant / "environment.txt").read_text(),
                        f"CPU_VALIDATION_NVML_STUB={validation_dir}\n",
                    )
                    self.assertEqual(
                        (variant / "checksums.sha256").read_text(),
                        f"{hashlib.sha256(stub_bytes).hexdigest()}  {copied}\n",
                    )
                else:
                    self.assertFalse(validation_dir.exists())
                    self.assertFalse((variant / "environment.txt").exists())
                    self.assertFalse((variant / "checksums.sha256").exists())
                self.assertEqual((variant / "library.ldd.txt").read_text(), expected_ld + "\n")

    def test_published_runtime_excludes_cpu_validation_stub(self):
        source = BUILDER.read_text()
        runtime_lines = [line.strip() for line in source.splitlines() if 'runtime_ld=' in line]
        self.assertEqual(runtime_lines, [
            'runtime_ld="$prefix/lib:$torch/lib:$cuda/lib64:$cuda/extras/CUPTI/lib64:$hwloc/lib:$base_ld"',
            '[[ "$mode" != scorep ]] || runtime_ld="$scorep/lib:$papi/lib:$runtime_ld"',
        ])
        self.assertIn('export LD_LIBRARY_PATH="$runtime_ld"\n        validation_ld="$runtime_ld"', source)
        self.assertIn('"$LD_LIBRARY_PATH" "$runtime_ld" > "$variant/environment.txt"', source)


if __name__ == "__main__":
    unittest.main()
