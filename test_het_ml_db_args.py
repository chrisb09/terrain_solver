"""No-launch checks for RedisAI argument forwarding in the het drivers."""

import os
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parent
DRIVERS = ("5r_driver_het.zsh", "5r_driver_hetsurrogate.zsh")
OPTIONS = {
    "ML_DB_BATCH_SIZE_ENV": "--ml-db-batch-size",
    "ML_DB_MIN_BATCH_SIZE_ENV": "--ml-db-min-batch-size",
    "ML_DB_MIN_BATCH_TIMEOUT_ENV": "--ml-db-min-batch-timeout",
}
FORWARDING = "        ${(j: :)${(q)ML_DB_ARGS[@]}} \\\n"


class HetMlDbArgsTest(unittest.TestCase):
    def test_syntax(self):
        for driver in DRIVERS:
            with self.subTest(driver=driver):
                subprocess.run(["zsh", "-n", str(ROOT / driver)], check=True)

    def test_constructed_commands(self):
        for driver in DRIVERS:
            source = (ROOT / driver).read_text()
            batching = source.split("ML_DB_ARGS=()", 1)[1].split(
                "# TensorFlow GPU backend", 1
            )[0]
            batching = "ML_DB_ARGS=()" + batching
            command = source.split('      _smartsim_solver_cmd="', 1)[1].split(
                '      SRUN_STATUS=$?', 1
            )[0]
            command = '      _smartsim_solver_cmd="' + command
            self.assertEqual(command.count(FORWARDING), 2)
            # Only evaluate argument construction, never source the job driver.
            variables = set(re.findall(r"\$\{([A-Za-z_][A-Za-z_0-9]*)", command))
            variables -= {"_smartsim_solver_cmd", "SLURM_HET_SIZE"}
            setup = "\n".join(f"{name}=1" for name in sorted(variables))
            setup += """
MODEL_IO_ARGS=(); RANK_GRID_ARGS=(); OVERWRITE_ARG=()
SOLVER_SRUN_EXTRA_ARGS=''; SRUN_DIST=''; SOLVER_EXE=solver_wrapper.sh
SOLVER_HET_GROUP=0; DB_HET_GROUP=1; MPI_RANKS=24
get_srun_het_flag() {
  if [[ -n "${SLURM_HET_SIZE:-}" ]]; then
    print -r -- "--het-group=$1"
  fi
}
srun() { printf '%s\\0' "$@"; }
"""
            samples = [{}, dict(zip(OPTIONS, ("128", "32", "7")))]
            samples += [{name: "9"} for name in OPTIONS]
            samples += [{name: "" for name in OPTIONS}]
            samples += [{"ML_DB_BATCH_SIZE_ENV": "128 with spaces"}]
            for het_size, side_tasks, components in (("2", "24", 2), ("2", "0", 1), ("", "0", 1)):
                env = {key: value for key, value in os.environ.items() if key not in OPTIONS}
                env["SLURM_HET_SIZE"] = het_size
                prefix = setup + f"\nG_SIDE_TASKS={side_tasks}\n"
                baseline = subprocess.run(
                    ["zsh", "-f"], input=prefix + command.replace(FORWARDING, ""),
                    env=env, text=True, capture_output=True, check=True,
                ).stdout.split("\n")[-1].split("\0")[:-1]
                for sample in samples:
                    with self.subTest(driver=driver, het=het_size, side=side_tasks, sample=sample):
                        result = subprocess.run(
                            ["zsh", "-f"], input=prefix + batching + command,
                            env=env | sample, text=True, capture_output=True, check=True,
                        )
                        # The optional batching diagnostic precedes the stub output.
                        argv = result.stdout.split("\n")[-1].split("\0")[:-1]
                        parts = [[]]
                        for arg in argv:
                            if arg == ":":
                                parts.append([])
                            else:
                                parts[-1].append(arg)
                        self.assertEqual(len(parts), components)
                        expected = {flag: sample[name] for name, flag in OPTIONS.items() if sample.get(name)}
                        for part in parts:
                            actual = {arg: part[i + 1] for i, arg in enumerate(part) if arg in OPTIONS.values()}
                            self.assertEqual(actual, expected)
                        stripped = []
                        index = 0
                        while index < len(argv):
                            if argv[index] in OPTIONS.values():
                                index += 2
                            else:
                                stripped.append(argv[index])
                                index += 1
                        self.assertEqual(stripped, baseline)


if __name__ == "__main__":
    unittest.main()
