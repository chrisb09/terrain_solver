import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import phydll_pin_exec as pin


class PinTests(unittest.TestCase):
    def run_mock(self, role="solver", rank=0, offset=0, count=24, budget=24,
                 rank_env="SLURM_LOCALID", allowed=None):
        allowed = set(range(100, 196)) if allowed is None else allowed
        masks = []

        def topology(path):
            cpu = int(path.parent.parent.name[3:])
            return "0" if path.name == "physical_package_id" else str((cpu - 100) % 96)

        def affinity(pid):
            return masks[-1] if masks else allowed

        with patch.dict(os.environ, {rank_env: str(rank)}, clear=True), \
                patch.object(Path, "read_text", topology), \
                patch.object(os, "sched_getaffinity", side_effect=affinity), \
                patch.object(os, "sched_setaffinity", side_effect=lambda pid, mask: masks.append(mask)), \
                patch.object(os, "execvp") as execute, \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            pin.main(["--budget", str(budget), "--role", role, "--rank-env", rank_env,
                      "--offset", str(offset), "--count", str(count), "--log", "--",
                      "program", "space argument", "--flag"])
            execute.assert_called_once_with("program", ["program", "space argument", "--flag"])
            self.assertEqual(os.environ["MLCOUPLING_INTRA_OP_THREADS"], "1")
            self.assertEqual(os.environ["MLCOUPLING_INTER_OP_THREADS"], "1")
            self.assertIn(f"role={role}", stderr.getvalue())
        return masks[0]

    def test_solver_24_and_96(self):
        for budget in (24, 96):
            for rank in (0, budget - 1):
                self.assertEqual(self.run_mock(rank=rank, count=budget, budget=budget), {100 + rank})

    def test_dl_shared_core_and_mpmd_offset(self):
        for source in ("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
            self.assertEqual(self.run_mock(role="dl", rank=27, offset=24, count=4,
                                           rank_env=source), {103})
        self.assertEqual(self.run_mock(role="dl", rank=1, count=4), {101})

    def test_smt_is_not_extra_physical_cores(self):
        with self.assertRaises(SystemExit) as error:
            self.run_mock(allowed=set(range(100, 112)) | set(range(196, 208)))
        self.assertEqual(error.exception.code, 2)

    def test_invalid_role_index_and_budget(self):
        for options in ({"rank": 24}, {"rank": 0, "offset": 24}, {"budget": 48},
                        {"count": 96}, {"count": 0}):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self.run_mock(**options)

    def test_actual_affinity_exec_argv_and_exit_code(self):
        cores = {}
        for cpu in sorted(os.sched_getaffinity(0)):
            root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            key = ((root / "physical_package_id").read_text(), (root / "core_id").read_text())
            cores.setdefault(key, cpu)
        if len(cores) < 24:
            self.skipTest("subprocess test needs 24 allowed physical cores")
        cpus = list(cores.values())[:24]
        code = "import json,os,sys; print(json.dumps([sorted(os.sched_getaffinity(0)),sys.argv[1:],os.environ['MLCOUPLING_INTRA_OP_THREADS']])); sys.exit(7)"
        env = dict(os.environ, OMPI_COMM_WORLD_LOCAL_RANK="25", SLURM_LOCALID="999")
        result = subprocess.run(["taskset", "-c", ",".join(map(str, cpus)), sys.executable,
                                 str(Path(pin.__file__)), "--budget", "24", "--role", "dl",
                                 "--rank-env", "OMPI_COMM_WORLD_LOCAL_RANK", "--offset", "24",
                                 "--count", "4", "--log", "--", sys.executable, "-c", code,
                                 "a b", ";literal"], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(json.loads(result.stdout), [[cpus[1]], ["a b", ";literal"], "1"])
        self.assertIn("index=1", result.stderr)


class LaunchTests(unittest.TestCase):
    def test_b_positive_constructs_only_then_launches_once(self):
        text = (Path(__file__).parent / "5r_driver_hetsurrogate.zsh").read_text()
        start = text.index('      if [[ "${G_SIDE_TASKS:-0}" -gt 0 ]]; then')
        end = text.index("      else\n      # HetJob mode: use srun", start)
        branch = text[start:end] + "\nfi\n"
        with tempfile.TemporaryDirectory(dir="/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode") as tmp:
            env = dict(os.environ, MINI_APP_DIR=tmp, G_SIDE_TASKS="24", G_SIDE_NODES="1",
                       MPI_RANKS="24", PHYDLL_NP_DL="4", RUN_ID="pin_test", USE_SCOREP="0",
                       SOLVER_HET_GROUP="0", DB_HET_GROUP="1", _nodes="1",
                       _ntasks_per_node_num="24", SOLVER_EXE="./5r_solver_wrapper.sh",
                       _ph_pin_solver="PIN_CPU", _ph_pin_gpu_solver="PIN_GPU",
                       _ph_pin_bind="--cpu-bind=none", SRUN_DIST="", _solver_step_gpus_args="",
                       DL_LD_LIBRARY_PATH="", device="GPU", GPUS_PER_NODE="4", DB_NODES="1",
                       ML_BATCH_SIZE="1", MODEL_PATH_FOR_SOLVER="model", MODEL_BACKEND="TORCH",
                       MODEL_IO_LAYOUT="flat_contiguous", PREP_H5="in.h5", TRAJ_H5="out.h5",
                       TOTAL_STEPS="6", SAVE_EVERY="6", SAVE_MODE="periodic", TRIANGULAR_SCALE="1",
                       CHUNK_SIZE="12", IO_MODE="rank0_gather", MPI_SYNC_MODE="none",
                       HDF5_XFER_MODE="independent")
            for rc in (0, 7):
                shell = ('set -eu; MODEL_IO_ARGS=(); RANK_GRID_ARGS=(); OVERWRITE_ARG=(); '
                         'DL_CLIENT_CMD=(client "space argument"); calls=0; '
                         'get_srun_het_flag() { print -r -- "--het-group=$1"; }; '
                         'srun() { (( calls += 1 )); return ' + str(rc) + '; };\n' + branch
                         + '\n[[ $calls == 0 ]]; SRUN_STATUS=0; '
                         'eval "${launch_cmd}" || SRUN_STATUS=$?; '
                         'print -r -- "calls=$calls rc=$SRUN_STATUS command=$launch_cmd"')
                result = subprocess.run(["zsh", "-c", shell], env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"calls=1 rc={rc}", result.stdout)
                self.assertIn("--ntasks \"28\"", result.stdout)
                self.assertEqual(result.stdout.count("--cpu-bind=none"), 2)
                scripts = Path(tmp) / "logs/dispatch_pin_test"
                self.assertIn("exec PIN_GPU ./5r_solver_wrapper.sh", (scripts / "solver.sh").read_text())
                self.assertIn(r"space\ argument", (scripts / "dl.sh").read_text())

    def test_phydll_common_executes_once_and_preserves_failure(self):
        root = Path(__file__).parent
        for driver in ("5r_driver_single.zsh", "5r_driver_hetsurrogate.zsh"):
            text = (root / driver).read_text()
            block = text[text.index("    DL_CLIENT_CMD=()"):text.index('      echo "PhyDLL launch_cmd failed')]
            self.assertEqual(block.count('eval "${launch_cmd}"'), 1)
            common = block[block.index("    SRUN_STATUS=0"):]
            for rc in (0, 7):
                result = subprocess.run(["zsh", "-c", 'set -eu; launch_cmd="(exit ' + str(rc) + ')"; '
                                         + common + '\nprint -r -- "$SRUN_STATUS"; fi'],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), "7" if rc else "")


if __name__ == "__main__":
    unittest.main()
