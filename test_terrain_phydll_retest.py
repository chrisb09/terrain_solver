import os
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

from validate_terrain_phydll_profile import validate

ROOT = Path(__file__).resolve().parent


def tree(dl=False, calls=10):
    names = ['phydll_dl_client', 'dl_recv', 'dl_torch_forward', 'dl_send_output'] if dl else ['terrain_solver', 'solver_step_ml_steady']
    return {i: SimpleNamespace(name=name, parent_id=None if i == 0 else 0,
                               visits=1 if i == 0 else calls, incl_time=1.0, excl_time=1.0)
            for i, name in enumerate(names)}


class RetestTests(unittest.TestCase):
    def test_all_plans(self):
        for mode in ('performance', 'memory'):
            for task in range(20):
                result = subprocess.run(['bash', str(ROOT / 'terrain_phydll_retest.sbatch')],
                    env={**os.environ, 'PLAN_ONLY': '1', 'MODE': mode,
                         'SLURM_ARRAY_TASK_ID': str(task), 'RETEST_CAMPAIGN': 'fixture'},
                    capture_output=True, text=True, check=True)
                text = result.stdout
                self.assertIn(f'readiness={str(task >= 10).lower()}', text)
                self.assertIn('model=' + ('watercnn' if task % 10 < 5 else 'giant'), text)
                self.assertIn(f'rep={task % 5 + 1}', text)
                self.assertIn('steps=' + ('22' if mode == 'performance' else '6'), text)
                self.assertIn('configs=PHYDLL_CPP', text)

    def test_bad_plan(self):
        for task in ('20', '-1', 'abc'):
            result = subprocess.run(['bash', str(ROOT / 'terrain_phydll_retest.sbatch')],
                env={**os.environ, 'PLAN_ONLY': '1', 'SLURM_ARRAY_TASK_ID': task}, capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_profile_coverage(self):
        for steps in (6, 22):
            calls = (steps - 2) // 2
            trees = [tree(calls=calls) for _ in range(96)] + [tree(True, calls) for _ in range(96)]
            self.assertEqual(validate(trees, steps, 'fixture')['steady_steps'], calls)
            with self.assertRaises(ValueError):
                validate(trees[:-1], steps, 'fixture')
            trees[-1][2].visits += 1
            with self.assertRaises(ValueError):
                validate(trees, steps, 'fixture')

    def test_output_barriers_rejected(self):
        trees = [tree() for _ in range(96)] + [tree(True) for _ in range(96)]
        trees[-1][4] = SimpleNamespace(name='MPI_Barrier', parent_id=3, visits=10, incl_time=0.1, excl_time=0.1)
        with self.assertRaises(ValueError):
            validate(trees, 22, 'fixture')

    def test_configs_and_isolation(self):
        for readiness in ('false', 'true'):
            text = (ROOT / f'config_terrain_phydll_{readiness}.toml').read_text()
            self.assertIn(f'solver_readiness_wait = {readiness}', text)
            self.assertIn('batch_size = 500000', text)
        build = (ROOT / 'build_terrain_phydll_retest.sbatch').read_text()
        self.assertIn('-DWITH_AIX=OFF', build)
        self.assertIn('"BUILD=$phy"', build)
        self.assertNotIn('--resume', build)
        self.assertIn('ls-remote origin refs/heads/master', build)


if __name__ == '__main__':
    unittest.main()
