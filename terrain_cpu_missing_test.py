"""Offline terrain-only mapping and plain timing contracts; never submit jobs."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class MissingTerrainTests(unittest.TestCase):
    def test_mapping_and_budget(self):
        for mode, group, tasks, configs in (('memory', 'other', 10, 3),
                                           ('timing', 'other', 10, 5),
                                           ('timing', 'phydll', 20, 1)):
            for task in range(tasks):
                env = {**os.environ, 'MODE': mode, 'TERRAIN_GROUP': group,
                       'PLAN_ONLY': '1', 'SLURM_ARRAY_TASK_ID': str(task)}
                env.pop('STEPS', None)
                result = subprocess.run(['bash', str(HERE / 'terrain_cpu_missing.sbatch')],
                                        env=env, capture_output=True, text=True, check=True)
                self.assertIn(f'model={"watercnn" if task % 10 < 5 else "giant"}', result.stdout)
                self.assertIn(f'rep={task % 5 + 1}', result.stdout)
                self.assertIn(f'steps={6 if mode == "memory" else 22}', result.stdout)
                self.assertEqual(len(result.stdout.split('configs=')[1].split()), configs)
        text = (HERE / 'terrain_cpu_missing.sbatch').read_text()
        self.assertIn('task % 10', text)
        self.assertIn('TERRAIN_READINESS=true', text)
        self.assertIn('exec bash -c "$(<"$mini/terrain_cpu_final.sbatch")"', text)

    def test_timing_record_has_no_profile_and_keeps_decimal(self):
        text = (HERE / 'terrain_cpu_final.sbatch').read_text()
        script = text.split('record() {', 1)[1].split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'PHYDLL_CPP'
            directory.mkdir()
            (directory / 'solver.log').write_text(''.join(
                f'STEP_TIMING step={step} solver=ML step_ms=1.123456789123456789 rank_min_ms=1 rank_max_ms=2\n'
                for step in (2, 4, 6)))
            env = {**os.environ, 'MODE': 'timing', 'MODEL': 'giant', 'REP': '5', 'STEPS': '6',
                   'ROOT': tmp, 'RUN_ROOT': tmp, 'TERRAIN_PLAIN_BUILD': '/plain', 'TERRAIN_READINESS': 'true'}
            env.pop('TERRAIN_RETEST_BUILD', None)
            subprocess.run([sys.executable, '-c', script, str(directory), 'SUCCESS', '1', '0', '1', '0', '0'],
                           env=env, cwd=HERE, check=True)
            with (root / 'results.csv').open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['mean_step_ms'], '1.123456789123456789')
            self.assertEqual(row['cubex'], '')
            self.assertEqual(row['scorep'], '0')
            self.assertEqual(row['memory_sampler'], '0')
            self.assertEqual(row['solver_readiness_wait'], 'true')

    def test_aix_sampler_census_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'memory_summary.json'
            data = dict(coverage_ok=True, live_read_errors=0, cleanup_sigkill_pids=[],
                        expected_processes=dict(solver=96, dl=0, db=0),
                        max_direct_processes=dict(solver=96, dl=0, db=0), simultaneous_coverage_samples=2)
            command = [sys.executable, str(HERE / 'terrain_cpu_diagnostics.py'), '--validate-memory', str(path),
                       '--memory-config', 'AIx']
            path.write_text(json.dumps(data))
            subprocess.run(command, check=True)
            for role in ('solver', 'dl', 'db'):
                bad = {**data, 'max_direct_processes': {**data['max_direct_processes'], role: 1}}
                path.write_text(json.dumps(bad))
                self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_plain_gate_contract(self):
        gate = (HERE / 'terrain_cpu_plain_gate.sh').read_text()
        for check in ('WITH_SCOREP:BOOL=OFF', 'USE_SCOREP|scorep-mpi|finstrument-functions',
                      'libscorep', 'SCOREP_*', '__cyg_profile', 'source.sha256', 'binaries.sha256'):
            self.assertIn(check, gate)
        text = (HERE / 'terrain_cpu_final.sbatch').read_text()
        timing_exports = text.split('if [[ $MODE == timing ]]; then\n    exports=', 1)[1].split('\nfi', 1)[0]
        self.assertNotIn('SCOREP', timing_exports)
        self.assertIn('if [[ $MODE == memory ]]; then', text)
        self.assertIn('[[ ! -e $dir/scorep && ! -e $dir/memory_samples.csv', text)
        self.assertLess(text.index('export TERRAIN_DB_LIBRARY_PATH=$LD_LIBRARY_PATH'),
                        text.index('source "$MINI_APP_DIR/terrain_cpu_plain_gate.sh"'))
        self.assertIn('export LD_LIBRARY_PATH=$TERRAIN_DB_LIBRARY_PATH', text)
        self.assertIn('sha256sum controller.env > controller_environment.sha256', text)

    def test_invalid_mapping_and_submission_manifest(self):
        for mode, group, task in (('memory', 'phydll', '0'), ('timing', 'other', '10'),
                                  ('timing', 'phydll', '20'), ('timing', 'other', '-1'),
                                  ('performance', 'other', '0')):
            result = subprocess.run(['bash', str(HERE / 'terrain_cpu_missing.sbatch')],
                                    env={**os.environ, 'PLAN_ONLY': '1', 'MODE': mode,
                                         'TERRAIN_GROUP': group, 'SLURM_ARRAY_TASK_ID': task},
                                    capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        path = HERE / 'terrain_cpu_missing_terrain_missing_20261012_jobs.csv'
        if path.exists():
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertGreaterEqual(len(rows), 8)
            self.assertTrue(all(None not in row and len(row) == 8 for row in rows))
            self.assertEqual([r['array'] for r in rows[:3]], ['0,5%2', '0,5%2', '0,5,10,15%2'])
            self.assertEqual([r['afterok'] for r in rows[3:6]], [r['job_id'] for r in rows[:3]])
            self.assertEqual(rows[7]['afterok'], rows[6]['job_id'])

    def test_superseded_success_rows_are_rejected(self):
        from terrain_cpu_final_analysis import collect
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rep = root / 'job_failed' / 'watercnn' / 'rep1'
            rep.mkdir(parents=True)
            (rep / 'results.csv').write_text('status\nSUCCESS\n')
            (rep / 'SUPERSEDED.json').write_text('{"include_in_production_measurements": false}\n')
            with self.assertRaisesRegex(ValueError, 'superseded Slurm-failed task'):
                collect([root], 'memory', 5, 2)


if __name__ == '__main__':
    unittest.main()
