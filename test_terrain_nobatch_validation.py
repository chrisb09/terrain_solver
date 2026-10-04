"""Coverage and submission-environment tests; never submit scheduler jobs."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from extract_scorep_tables import ORDER, validate_campaign_coverage

MINI_APP = Path(__file__).resolve().parent


def coverage(reps=(1,), steps=2):
    memory, phases = [], []
    for model in ('watercnn', 'giant'):
        for cfg in ORDER:
            for rep in reps:
                meta = dict(model=model, config=cfg, rep=str(rep))
                memory.append(dict(meta, solver_ranks='96', steady_steps=str(steps)))
                for view, scope in [('application', 'solver_steady'),
                                    ('provider_dl', 'dl_steady_count_validated') if cfg == 'PHYDLL_CPP'
                                    else ('provider', 'solver_steady')]:
                    phases.append(dict(meta, view=view, scope=scope, phase='synthetic',
                                       rank_count='96', steady_steps=str(steps),
                                       cubex=f'/synthetic/{model}/{cfg}/rep{rep}/profile.cubex'))
    return memory, phases


class CampaignCoverage(unittest.TestCase):
    def test_smoke_and_production(self):
        for reps, steps, count in [((1,), 2, 12), (range(1, 6), 10, 60)]:
            with self.subTest(count=count):
                memory, phases = coverage(reps, steps)
                self.assertEqual(len(memory), count)
                validate_campaign_coverage(memory, phases, reps, steps)

    def test_bad_memory_coverage(self):
        memory, phases = coverage()
        for rows in [memory[:-1], memory + [memory[0]],
                     [dict(r, rep='2') for r in memory],
                     [dict(memory[0], model='unknown')] + memory[1:]]:
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, 'Memory/latency coverage'):
                    validate_campaign_coverage(rows, phases, (1,), 2)

    def test_missing_production_profile(self):
        memory, phases = coverage(range(1, 6), 10)
        phases = [r for r in phases if not (r['model'] == 'giant' and
                  r['config'] == 'AIx' and r['rep'] == '5')]
        with self.assertRaisesRegex(ValueError, 'Score-P coverage'):
            validate_campaign_coverage(memory, phases, range(1, 6), 10)

    def test_missing_application_or_dl_scope(self):
        memory, phases = coverage()
        for view in ('application', 'provider_dl'):
            with self.subTest(view=view):
                rows = [r for r in phases if not (r['model'] == 'watercnn' and
                        r['config'] == 'PHYDLL_CPP' and r['view'] == view)]
                with self.assertRaisesRegex(ValueError, 'Missing Score-P scope'):
                    validate_campaign_coverage(memory, rows, (1,), 2)

    def test_bad_rank_and_step_counts(self):
        memory, phases = coverage()
        for index, field, value in [(0, 'rank_count', '95'), (3, 'rank_count', '95'),
                                    (0, 'steady_steps', '10'), (3, 'steady_steps', '1')]:
            with self.subTest(index=index, field=field):
                rows = copy.deepcopy(phases)
                rows[index][field] = value
                with self.assertRaisesRegex(ValueError, 'Invalid Score-P ranks or steady steps'):
                    validate_campaign_coverage(memory, rows, (1,), 2)
        for field in ('solver_ranks', 'steady_steps'):
            with self.subTest(field=field):
                rows = copy.deepcopy(memory)
                rows[0][field] = '1'
                with self.assertRaisesRegex(ValueError, 'Invalid memory/latency ranks or steady steps'):
                    validate_campaign_coverage(rows, phases, (1,), 2)

    def test_unexpected_scope_and_run(self):
        memory, phases = coverage()
        for field, value in [('scope', 'solver_steady'), ('view', 'application'), ('rep', '2')]:
            with self.subTest(field=field):
                rows = copy.deepcopy(phases)
                rows[3][field] = value
                with self.assertRaisesRegex(ValueError, 'Unexpected Score-P'):
                    validate_campaign_coverage(memory, rows, (1,), 2)

    def test_duplicate_phase_or_multiple_profiles(self):
        memory, phases = coverage()
        with self.assertRaisesRegex(ValueError, 'Duplicate Score-P phase'):
            validate_campaign_coverage(memory, phases + [phases[0]], (1,), 2)
        rows = copy.deepcopy(phases)
        rows[1]['cubex'] = '/synthetic/other.cubex'
        with self.assertRaisesRegex(ValueError, 'Multiple Score-P profiles'):
            validate_campaign_coverage(memory, rows, (1,), 2)

    def test_profile_reuse(self):
        memory, phases = coverage()
        rows = [dict(r, cubex='/synthetic/reused.cubex') for r in phases]
        with self.assertRaisesRegex(ValueError, 'profile reused'):
            validate_campaign_coverage(memory, rows, (1,), 2)


class BatchValidation(unittest.TestCase):
    def test_batch_inline_checks_reject_missing_dl_with_optimization(self):
        for filename, variable, reps, steps in [
                ('validate_submit_terrain_nobatch.sbatch', 'TERRAIN_GATE_FIGURES', (1,), 2),
                ('analyze_terrain_nobatch.sbatch', 'TERRAIN_ANALYSIS_FIGURES', range(1, 6), 10)]:
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as tmp:
                script = (MINI_APP / filename).read_text()
                inline = script.split(f'export {variable}=', 1)[1].split("<<'PY'\n", 1)[1].split('\nPY', 1)[0]
                memory, phases = coverage(reps, steps)
                phases = [r for r in phases if r['view'] != 'provider_dl']
                for relative, rows in [('memory_latency/memory_latency_replicates.csv', memory),
                                       ('phases/phase_replicates.csv', phases)]:
                    path = Path(tmp) / relative
                    path.parent.mkdir(parents=True)
                    with path.open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                env = dict(os.environ, **{variable: tmp})
                result = subprocess.run([sys.executable, '-B', '-O', '-c', inline], env=env,
                                        cwd=MINI_APP, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Missing Score-P scope', result.stderr)

    def test_submission_environment_and_dependency(self):
        # Execute only the submission tail with a fake sbatch first on PATH.
        script = (MINI_APP / 'validate_submit_terrain_nobatch.sbatch').read_text()
        tail = script[script.index('# Remove inherited campaign/runtime'):]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sbatch = root / 'sbatch'
            sbatch.write_text('''#!/usr/bin/python3
import json, os, sys
with open(os.environ['TEST_SUBMISSIONS'], 'a') as stream:
    stream.write(json.dumps({'args': sys.argv[1:], 'env': dict(os.environ)}) + '\\n')
print('12345;testcluster')
''')
            sbatch.chmod(0o755)
            env = dict(os.environ, PATH=f'{root}:{os.environ["PATH"]}',
                       TEST_SUBMISSIONS=str(root / 'submissions.jsonl'),
                       MINI_APP_DIR=str(MINI_APP), FIGURES=tmp, SLURM_JOB_ID='synthetic',
                       ANALYSIS_PYTHON=str(MINI_APP.parent / 'CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3'))
            dirty = {'MODEL': 'giant', 'REP': '99', 'RUN_ROOT': '/wrong', 'LOG_ROOT': '/wrong',
                     'OUTPUT_BASE': '/wrong', 'CAMPAIGN_ROOT': '/wrong', 'CONFIGS': 'AIX',
                     'MODELS': 'giant', 'REPS': '99', 'STEPS': '6', 'MEMORY_INTERVAL': '.2',
                     'MPI_RANKS': '1', 'RANK_GRID_X': '1', 'RANK_GRID_Z': '1',
                     'SOLVER_BIN': '/wrong', 'SMARTSIM_PYTHON': '/wrong',
                     'MONITOR_PYTHON': '/wrong', 'PHYDLL_DL_CLIENT': '/wrong',
                     'RUN_TIMEOUT': '1', 'DB_START_TIMEOUT': '1', 'SSDB': 'wrong',
                     'SBATCH_ARRAY_INX': '0-1', 'SBATCH_DEPENDENCY': 'afterok:999',
                     'SMARTSIM_DB_NODELIST': 'wrong', 'MLCOUPLING_INTRA_OP_THREADS': '99',
                     'PHYDLL_DL_COUNT': '99', 'OMP_NUM_THREADS': '99', 'MKL_NUM_THREADS': '99',
                     'OPENBLAS_NUM_THREADS': '99', 'KMP_AFFINITY': 'wrong', 'GOMP_CPU_AFFINITY': '0',
                     'SCOREP_FILTERING_FILE': '/wrong',
                     'OMPI_MCA_btl': 'wrong', 'PMIX_MCA_gds': 'wrong'}
            env.update(dirty)
            result = subprocess.run(['bash', '-euo', 'pipefail', '-c', tail], env=env,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            submissions = [json.loads(line) for line in (root / 'submissions.jsonl').read_text().splitlines()]
            self.assertEqual(len(submissions), 2)
            self.assertIn('--array=0-9%2', submissions[0]['args'])
            self.assertIn('--dependency=afterok:12345', submissions[1]['args'])
            for submitted in submissions:
                self.assertIn('--export=ALL', submitted['args'])
                actual = submitted['env']
                for name in dirty:
                    self.assertNotEqual(actual.get(name), dirty[name], name)
                self.assertEqual(actual['STEPS'], '22')
                self.assertEqual(actual['MPI_RANKS'], '96')
                self.assertEqual(actual['RANK_GRID_X'], '12')
                self.assertEqual(actual['RANK_GRID_Z'], '8')
                self.assertEqual(actual['REPS'], '5')
                self.assertEqual(actual['CONFIGS'].split(), ['AIX', 'PHYDLL_CPP', 'SS_DEFAULT',
                                 'SS_TPQ12_I8_B96', 'SS_TPQ96_I1_B96', 'SS_TPQ1_I96_B96'])
                self.assertEqual(actual['MEMORY_INTERVAL'], '1.0')
                self.assertEqual(actual['SMARTSIM_PYTHON'], env['ANALYSIS_PYTHON'])
                self.assertEqual(actual['MONITOR_PYTHON'], env['ANALYSIS_PYTHON'])
                self.assertEqual(actual['OUTPUT_BASE'], str(MINI_APP / 'results/terrain_nobatch/validated_synthetic'))
            self.assertNotIn('CAMPAIGN_ROOT', submissions[0]['env'])
            self.assertEqual(submissions[1]['env']['CAMPAIGN_ROOT'], submissions[0]['env']['OUTPUT_BASE'])

    def test_source_without_nounset_and_missing_roots_fail(self):
        for filename in ('validate_submit_terrain_nobatch.sbatch', 'analyze_terrain_nobatch.sbatch'):
            with self.subTest(filename=filename):
                script = (MINI_APP / filename).read_text()
                setup = script[script.index('unset PYTHONHOME'):script.index('ANALYSIS_PYTHON=')]
                setup = setup.replace('source /rwthfs/rz/cluster/hpcwork/ro092286/smartsim/set_env_claix23_cuda12.4.sh',
                                      '[[ "$-" != *u* ]] || exit 42; unset EBROOTOPENSSL; export LD_LIBRARY_PATH=/synthetic')
                result = subprocess.run(['bash', '-euo', 'pipefail', '-c', setup],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 1)
                self.assertIn('Missing module root: EBROOTOPENSSL', result.stderr)


if __name__ == '__main__':
    unittest.main()
