"""Offline contract tests for the final runner. Never submit jobs or load modules."""
import csv
import hashlib
import json
from itertools import product
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
RUNNER = HERE / 'terrain_cpu_final.sbatch'
TEXT = RUNNER.read_text()
CONFIGS = ('AIx', 'PHYDLL_CPP', 'SS_DEFAULT', 'SS_TPQ1_I96_B96',
           'SS_TPQ12_I8_B96', 'SS_TPQ96_I1_B96')


class FinalRunnerTests(unittest.TestCase):
    def test_bash_syntax(self):
        subprocess.run(['bash', '-n', str(RUNNER)], check=True)
        self.assertIn('#SBATCH --array=0-9%2\n', TEXT)

    def test_bootstrap_environment_and_module_roots(self):
        bootstrap = TEXT.split('unset PYTHONHOME', 1)[1].split('export LD_LIBRARY_PATH=', 1)[0]
        bootstrap = 'unset PYTHONHOME' + bootstrap
        bootstrap = bootstrap.replace('source "$BASE_DIR/set_env_claix23_cuda12.4.sh"', ':')
        names = ('EBROOTOPENSSL', 'EBROOTCLANG', 'EBROOTGCCCORE', 'EBROOTOPENMPI',
                 'EBROOTHDF5', 'EBROOTPNETCDF', 'EBROOTCUDA', 'EBROOTCUDNN', 'EBROOTIMKL', 'EBROOTHWLOC')
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, **dict.fromkeys(names, directory), 'LD_LIBRARY_PATH': directory,
                   'PYTHONHOME': '/invalid', 'PYTHONPATH': '/invalid', 'PYTHONOPTIMIZE': '2',
                   'LD_PRELOAD': '', 'OMP_NUM_THREADS': '17'}
            checks = '\n[[ ! -v PYTHONHOME && ! -v PYTHONPATH && ! -v PYTHONOPTIMIZE && ! -v LD_PRELOAD && $OMP_NUM_THREADS == 17 ]]'
            result = subprocess.run(['bash', '-ec', bootstrap + checks], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for value in ('', directory + '/missing'):
                result = subprocess.run(['bash', '-ec', bootstrap], env={**env, 'EBROOTOPENSSL': value},
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Missing module root: EBROOTOPENSSL', result.stderr)

    def plan(self, **settings):
        env = {**os.environ, 'PLAN_ONLY': '1'}
        for name in ('MODE', 'CONFIGS', 'STEPS', 'SLURM_ARRAY_TASK_ID'):
            env.pop(name, None)
        env.update(settings)
        return subprocess.run(['bash', str(RUNNER)], env=env, capture_output=True, text=True)

    def test_full_mapping(self):
        for mode in ('performance', 'memory'):
            for task in range(10):
                result = self.plan(MODE=mode, SLURM_ARRAY_TASK_ID=str(task))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('model=' + ('watercnn' if task < 5 else 'giant'), result.stdout)
                self.assertIn(f'rep={task % 5 + 1}', result.stdout)
                self.assertIn('steps=' + ('22' if mode == 'performance' else '6'), result.stdout)
                expected = CONFIGS if mode == 'performance' else (CONFIGS[3], CONFIGS[4], CONFIGS[1])
                self.assertIn('configs=' + ' '.join(expected), result.stdout)

    def test_subsets_and_bad_inputs(self):
        self.assertEqual(self.plan(STEPS='6', CONFIGS=' '.join(CONFIGS)).returncode, 0)
        self.assertEqual(self.plan(MODE='memory', CONFIGS='PHYDLL_CPP').returncode, 0)
        for settings in ({'MODE': 'unknown'}, {'CONFIGS': 'AIX'}, {'CONFIGS': 'AIx AIx'},
                         {'STEPS': '5'}, {'STEPS': '7'}, {'SLURM_ARRAY_TASK_ID': '10'}):
            self.assertNotEqual(self.plan(**settings).returncode, 0)

    def test_effective_policies(self):
        # Execute the actual shell policy block, without the filesystem/launch sections.
        block = TEXT.split('    case $config in\n', 1)[1].split('    export SCOREP_EXPERIMENT_DIRECTORY', 1)[0]
        block = 'case $config in\n' + block
        for total in (8294400, 46080):
            for cfg in CONFIGS:
                code = (f'config={cfg}; total_bs={total}; MODEL=watercnn; MINI_APP_DIR=/mini; '
                        'SLURM_JOB_ID=123; task=0; dir=/run; tpq=0; intra=0; db_batch=0; '
                        'omit=0; expected_dl=0; expected_db=0; extra=();\n' + block +
                        '\nprintf "%s %s %s %s %s" "$tpq" "$intra" "$db_batch" "$omit" "$expected_dl"')
                result = subprocess.run(['bash', '-c', code], capture_output=True, text=True, check=True)
                expected = {
                    'AIx': (0, 0, 0, 0, 0), 'PHYDLL_CPP': (0, 1, 0, 0, 96),
                    'SS_DEFAULT': (1, 1, 0, 1, 0),
                    'SS_TPQ1_I96_B96': (1, 96, total, 0, 0),
                    'SS_TPQ12_I8_B96': (12, 8, total // 12, 0, 0),
                    'SS_TPQ96_I1_B96': (96, 1, 0, 0, 0),
                }[cfg]
                self.assertEqual(tuple(map(int, result.stdout.split())), expected)

    def test_record_precision_and_memory_exclusion(self):
        script = TEXT.split('record() {', 1)[1].split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        for mode, config in product(('performance', 'memory'), CONFIGS):
            with self.subTest(mode=mode, config=config), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                directory = root / config
                (directory / 'scorep').mkdir(parents=True)
                (directory / 'scorep/profile.cubex').touch()
                provenance = root / 'provenance.json'
                provenance.write_text(json.dumps({'device': 'CPU', 'mode': mode}))
                run_environment = directory / 'run.env'
                run_environment.write_text('PHYDLL_IO_LOG=0\nPHYDLL_IO_LOG_BARRIERS=0\n')
                digest = hashlib.sha256(run_environment.read_bytes()).hexdigest()
                (directory / 'environment.sha256').write_text(f'{digest}  {run_environment}\n')
                (directory / 'solver.log').write_text(''.join(
                    f'STEP_TIMING step={step} solver=ML step_ms=0.123456789123456789 '
                    'rank_min_ms=0.111111111111111111 rank_max_ms=0.222222222222222222\n'
                    for step in (2, 4, 6)))
                env = {**os.environ, 'PYTHONPATH': str(HERE), 'MODE': mode, 'MODEL': 'giant',
                       'REP': '5', 'STEPS': '6', 'RUN_ROOT': str(root), 'ROOT': str(root)}
                tpq, intra, batch, omit = {
                    'AIx': ('0', '0', '0', '0'), 'PHYDLL_CPP': ('0', '1', '0', '0'),
                    'SS_DEFAULT': ('1', '1', '0', '1'),
                    'SS_TPQ1_I96_B96': ('1', '96', '46080', '0'),
                    'SS_TPQ12_I8_B96': ('12', '8', '3840', '0'),
                    'SS_TPQ96_I1_B96': ('96', '1', '0', '0'),
                }[config]
                subprocess.run([sys.executable, '-c', script, str(directory), 'SUCCESS',
                                '123', tpq, intra, batch, omit], env=env, check=True)
                with (root / 'results.csv').open() as stream:
                    row = next(csv.DictReader(stream))
                self.assertEqual(row['steady_steps'], '2')
                self.assertEqual(row['monitoring'], str(int(mode == 'memory')))
                self.assertEqual(row['mean_step_ms'], '0.123456789123456789' if mode == 'performance' else '')
                self.assertEqual(row['step4_ms'], '0.123456789123456789' if mode == 'performance' else '')
                self.assertEqual(row['rank_min_ms'], '0.111111111111111111' if mode == 'performance' else '')
                self.assertEqual(row['provenance'], str(provenance.resolve()))
                self.assertEqual(row['run_environment'], str(run_environment.resolve()))
                self.assertEqual(row['environment_sha256'], str((directory / 'environment.sha256').resolve()))
                self.assertEqual(row['run_environment_sha256'], digest)
                self.assertEqual(row['inter'], '0' if config == 'AIx' else '1')
                self.assertEqual(row['thread_settings'], 'unforced' if config == 'AIx' else
                                 'omitted_nominal' if config == 'SS_DEFAULT' else 'explicit')
                self.assertEqual(row['memory_sampler'], str(int(mode == 'memory')))
                self.assertEqual(row['phydll_logging'], '0')
                self.assertEqual(row['internal_memory'], '0')

    def test_preflight_pin_and_defaults(self):
        script = TEXT.split('"$HELPER_PYTHON" - "$ROOT" "$prep"', 1)[1].split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / 'libphydll.so'
            library.write_bytes(b'fixture PHYDLL_IO_LOG\0 PHYDLL_IO_LOG_BARRIERS\0')
            source = root / 'extern/phydll/src/core/io.c'
            source.parent.mkdir(parents=True)
            source.write_text('\n'.join(
                f'static bool {name}(void) {{ return value != NULL && strcmp(value, "1") == 0; }}'
                for name in ('io_log_enabled', 'io_log_barriers_enabled')))
            for name in ('solver', 'dl'):
                (root / f'{name}_libraries.txt').write_text(f' libphydll.so => {library} (0x123)\n')
            for name in ('solver', 'dl', 'prep.h5', 'terrain_cpu_final.sbatch', 'terrain_cpu_diagnostics.py',
                         'terrain_memory_sampler.py', 'smartsim_controller.py', 'pin_wrapper.sh', 'config.toml',
                         'config_aix.toml', 'config_phydll.toml', 'train_models/model_a/giant_cpu.pt',
                         'train_models/model_a/giant_split_cpu.pt'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            env = {**os.environ, 'CPP_DIR': str(root), 'MINI_APP_DIR': str(root), 'MODEL': 'giant',
                   'MODE': 'performance',
                   'SOLVER_BIN': str(root / 'solver'), 'PHYDLL_DL_CLIENT': str(root / 'dl'),
                   'APPROVED_PHYDLL_SHA256': hashlib.sha256(library.read_bytes()).hexdigest()}
            def run():
                return subprocess.run([sys.executable, '-c', script, str(root), 'prep.h5'],
                                      env=env, capture_output=True, text=True)
            self.assertEqual(run().returncode, 0)
            self.assertTrue((root / 'provenance.json').is_file())
            provenance = json.loads((root / 'provenance.json').read_text())
            for name, value in {'device': 'CPU', 'mode': 'performance', 'memory_sampler': 0,
                                'phydll_logging': 0, 'internal_memory': 0, 'timing_sync': 1,
                                'db_min_batch': 0, 'db_timeout': 0}.items():
                self.assertEqual(provenance[name], value)
            approved = env['APPROVED_PHYDLL_SHA256']
            env['APPROVED_PHYDLL_SHA256'] = '0' * 64
            self.assertNotEqual(run().returncode, 0)
            env['APPROVED_PHYDLL_SHA256'] = approved
            source.write_text(source.read_text().replace('value != NULL &&', 'value == NULL ||'))
            self.assertNotEqual(run().returncode, 0)

    def test_lifecycle_and_launch_contract(self):
        cleanup = TEXT.split('cleanup() {', 1)[1].split('\nrecord()', 1)[0]
        self.assertLess(cleanup.index('wait "$ctrl_pid"'), cleanup.index('touch "$dir/memory.stop"'))
        launches = TEXT.split('for config in $CONFIGS; do', 2)[2]
        self.assertLess(launches.index('terrain_memory_sampler.py'), launches.index('controller=('))
        self.assertIn('if [[ $MODE == memory ]]; then', launches)
        self.assertIn('command+=(: "${exports[@]}" "${extra[@]}" -n 96', launches)
        self.assertIn('-x PHYDLL_IO_LOG -x PHYDLL_IO_LOG_BARRIERS', TEXT)
        self.assertIn('--bind-to none --report-bindings --oversubscribe', TEXT)
        self.assertIn('--ml-db-min-batch-size 0 --ml-db-min-batch-timeout 0', TEXT)
        self.assertIn('--validate-log "$dir/solver.log" --steps "$STEPS" < /dev/null || rc=1', TEXT)
        self.assertIn('(( rc == 0 )) || exit "$rc"', TEXT)
        self.assertNotIn('pkill', TEXT)
        self.assertNotIn('export OMP_NUM_THREADS=', TEXT)
        self.assertIn('APPROVED_PHYDLL_SHA256', TEXT)


if __name__ == '__main__':
    unittest.main()
