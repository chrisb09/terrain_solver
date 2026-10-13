"""Focused tests with fake builds/drivers/Slurm; never submits a real job."""
import csv
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


MINI = Path(__file__).resolve().parent
TMP = Path('/hpcwork/ro092286/smartsim/CPP-ML-Interface/tmp/opencode')


class PilotHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pilot-harness-', dir=TMP)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.mini = self.root / 'mini_app'
        self.mini.mkdir()
        for pattern in ('retest_scorep_pilot.zsh', 'submit_scorep_pilots.sh',
                        'scorep_pilot_*.sbatch', 'config*.toml'):
            for source in MINI.glob(pattern):
                shutil.copy2(source, self.mini / source.name)
        (self.mini / 'logs').mkdir()
        (self.mini / 'results').mkdir()
        self.env = dict(os.environ, PILOT_CAMPAIGN_ENV='test22',
                        PILOT_BUILD_JOB_ID_ENV='123456', SLURM_JOB_ID='654321',
                        SLURM_NTASKS='24', SLURM_JOB_NUM_NODES='1')

    def run_script(self, script, *args, env=None):
        shell = 'bash' if script.endswith('.sh') else 'zsh'
        return subprocess.run([shell, str(self.mini / script), *args],
                              env=env or self.env, text=True, capture_output=True)

    def cases(self):
        result = self.run_script('retest_scorep_pilot.zsh', '--list')
        self.assertEqual(result.returncode, 0, result.stderr)
        return [tuple(line.split()) for line in result.stdout.splitlines()]

    def build_fixture(self, driver_rc=0):
        solver = self.mini / 'solver_cpp/build_scorep_pilots'
        dl = self.root / 'CPP-ML-Interface/dl_clients/build-scorep-pilots'
        provenance = solver / 'provenance'
        provenance.mkdir(parents=True)
        dl.mkdir(parents=True)
        (solver / 'cpp-ml-interface-build').mkdir()
        for directory in (solver, dl):
            (directory / 'BUILD_READY').write_text('BUILD_READY job=123456 completed=now\n')
            (directory / 'CMakeCache.txt').write_text(
                'WITH_SCOREP:BOOL=ON\nPHYDLL_DL_WITH_TORCH:BOOL=ON\n'
                'CPPML_SCOREP_MPP:STRING=mpi\nUSE_CPP_ML_INTERFACE:BOOL=ON\n'
                'WITH_PHYDLL:BOOL=ON\nWITH_SMARTSIM:BOOL=ON\nWITH_AIX:BOOL=ON\n')
        binaries = [solver / 'terrain_solver', dl / 'phydll_dl_client',
                    dl / 'libphydll_dl_runtime.so']
        for binary in binaries:
            binary.write_text('#!/bin/sh\nexit 0\n')
            binary.chmod(0o755)
        (provenance / 'binaries.sha256').write_text(''.join(
            f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}\n' for p in binaries))
        (solver / 'build.env').write_text(
            f'SOLVER_BIN={shlex.quote(str(binaries[0]))}\n'
            f'PHYDLL_DL_CLIENT={shlex.quote(str(binaries[1]))}\n'
            f'USE_SCOREP=1\nSCOREP_MPP=mpi\nPROVENANCE={shlex.quote(str(provenance))}\n')
        for path in (solver / 'generated_registry.hpp',
                     solver / 'cpp-ml-interface-build/generated_registry.hpp'):
            path.write_text('solver_readiness_wait\n')
        (provenance / 'solver-strings.txt').write_text('PhyDLL readiness sequence mismatch\n')
        (provenance / 'dl-strings.txt').write_text('invalid solver_readiness_wait flag\n')
        for name in ('terrain_solver', 'phydll_dl_client'):
            (provenance / f'{name}.build-only.ldd.txt').write_text('libscorep_mpp_mpi\n')
        # Fixture drivers create the same discovery paths as the real drivers.
        driver = (
            '#!/bin/zsh\nset -eu\n'
            '[[ $TOTAL_STEPS_ENV == 12 && $USE_SCOREP_ENV == 1 && $SKIP_COMPILE_ENV == 1 ]]\n'
            '[[ $SCOREP_ENABLE_TRACING_ENV == false && $SCOREP_ENABLE_PROFILING_ENV == true ]]\n'
            '[[ $PHYDLL_DL_NODES_ENV == 1 && $SCOREP_SHARED_PROFILE_ENV == 1 && $GPU_MEM_SAMPLING_ENV == 0 ]]\n'
            '[[ $SMARTSIM_INTRA_OP_THREADS == 8 && $SMARTSIM_INTER_OP_THREADS == 1 ]]\n'
            'if [[ $CPP_ML_INTERFACE_PROVIDER_ENV == PHYDLL ]]; then [[ $MLCOUPLING_INTRA_OP_THREADS == 1 && $MLCOUPLING_INTER_OP_THREADS == 1 ]]; fi\n'
            f'base={shlex.quote(str(self.mini))}/scorep_runs/${{SCOREP_DIR_TAG_ENV}}_${{RUN_ID_ENV}}\n'
            'mkdir -p "${base}"\n'
            ': > "${base}/profile.cubex"\n'
            f'exit {driver_rc}\n')
        for name in ('5r_driver_single.zsh', '5r_driver_hetsurrogate.zsh'):
            (self.mini / name).write_text(driver)
        return solver, dl

    def test_exact_authorized_matrix_and_descriptions(self):
        configs = ('24c1g', '96c4g', '24p24c1g', '96p96c4g')
        expected = {(cfg, 'watercnn8k', 'PHYDLL', variant)
                    for cfg in configs for variant in ('default', 'readiness')}
        expected |= {(cfg, 'watercnn8k', 'SMARTSIM', 'shared') for cfg in configs}
        expected |= {(cfg, 'watercnn8k', 'SMARTSIM', 'pergpu') for cfg in configs if cfg.startswith('96')}
        expected |= {(cfg, 'watercnn8k', 'AIX', 'default') for cfg in ('24c1g', '96p96c4g')}
        expected |= {('24p24c1g', 'giantmlp288', 'PHYDLL', v) for v in ('default', 'readiness')}
        expected |= {(cfg, 'giantmlp288', 'SMARTSIM', 'shared') for cfg in configs if 'p' in cfg}
        expected |= {(cfg, 'giantmlp288', 'SMARTSIM', 'pergpu') for cfg in configs if cfg.startswith('96')}
        cases = self.cases()
        self.assertEqual(len(cases), 22)
        self.assertEqual(set(cases), expected)
        for case in cases:
            result = self.run_script('retest_scorep_pilot.zsh', '--describe', *case)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('steps=12 ml_calls=6 warm=1 steady=5', result.stdout)
            self.assertIn('dimensions=7680x4320 chunk=60 cap=20000000' if case[1] == 'watercnn8k'
                          else 'dimensions=480x288 chunk=16 cap=500000', result.stdout)
        self.assertFalse((self.mini / 'results/scorep_pilots').exists())
        rejected = self.run_script('retest_scorep_pilot.zsh', '--describe',
                                   '24c1g', 'giantmlp288', 'AIX', 'default')
        self.assertNotEqual(rejected.returncode, 0)

    def test_dry_run_is_read_only_and_dependent(self):
        result = self.run_script('submit_scorep_pilots.sh', '--dry-run', '123456', 'test22')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('sbatch ')]
        self.assertEqual(len(commands), 22)
        for cmd in commands:
            self.assertIn('--dependency=afterok:123456', cmd)
            self.assertIn(f'--chdir={self.mini}', cmd)
            self.assertIn('--export=PILOT_CAMPAIGN_ENV=test22,PILOT_BUILD_JOB_ID_ENV=123456', cmd)
        self.assertFalse((self.mini / 'results/scorep_pilots').exists())

    def test_templates_and_configs(self):
        for template in self.mini.glob('scorep_pilot_*.sbatch'):
            text = template.read_text()
            cfg = template.stem.removeprefix('scorep_pilot_')
            ranks, gpus = (96, 4) if cfg.startswith('96') else (24, 1)
            for component in text.split('#SBATCH hetjob'):
                for flag in ('--exclusive', '--mem=0', '--time=00:20:00',
                             f'--ntasks={ranks}', f'--gres=gpu:{gpus}'):
                    self.assertIn(f'#SBATCH {flag}\n', component)
            for option in ('output', 'error', 'chdir'):
                self.assertIn(f'#SBATCH --{option}=/rwthfs/', text)
        original = (MINI / 'config_phydll.toml').read_text()
        for variant, value in (('default', 'false'), ('readiness', 'true')):
            clone = (self.mini / f'config_phydll_pilot_{variant}.toml').read_text()
            self.assertEqual(clone.replace(f'solver_readiness_wait = {value}\n', ''), original)

    def test_completed_build_needs_no_dependency(self):
        self.build_fixture()
        result = self.run_script('submit_scorep_pilots.sh', '--dry-run', '123456', 'ready22')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('sbatch ')]
        self.assertEqual(len(commands), 22)
        for command in commands:
            self.assertFalse(any(arg.startswith('--dependency') for arg in command))
        self.assertFalse((self.mini / 'results/scorep_pilots').exists())

    def test_all_runtime_cases_and_profile_discovery(self):
        self.build_fixture()
        for index, case in enumerate(self.cases()):
            cfg, model, provider, variant = case
            ranks = '96' if cfg.startswith('96') else '24'
            env = dict(self.env, SLURM_JOB_ID=str(654321 + index), SLURM_NTASKS=ranks)
            if 'p' in cfg:
                env.update(SLURM_HET_SIZE='2', SLURM_NTASKS_HET_GROUP_0=ranks,
                           SLURM_NTASKS_HET_GROUP_1=ranks, SLURM_JOB_NUM_NODES_HET_GROUP_0='1',
                           SLURM_JOB_NUM_NODES_HET_GROUP_1='1')
            result = self.run_script('retest_scorep_pilot.zsh', *case, env=env)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        manifests = list((self.mini / 'results/scorep_pilots/test22').glob('*/manifest.csv'))
        self.assertEqual(len(manifests), 22)
        for manifest in manifests:
            with manifest.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row['rc'], '0')
            self.assertEqual(len(Path(row['profiles_file']).read_text().splitlines()), 1)
            if row['provider'] == 'PHYDLL':
                config = manifest.with_name('config.toml').read_text()
                cap = 20000000 if row['model'] == 'watercnn8k' else 500000
                self.assertIn(f'batch_size = {cap}\n', config)

    def test_failure_propagates_and_build_rebuild_guard(self):
        solver, _ = self.build_fixture(driver_rc=37)
        case = ('24c1g', 'watercnn8k', 'PHYDLL', 'default')
        result = self.run_script('retest_scorep_pilot.zsh', *case)
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertIn('rc=37', result.stdout)
        cache = solver / 'CMakeCache.txt'
        cache.write_text(cache.read_text().replace('WITH_PHYDLL:BOOL=ON', 'WITH_PHYDLL:BOOL=OFF'))
        env = dict(self.env, SLURM_JOB_ID='654322')
        result = self.run_script('retest_scorep_pilot.zsh', *case, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('would risk driver rebuild', result.stderr)
        outputs = list((self.mini / 'results/scorep_pilots/test22').glob('*654322'))
        self.assertFalse((outputs[0] / 'driver.log').exists())
        self.assertTrue((outputs[0] / 'manifest.csv').exists())

    def test_submission_stdout_and_swatch_with_mocks(self):
        # Only the copied harness sees these mocks; no real Slurm command runs.
        tools = self.root / 'bin'
        tools.mkdir()
        home = self.root / 'home'
        (home / 'scripts').mkdir(parents=True)
        sbatch = tools / 'sbatch'
        sbatch.write_text('#!/bin/bash\nprintf "warning on stderr\\n" >&2\nprintf "987654;cluster\\n"\n')
        sbatch.chmod(0o755)
        swatch = home / 'scripts/swatch'
        swatch.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$HOME/watch.log"\n')
        swatch.chmod(0o755)
        env = dict(self.env, HOME=str(home), PATH=f'{tools}:{os.environ["PATH"]}')
        result = self.run_script('submit_scorep_pilots.sh', '--submit', '123456', 'mock22', env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = self.mini / 'results/scorep_pilots/mock22/submissions.csv'
        with manifest.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 22)
        self.assertEqual({row['job_id'] for row in rows}, {'987654'})
        self.assertEqual((home / 'watch.log').read_text().splitlines(),
                         ['--jobid ' + ' '.join(['987654'] * 22)])
        again = self.run_script('submit_scorep_pilots.sh', '--submit', '123456', 'mock22', env=env)
        self.assertNotEqual(again.returncode, 0)

    def test_build_identity_readiness_integrity_and_allocation_guards(self):
        solver, dl = self.build_fixture()
        case = ('24c1g', 'watercnn8k', 'PHYDLL', 'readiness')
        checks = [
            (dl / 'BUILD_READY', 'BUILD_READY job=999 completed=now\n', 'Build markers differ'),
            (solver / 'BUILD_READY', 'BUILD_READY job=999 completed=now\n', 'Build markers differ'),
            (solver / 'generated_registry.hpp', 'old registry\n', 'Readiness registry missing'),
            (solver / 'provenance/dl-strings.txt', 'old binary\n', 'DL readiness implementation missing'),
            (solver / 'terrain_solver', 'changed binary\n', 'Pilot binaries changed since build'),
        ]
        for index, (path, changed, message) in enumerate(checks):
            original = path.read_text()
            path.write_text(changed)
            env = dict(self.env, SLURM_JOB_ID=str(700000 + index))
            result = self.run_script('retest_scorep_pilot.zsh', *case, env=env)
            path.write_text(original)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)
        env = dict(self.env, SLURM_JOB_ID='800000', SLURM_NTASKS='96')
        result = self.run_script('retest_scorep_pilot.zsh', *case, env=env)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('allocation/grid mismatch', result.stderr)


if __name__ == '__main__':
    unittest.main()
