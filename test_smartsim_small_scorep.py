"""Isolated four-case harness; all submissions use fake sbatch and swatch."""
import csv
import os
from pathlib import Path
import shlex
import shutil
import unittest

import test_scorep_pilot_harness as legacy


class SmartSimSmallTests(legacy.PilotHarnessTests):
    def setUp(self):
        super().setUp()
        shutil.copy2(legacy.MINI / 'submit_smartsim_small_scorep.sh', self.mini)

    def small_fixture(self, archive=True, empty=False, driver_rc=0):
        self.build_fixture()
        driver = '''#!/bin/zsh
set -eu
[[ $MODEL_NAME_ENV == benchmark_giant_mlp && $TARGET_WIDTH_ENV == 480 && $TARGET_HEIGHT_ENV == 288 ]]
[[ $CHUNK_SIZE_ENV == 16 && $ML_BATCH_SIZE_ENV == 500000 && $DB_NODES_ENV == 4 ]]
[[ $SMARTSIM_THREADS_PER_QUEUE_ENV == $PILOT_TPQ_ENV && $SMARTSIM_THREADS_PER_QUEUE == $PILOT_TPQ_ENV ]]
[[ $SCOREP_ENABLE_PROFILING_ENV == true && $SCOREP_ENABLE_PROFILING == true ]]
if [[ $PILOT_MODE_ENV == trace ]]; then
  [[ $TOTAL_STEPS_ENV == 8 && $SAVE_EVERY_ENV == 8 && $PILOT_TPQ_ENV == 1 ]]
  [[ $SCOREP_ENABLE_TRACING_ENV == true && $SCOREP_ENABLE_TRACING == true ]]
else
  [[ $TOTAL_STEPS_ENV == 12 && $SAVE_EVERY_ENV == 12 && $PILOT_TPQ_ENV == 2 ]]
  [[ $SCOREP_ENABLE_TRACING_ENV == false && $SCOREP_ENABLE_TRACING == false ]]
fi
base=${0:A:h}/scorep_runs/${SCOREP_DIR_TAG_ENV}_${RUN_ID_ENV}
mkdir -p "$base/traces"
print profile > "$base/profile.cubex"
'''
        if archive:
            driver += (': > "$base/traces.otf2"\n: > "$base/traces/0.evt"\n' if empty else
                       'print anchor > "$base/traces.otf2"\nprint event > "$base/traces/0.evt"\n')
        driver += f'exit {driver_rc}\n'
        for name in ('5r_driver_single.zsh', '5r_driver_hetsurrogate.zsh'):
            (self.mini / name).write_text(driver)

    def small_env(self, cfg, mode, jid='900001'):
        env = dict(self.env, PILOT_MODE_ENV=mode, PILOT_TPQ_ENV='2' if mode == 'profile' else '1',
                   SLURM_JOB_ID=jid, SLURM_NTASKS='96')
        if 'p' in cfg:
            env.update(SLURM_HET_SIZE='2', SLURM_NTASKS_HET_GROUP_0='96',
                       SLURM_NTASKS_HET_GROUP_1='96', SLURM_JOB_NUM_NODES_HET_GROUP_0='1',
                       SLURM_JOB_NUM_NODES_HET_GROUP_1='1')
        return env

    def test_small_runtime_matrix_and_metadata(self):
        self.small_fixture()
        for index, (cfg, mode) in enumerate((c, m) for c in ('96c4g', '96p96c4g')
                                            for m in ('profile', 'trace')):
            env = self.small_env(cfg, mode, str(900001 + index))
            result = self.run_script('retest_scorep_pilot.zsh', cfg, 'giantmlp288', 'SMARTSIM', 'pergpu', env=env)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        manifests = list((self.mini / 'results/scorep_pilots/test22').glob('*/manifest.csv'))
        self.assertEqual(len(manifests), 4)
        for manifest in manifests:
            with manifest.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertIn(f"_{row['mode']}_tpq{row['tpq']}_", row['run_id'])
            self.assertEqual(row['steps'], '8' if row['mode'] == 'trace' else '12')
            self.assertEqual(row['steady'], '3' if row['mode'] == 'trace' else '5')
            self.assertEqual(row['rc'], '0')
            self.assertTrue(all(Path(p).stat().st_size > 0 for p in Path(row['traces_file']).read_text().splitlines()))

    def test_trace_missing_or_empty_archive_fails(self):
        for archive, empty in ((False, False), (True, True)):
            with self.subTest(archive=archive, empty=empty):
                # Each fixture uses its own job path; build_fixture may reuse directories only once.
                if not (self.mini / 'solver_cpp').exists():
                    self.small_fixture(archive=archive, empty=empty)
                else:
                    for driver in self.mini.glob('5r_driver_*.zsh'):
                        text = driver.read_text().replace('exit 0\n', ': > "$base/traces.otf2"\n: > "$base/traces/0.evt"\nexit 0\n')
                        driver.write_text(text)
                env = self.small_env('96c4g', 'trace', '910001' if not archive else '910002')
                result = self.run_script('retest_scorep_pilot.zsh', '96c4g', 'giantmlp288', 'SMARTSIM', 'pergpu', env=env)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Required nonempty archived OTF2', result.stderr)
                self.assertIn('rc=1', result.stdout)

    def test_invalid_overrides_rejected(self):
        for mode, tpq in (('', '1'), ('invalid', '1'), ('profile', ''), ('profile', '0'),
                          ('profile', '3'), ('trace', '2')):
            result = self.run_script('retest_scorep_pilot.zsh', '--describe', '96c4g', 'giantmlp288',
                                     'SMARTSIM', 'pergpu', env=dict(self.env, PILOT_MODE_ENV=mode, PILOT_TPQ_ENV=tpq))
            self.assertNotEqual(result.returncode, 0)
        result = self.run_script('retest_scorep_pilot.zsh', '--describe', '24c1g', 'watercnn8k', 'PHYDLL',
                                 'default', env=dict(self.env, PILOT_MODE_ENV='trace', PILOT_TPQ_ENV='1'))
        self.assertNotEqual(result.returncode, 0)

    def test_exact_four_dry_run_and_submission(self):
        self.build_fixture()
        result = self.run_script('submit_smartsim_small_scorep.sh', '--dry-run', '123456', 'smartsim_small_test')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith('sbatch ')]
        self.assertEqual(len(commands), 4)
        self.assertFalse((self.mini / 'results/scorep_pilots').exists())
        expected = {(cfg, mode, tpq) for cfg in ('96c4g', '96p96c4g') for mode, tpq in (('profile', '2'), ('trace', '1'))}
        actual = set()
        for cmd in commands:
            self.assertFalse(any(arg.startswith('--dependency') for arg in cmd))
            exports = dict(item.split('=', 1) for item in next(a for a in cmd if a.startswith('--export=')).removeprefix('--export=').split(','))
            cfg = Path(cmd[-4]).stem.removeprefix('scorep_pilot_')
            actual.add((cfg, exports['PILOT_MODE_ENV'], exports['PILOT_TPQ_ENV']))
            self.assertEqual(cmd[-3:], ['giantmlp288', 'SMARTSIM', 'pergpu'])
        self.assertEqual(actual, expected)
        tools = self.root / 'bin'
        tools.mkdir()
        home = self.root / 'home'
        (home / 'scripts').mkdir(parents=True)
        sbatch = tools / 'sbatch'
        sbatch.write_text('#!/bin/bash\nprintf "warning\\n" >&2\nprintf "987654;cluster\\n"\n')
        sbatch.chmod(0o755)
        swatch = home / 'scripts/swatch'
        swatch.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$HOME/watch.log"\n')
        swatch.chmod(0o755)
        env = dict(self.env, HOME=str(home), PATH=f'{tools}:{os.environ["PATH"]}')
        result = self.run_script('submit_smartsim_small_scorep.sh', '--submit', '123456', 'smartsim_small_test', env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.mini / 'results/scorep_pilots/smartsim_small_test/submissions.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertEqual((home / 'watch.log').read_text().splitlines(), ['--jobid ' + ' '.join(['987654'] * 4)])
        again = self.run_script('submit_smartsim_small_scorep.sh', '--submit', '123456', 'smartsim_small_test', env=env)
        self.assertNotEqual(again.returncode, 0)


if __name__ == '__main__':
    unittest.main()
