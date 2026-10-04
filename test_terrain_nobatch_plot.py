"""Synthetic sampler-schema/campaign fixtures only; no jobs or memory workloads."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from extract_scorep_tables import ORDER, discover
from plot_terrain_nobatch import collect, validate_memory
from terrain_memory_sampler import LIFECYCLE_ERRORS, ROLES


def fixture(root):
    job = root / 'job_fixture'
    logs = job / 'logs'
    logs.mkdir(parents=True)
    aliases = {'AIx': 'AIX', 'TPQ1_I96': 'SS_TPQ1_I96_B96',
               'TPQ12_I8': 'SS_TPQ12_I8_B96', 'TPQ96_I1': 'SS_TPQ96_I1_B96'}
    for model in ('watercnn', 'giant'):
        rep = job / model / 'rep1'
        rep.mkdir(parents=True)
        results = []
        for ci, cfg in enumerate(ORDER):
            name = aliases.get(cfg, cfg)
            directory = rep / name
            directory.mkdir()
            (directory / 'scorep').mkdir()
            (directory / 'scorep/profile.cubex').touch()
            expected = {'solver': 1, 'dl': int(cfg == 'PHYDLL_CPP'), 'db': int(cfg not in ('AIx', 'PHYDLL_CPP'))}
            counts = {**expected, 'support': 1}
            peaks = {role: {'rss_bytes': counts[role] * (ci + 2) * 2**20,
                            'pss_bytes': counts[role] * (ci + 1) * 2**20,
                            'max_processes': counts[role]} for role in ROLES}
            peaks['total'] = {k: sum(r[k] for r in peaks.values()) for k in ('rss_bytes', 'pss_bytes', 'max_processes')}
            samples = []
            for sample in range(2):
                for role in (*ROLES, 'total'):
                    p = peaks[role]
                    samples.append({'sample': sample, 'role': role, 'processes': p['max_processes'],
                                    'rss_read': p['max_processes'], 'pss_read': p['max_processes'],
                                    'rss_bytes': p['rss_bytes'], 'pss_bytes': p['pss_bytes'], 'sweep_s': .05,
                                    'direct_solver': expected['solver'], 'direct_dl': expected['dl'],
                                    'direct_db': expected['db'], 'sample_interval_s': .5 if sample else ''})
            with (directory / 'memory_samples.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
                writer.writeheader()
                writer.writerows(samples)
            for role in ROLES:
                if not counts[role]:
                    peaks[role]['rss_bytes'] = peaks[role]['pss_bytes'] = None
            summary = {'coverage_ok': True, 'expected_processes': expected, 'max_direct_processes': expected,
                       'samples': 2, 'simultaneous_coverage_samples': 2, 'interval_s': .5, 'max_sweep_s': .05,
                       'process_read_errors': 0, 'cleanup_sigkill_pids': [], 'host': 'synthetic-fixture-not-measured',
                       'lifecycle_read_races': 0, 'live_read_errors': 0, 'read_error_counts': {},
                       'actual_intervals_s': {'count': 1, 'min': .5, 'mean': .5, 'median': .5, 'p95': .5, 'max': .5},
                       'overrun_samples': 0, 'max_cycle_s': .05,
                       'token': f'{model}_{cfg}', 'sampled_simultaneous_peaks': peaks}
            (directory / 'memory_summary.json').write_text(json.dumps(summary))
            results.append({'model': model, 'rep': 1, 'config': name, 'status': 'SUCCESS', 'cubex': 'profile.cubex'})
            (logs / f'terrain_nobatch_{model}_r1_{name}_fixture.log').write_text(''.join(
                f'STEP_TIMING step={step} solver=ML step_ms={100 + ci + step}\n' for step in range(2, 24, 2)))
        with (rep / 'results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    return job


class CampaignPlotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.job = fixture(self.root)
        self.directory = self.job / 'watercnn/rep1/AIX'

    def test_schema_and_new_layout(self):
        rows = collect([self.root], [], None, 1, 10)
        self.assertEqual(len(rows), 12)
        self.assertEqual({r['config'] for r in rows}, set(ORDER))
        self.assertEqual(rows[0]['steady_steps'], 10)
        self.assertEqual(rows[0]['steady_mean_ms'], 113)
        self.assertEqual(len(json.loads(rows[0]['steady_samples_ms'])), 10)
        runs = discover([self.job], None)
        self.assertEqual(len(runs), 12)
        self.assertTrue(all(Path(r['cubex']).is_file() for r in runs))

    def test_bad_json_fails(self):
        path = self.directory / 'memory_summary.json'
        original = json.loads(path.read_text())
        for key, value in [('coverage_ok', False), ('max_sweep_s', .6), ('live_read_errors', 1),
                           ('cleanup_sigkill_pids', [99]), ('expected_processes', {'solver': 0, 'dl': 0, 'db': 0})]:
            with self.subTest(key=key):
                path.write_text(json.dumps({**original, key: value}))
                with self.assertRaises(ValueError):
                    validate_memory(self.directory, 'AIx', 1)

    def test_missing_pss_and_wrong_peak_fail(self):
        path = self.directory / 'memory_summary.json'
        summary = json.loads(path.read_text())
        summary['sampled_simultaneous_peaks']['total']['pss_bytes'] = None
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'Missing/invalid'):
            validate_memory(self.directory, 'AIx', 1)
        summary['sampled_simultaneous_peaks']['total']['pss_bytes'] = 1
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'peak mismatch'):
            validate_memory(self.directory, 'AIx', 1)

    def test_incomplete_reads_fail(self):
        path = self.directory / 'memory_samples.csv'
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        rows[0]['pss_read'] = 0
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, 'incomplete metric'):
            validate_memory(self.directory, 'AIx', 1)

    def test_lifecycle_races_exclude_incomplete_samples(self):
        path = self.directory / 'memory_samples.csv'
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for r in rows:
            if r['sample'] == '1' and r['role'] in ('solver', 'total'):
                for metric, read in (('rss_bytes', 'rss_read'), ('pss_bytes', 'pss_read')):
                    r[metric] = ''
                    r[read] = int(r[read]) - 1
            elif r['sample'] == '1' and r['role'] == 'support':
                # A complete role can peak in a sweep whose total is incomplete.
                r['rss_bytes'] = 4 * 2**20
                r['pss_bytes'] = 2 * 2**20
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary_path = self.directory / 'memory_summary.json'
        summary = json.loads(summary_path.read_text())
        summary['sampled_simultaneous_peaks']['support']['rss_bytes'] = 4 * 2**20
        summary['sampled_simultaneous_peaks']['support']['pss_bytes'] = 2 * 2**20
        for error in LIFECYCLE_ERRORS:
            with self.subTest(error=error):
                summary.update(lifecycle_read_races=1, process_read_errors=1, read_error_counts={error: 1})
                summary_path.write_text(json.dumps(summary))
                result = validate_memory(self.directory, 'AIx', 1)
                self.assertEqual(result['excluded_total_rss_samples'], 1)
                self.assertEqual(result['excluded_total_pss_samples'], 1)
                self.assertEqual(result['rss_peak_bytes'], 4 * 2**20)
                self.assertEqual(result['lifecycle_read_races'], 1)

    def test_unexplained_incomplete_reads_fail(self):
        path = self.directory / 'memory_samples.csv'
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for r in rows:
            if r['sample'] == '1' and r['role'] in ('solver', 'total'):
                r['rss_bytes'] = r['pss_bytes'] = ''
                r['rss_read'] = int(r['rss_read']) - 1
                r['pss_read'] = int(r['pss_read']) - 1
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, 'incomplete reads do not match lifecycle races'):
            validate_memory(self.directory, 'AIx', 1)

    def test_actual_interval_metadata_mismatch_fails(self):
        path = self.directory / 'memory_summary.json'
        summary = json.loads(path.read_text())
        summary['actual_intervals_s']['mean'] = .7
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'actual interval statistics mismatch'):
            validate_memory(self.directory, 'AIx', 1)

    def test_overruns_warn_and_report_not_reject(self):
        path = self.directory / 'memory_samples.csv'
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for r in rows:
            r['sweep_s'] = .7
            if r['sample'] == '1':
                r['sample_interval_s'] = .8
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary_path = self.directory / 'memory_summary.json'
        summary = json.loads(summary_path.read_text())
        summary.update(max_sweep_s=.7, max_cycle_s=.75, overrun_samples=2,
                       actual_intervals_s={'count': 1, 'min': .8, 'mean': .8, 'median': .8, 'p95': .8, 'max': .8})
        summary_path.write_text(json.dumps(summary))
        with self.assertWarnsRegex(UserWarning, 'Sampling resolution warning'):
            result = validate_memory(self.directory, 'AIx', 1)
        self.assertEqual(result['overrun_samples'], 2)
        self.assertEqual(result['max_cycle_s'], .75)
        self.assertIn('0.8', result['actual_intervals_s'])

    def test_missing_latency_step_fails(self):
        log = self.job / 'logs/terrain_nobatch_watercnn_r1_AIX_fixture.log'
        log.write_text('STEP_TIMING step=4 solver=ML step_ms=100\n')
        with self.assertRaisesRegex(ValueError, 'incomplete/unexpected steady'):
            collect([self.job], [], None, 1, 10)

    def test_descendants_cannot_mask_missing_direct_solver_at_raw_peak(self):
        path = self.directory / 'memory_samples.csv'
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        # Two solver-role descendants still cannot stand in for a missing direct solver.
        for r in rows:
            if r['role'] == 'solver':
                r['processes'] = r['rss_read'] = r['pss_read'] = 2
            elif r['role'] == 'total':
                r['processes'] = r['rss_read'] = r['pss_read'] = 3
            if r['sample'] == '1':
                r['direct_solver'] = 0
                r['rss_bytes'] = int(r['rss_bytes']) * 2
                r['pss_bytes'] = int(r['pss_bytes']) * 2
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary_path = self.directory / 'memory_summary.json'
        summary = json.loads(summary_path.read_text())
        for role, p in summary['sampled_simultaneous_peaks'].items():
            for metric in ('rss_bytes', 'pss_bytes'):
                if p[metric] is not None:
                    p[metric] *= 2
        summary['sampled_simultaneous_peaks']['solver']['max_processes'] = 2
        summary['sampled_simultaneous_peaks']['total']['max_processes'] = 3
        summary['simultaneous_coverage_samples'] = 1
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, 'raw lifetime.*outside exact expected direct census; refusing to lower peak'):
            validate_memory(self.directory, 'AIx', 1)

    def test_missing_config_coverage_fails(self):
        path = self.job / 'watercnn/rep1/results.csv'
        with path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows[:-1])
        with self.assertRaisesRegex(ValueError, 'matching replicate coverage'):
            collect([self.job], [], None, 1, 10)

    def test_cli_renders_fixture_both_model_panels(self):
        output = self.root / 'synthetic_figures'
        command = [sys.executable, str(Path(__file__).with_name('plot_terrain_nobatch.py')),
                   '--campaign-root', str(self.job), '--solver-ranks', '1', '--output-dir', str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ('external_memory_peaks', 'steady_latency_boxplots', 'steady_latency'):
            for suffix in ('png', 'pdf'):
                self.assertTrue((output / f'{name}.{suffix}').is_file())
        with (output / 'steady_latency_steps.csv').open(newline='') as stream:
            steps = list(csv.DictReader(stream))
        self.assertEqual(len(steps), 120)
        self.assertNotIn('2', {r['step'] for r in steps})
        self.assertEqual({int(r['step']) for r in steps}, set(range(4, 24, 2)))
        print('Synthetic-only CLI verification: 12 config replicates, 120 steady-step samples; memory/boxplot/mean PNG/PDF rendered')


if __name__ == '__main__':
    unittest.main()
