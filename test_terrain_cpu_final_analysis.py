"""Synthetic fixtures only; these tests do not produce campaign measurements."""
import json
import hashlib
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import terrain_cpu_final_analysis as analysis


class FinalAnalysisTests(unittest.TestCase):
    def setUp(self):
        # Synthetic artifact approval exists only inside this test process.
        approval = patch.object(analysis, 'APPROVED_PHYDLL_SHA256', hashlib.sha256(b'synthetic library').hexdigest())
        approval.start()
        self.addCleanup(approval.stop)

    def attest_env(self, directory, environment):
        path = directory / 'run.env'
        path.write_text(''.join(f'{k}={v}\n' for k, v in sorted(environment.items())))
        (directory / 'environment.sha256').write_text(f'{analysis.source(path)["sha256"]}  {path}\n')

    def fixture(self, root, mode='performance', configs=None):
        runner = (analysis.HERE / 'terrain_cpu_final.sbatch').read_text()
        record = runner.split('record() {', 1)[1].split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        record = compile(record, 'terrain_cpu_final.sbatch:record', 'exec')
        names = {'TPQ1_I96': 'SS_TPQ1_I96_B96', 'TPQ12_I8': 'SS_TPQ12_I8_B96', 'TPQ96_I1': 'SS_TPQ96_I1_B96'}
        for model in analysis.MODELS:
            job = root / f'job_{model}'
            directory = job / model / 'rep1'
            directory.mkdir(parents=True)
            library = job / 'libphydll.so'
            library.write_bytes(b'synthetic library')
            inputs = {str(library.resolve()): analysis.source(library)['sha256']}
            for name in (f'{model}_cpu.pt', f'{model}_split_cpu.pt', 'config.toml', 'config_aix.toml',
                          'config_phydll.toml', 'terrain_cpu_final.sbatch', 'pin_wrapper.sh',
                          'prep_watercnn_4k.h5' if model == 'watercnn' else 'prep_giant_240x192.h5'):
                path = job / name
                path.write_text('synthetic input; not a measured output')
                inputs[str(path.resolve())] = analysis.source(path)['sha256']
            provenance = dict(inputs=inputs, phydll_default_logging=0, phydll_default_barriers=0,
                              approved_phydll_sha256=analysis.APPROVED_PHYDLL_SHA256,
                              device='CPU', mode=mode, memory_sampler=int(mode == 'memory'),
                              phydll_logging=0, phydll_io_log=0, phydll_io_log_barriers=0, internal_memory=0,
                              internal_mem_verbose=0, internal_mem_collective=0, telemetry=0, timing_sync=1,
                              solver_ranks=96, rank_grid='12x8', db_min_batch=0, db_timeout=0,
                              policy='final_total_per_queue_except_default_and_TPQ96',
                              helper_python=sys.executable, environment={'MODE': mode, 'MODEL': model})
            (job / 'provenance.json').write_text(json.dumps(provenance))
            for name in ('solver', 'dl'):
                (job / f'{name}_libraries.txt').write_text(f'libphydll.so => {library} (0x0)\n')
            for cfg in configs or analysis.ORDER:
                run = directory / names.get(cfg, cfg)
                (run / 'scorep').mkdir(parents=True)
                log = run / 'solver.log'
                log.write_text(''.join(f'STEP_TIMING step={s} solver=ML step_ms={s} rank_min_ms=1 rank_max_ms=10\n' for s in (2, 4, 6)))
                profile = run / 'scorep/profile.cubex'
                profile.write_text('synthetic profile placeholder; never extracted')
                total = analysis.TOTAL_BS[model]
                tpq, intra, omit = analysis.THREADS[cfg]
                batch = total if cfg == 'TPQ1_I96' else total // 12 if cfg == 'TPQ12_I8' else 0
                env = {**analysis.RUNTIME, 'MODE': mode, 'MODEL': model, 'REP': '1', 'STEPS': '6',
                       'ROOT': str(job), 'RUN_ROOT': str(directory),
                       'APPROVED_PHYDLL_SHA256': analysis.APPROVED_PHYDLL_SHA256,
                       'SCOREP_EXPERIMENT_DIRECTORY': str(run / 'scorep')}
                if cfg == 'PHYDLL_CPP':
                    env.update(MLCOUPLING_INTRA_OP_THREADS='1', MLCOUPLING_INTER_OP_THREADS='1',
                               PHYDLL_DL_COUNT='1', PHYDLL_DL_FIELD_COUNT='1')
                smart = cfg.startswith('TPQ') or cfg == 'SS_DEFAULT'
                if smart:
                    env.update(SMARTSIM_OMIT_THREADS=str(omit), SMARTSIM_OMIT_SET_CPUS=str(omit),
                               SMARTSIM_THREADS_PER_QUEUE=str(tpq), SMARTSIM_INTRA_OP_THREADS=str(intra),
                                SMARTSIM_INTER_OP_THREADS='1')
                if mode == 'memory':
                    env['TERRAIN_MEMORY_TOKEN'] = str(run)
                self.attest_env(run, env)
                pin = (analysis.HERE / 'pin_wrapper.sh').resolve()
                command = ['mpirun']
                for _ in range(2 if cfg == 'PHYDLL_CPP' else 1):
                    for key in analysis.RUNTIME:
                        command += ['-x', key]
                    if cfg == 'PHYDLL_CPP':
                        for key in ('MLCOUPLING_INTRA_OP_THREADS', 'MLCOUPLING_INTER_OP_THREADS', 'PHYDLL_DL_COUNT', 'PHYDLL_DL_FIELD_COUNT'):
                            command += ['-x', key]
                    command += ['-n', '96']
                    if cfg in ('AIx', 'PHYDLL_CPP'):
                        command += [str(pin)]
                command += ['--device', 'CPU', '--gpus-per-node', '0', '--rank-grid-x', '12', '--rank-grid-z', '8',
                            '--steps', '6', '--ml-batch-size', '10000000', '--model-backend', 'TORCH',
                            '--model-io-layout', 'split_3x3' if smart else 'flat_contiguous',
                            '--model-path', str(job / f'{model}{"_split" if smart else ""}_cpu.pt'),
                            '--input-hdf5', str(job / ('prep_watercnn_4k.h5' if model == 'watercnn' else 'prep_giant_240x192.h5')),
                            '--cpp-ml-config', str(job / ('config_phydll.toml' if cfg == 'PHYDLL_CPP' else 'config_aix.toml' if cfg == 'AIx' else 'config.toml'))]
                if smart:
                    command += ['--ml-db-batch-size', str(batch), '--ml-db-min-batch-size', '0', '--ml-db-min-batch-timeout', '0']
                (run / 'command.sh').write_text(shlex.join(command))
                # Execute the actual runner writer, but never its shell/launch code.
                with patch.dict(os.environ, env), patch.object(sys, 'argv',
                        ['record', str(run), 'SUCCESS', '123', str(tpq), str(intra), str(batch), str(omit)]):
                    exec(record, {})

    def test_performance_without_memory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            with patch.object(analysis, 'validate_memory', side_effect=AssertionError('must not read memory')):
                rows = analysis.collect([root], 'performance', 1, 2)
            self.assertEqual(len(rows), 12)
            self.assertEqual(rows[0]['steady_mean_ms'], 5)
            self.assertEqual(set(json.loads(rows[0]['steady_samples_ms'])), {'4', '6'})
            phases = []
            for row in rows:
                for view, scope in [('application', 'solver_steady'),
                                    ('provider_dl', 'dl_steady_count_validated') if row['config'] == 'PHYDLL_CPP'
                                    else ('provider', 'solver_steady')]:
                    phases.append({**{k: row[k] for k in ('model', 'config', 'rep', 'cubex')},
                                   'view': view, 'scope': scope, 'phase': 'synthetic',
                                   'rank_count': '96', 'steady_steps': '2'})
            analysis.validate_campaign_coverage(rows, phases, [1], 2)
            with self.assertRaisesRegex(ValueError, 'Coverage'):
                analysis.collect([root], 'performance', 5, 2)
            with self.assertRaisesRegex(ValueError, 'mode'):
                analysis.collect([root], 'memory', 1, 2)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                analysis.collect([root], 'performance', 1, 2, expected_provenance={'device': 'CPU'})

    def test_step_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            log = root / 'job_watercnn/watercnn/rep1/AIx/solver.log'
            log.write_text(log.read_text() + 'STEP_TIMING step=4 solver=ML step_ms=3\n')
            with self.assertRaisesRegex(ValueError, 'duplicate step'):
                analysis.collect([root], 'performance', 1, 2)

    def test_policy(self):
        for cfg, batch in [('TPQ1_I96', '0'), ('TPQ12_I8', '120'), ('TPQ96_I1', '0.0'), ('SS_DEFAULT', '1')]:
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                analysis.validate_policy({'total_bs': '120', 'db_batch': batch}, cfg)

    def test_actual_record_and_dataset_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = Path(temp) / 'results/terrain_cpu_final/smoke'
            mode_root = dataset / 'performance'
            self.fixture(mode_root)
            rows = analysis.collect([dataset], 'performance', 1, 2)
            self.assertEqual(len(rows), 12)
            self.assertEqual(len(analysis.collect([mode_root], 'performance', 1, 2)), 12)
            for model in analysis.MODELS:
                group = {r['config']: r for r in rows if r['model'] == model}
                self.assertEqual(group['AIx']['inter'], '0')
                self.assertEqual(group['AIx']['thread_settings'], 'unforced')
                self.assertEqual(group['SS_DEFAULT']['thread_settings'], 'omitted_nominal')
                self.assertEqual(group['SS_DEFAULT']['omit_threads'], '1')
                self.assertEqual(group['SS_DEFAULT']['tpq'], '1')
                self.assertEqual(group['TPQ12_I8']['total_bs'], str(analysis.TOTAL_BS[model]))
                self.assertEqual(group['TPQ12_I8']['db_batch'], str(analysis.TOTAL_BS[model] // 12))
                for row in group.values():
                    self.assertTrue(analysis.FIELDS <= row.keys())
                    self.assertEqual(row['run_environment_sha256'], row['run_env_source']['sha256'])
                    self.assertEqual(Path(row['provenance']), Path(row['provenance_source']['path']))

    def test_strict_metadata(self):
        for field in sorted(analysis.FIELDS - {'model', 'rep', 'config', 'status', 'cubex', 'log', 'mode', 'db_batch'}):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.fixture(root)
                manifest = root / 'job_watercnn/watercnn/rep1/results.csv'
                rows = analysis.read_csv(manifest)
                rows[0][field] = 'invalid'
                analysis.write_csv(manifest, rows)
                with self.assertRaises(ValueError):
                    analysis.collect([root], 'performance', 1, 2)

    def test_runtime_exports_and_attestation(self):
        for tamper in ('digest', 'logging', 'threads', 'monitoring', 'command', 'input', 'approval', 'missing'):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                self.fixture(root)
                job = root / 'job_watercnn'
                run = job / 'watercnn/rep1/SS_TPQ1_I96_B96'
                env = dict(line.split('=', 1) for line in (run / 'run.env').read_text().splitlines())
                if tamper == 'digest':
                    (run / 'run.env').write_text('tampered')
                elif tamper in ('logging', 'threads', 'monitoring'):
                    key = {'logging': 'PHYDLL_IO_LOG', 'threads': 'SMARTSIM_INTRA_OP_THREADS',
                           'monitoring': 'TERRAIN_MEMORY_TOKEN'}[tamper]
                    env[key] = 'wrong'
                    self.attest_env(run, env)
                    # Keep both attestations valid so this exercises semantic checks.
                    manifest = run.parent / 'results.csv'
                    rows = analysis.read_csv(manifest)
                    for row in rows:
                        if row['config'] == run.name:
                            row['run_environment_sha256'] = analysis.source(run / 'run.env')['sha256']
                    analysis.write_csv(manifest, rows)
                elif tamper == 'command':
                    path = run / 'command.sh'
                    path.write_text(path.read_text().replace('--ml-db-batch-size 8294400', '--ml-db-batch-size 0'))
                elif tamper == 'input':
                    (job / 'terrain_cpu_final.sbatch').write_text('changed runner after job start')
                elif tamper == 'missing':
                    (job / 'provenance.json').unlink()
                else:
                    path = job / 'provenance.json'
                    provenance = json.loads(path.read_text())
                    provenance['approved_phydll_sha256'] = '0' * 64
                    path.write_text(json.dumps(provenance))
                with self.assertRaises((ValueError, OSError)):
                    analysis.collect([root], 'performance', 1, 2)

    def test_render_temporary_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            rows = analysis.collect([root], 'performance', 1, 2)
            figures = []
            with patch('hh3_render.close_figure', side_effect=figures.append):
                analysis.render(rows, root)
                for row in rows:
                    row.update(pss_peak_bytes=2**30, rss_peak_bytes=2 * 2**30)
                analysis.render(rows, root, memory=True)
            from hh3_render import close_figure
            self.assertEqual(figures[0].axes[0].get_yscale(), 'log')
            self.assertTrue(any('n=1 independent runs' in text.get_text() for text in figures[0].texts))
            self.assertFalse(figures[1].texts)
            for fig in figures:
                self.assertEqual([ax.get_title() for ax in fig.axes], ['[TS] WaterCNN', '[TS] Giant MLP'])
                for ax in fig.axes:
                    self.assertFalse(any('n=' in text.get_text() for text in ax.get_xticklabels()))
                    self.assertTrue(ax.texts)
                    self.assertFalse(any('e+' in text.get_text() for text in ax.texts))
                close_figure(fig)
            self.assertTrue((root / 'latency_boxplots.png').is_file())
            self.assertTrue((root / 'latency_boxplots.pdf').is_file())

    def test_memory_maxima_and_bars(self):
        # Role peaks deliberately exceed the simultaneous totals and must be ignored.
        rows = [{'model': model, 'config': cfg, 'rep': str(rep), 'dataset': 'synthetic',
                 'pss_peak_bytes': pss * 2**30, 'rss_peak_bytes': rss * 2**30,
                 'solver_peak_bytes': 100 * 2**30, 'dl_peak_bytes': 100 * 2**30,
                 'memory_json': f'/fixture/{model}/{cfg}/{rep}/memory_summary.json'}
                for model in analysis.MODELS for cfg in analysis.ORDER
                for rep, (pss, rss) in enumerate(((1, 8), (5, 6), (3, 7), (2, 4), (4, 5)), 1)]
        summaries = analysis.memory_maxima(rows)
        self.assertEqual(len(summaries), 12)
        for summary in summaries:
            self.assertEqual(summary['rep_count'], 5)
            self.assertEqual(summary['max_pss_peak_bytes'], 5 * 2**30)
            self.assertEqual(summary['max_rss_peak_bytes'], 8 * 2**30)
            self.assertEqual(summary['max_pss_peak_bytes_reps'], '2')
            self.assertEqual(summary['max_rss_peak_bytes_reps'], '1')
        with tempfile.TemporaryDirectory() as temp, \
                patch('hh3_render.save_figure'), patch('matplotlib.figure.Figure.savefig'), \
                patch('hh3_render.close_figure') as close:
            analysis.render(rows, Path(temp), memory=True)
            fig = close.call_args.args[0]
            self.assertFalse(fig.texts)
            for ax in fig.axes:
                self.assertEqual([bar.get_height() for bar in ax.patches], [5, 8] * 6)
                self.assertEqual(ax.get_ylabel(), 'Maximum sampled peak (GiB)')
                self.assertEqual([text.get_text() for text in ax.texts], ['5.00', '8.00'] * 6)
                self.assertFalse(any(getattr(container, 'has_yerr', False) for container in ax.containers))
            from hh3_render import plt
            plt.close(fig)

    def test_memory_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root, mode='memory', configs=('TPQ1_I96', 'TPQ12_I8'))
            for model in analysis.MODELS:
                manifest = root / f'job_{model}' / model / 'rep1/results.csv'
                rows = analysis.read_csv(manifest)
                for row in rows:
                    directory = manifest.parent / row['config']
                    (directory / 'memory_summary.json').write_text('{}')
                    (directory / 'memory_samples.csv').write_text('synthetic mock audit input')
            def memory(directory, cfg, ranks):
                return {'token': str(directory), 'memory_json': str(directory / 'memory_summary.json'),
                        'pss_peak_bytes': 1, 'rss_peak_bytes': 2}
            with patch.object(analysis, 'validate_memory', side_effect=memory):
                rows = analysis.collect([root], 'memory', 1, 2)
            self.assertEqual(len(rows), 4)
            self.assertFalse(any('steady_mean_ms' in r for r in rows))


if __name__ == '__main__':
    unittest.main()
