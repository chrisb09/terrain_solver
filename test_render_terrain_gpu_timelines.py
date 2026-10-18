import copy
import json
from pathlib import Path
import re
import subprocess
import unittest
from unittest.mock import patch
import render_terrain_gpu_timelines as t


class TimelineTests(unittest.TestCase):
    def test_controller_span_and_missing_peers(self):
        rows = []
        def add(event, time, peer=-1, first=-1, stop=-1):
            rows.append(dict(event=event, time_s=100+time, peer_workgroup_rank=peer,
                             range_first_rank=first, range_end_rank=stop, sample_start=0, sample_count=10))
        add('ml_step_start', 0)
        add('input_listen_start', 1)
        add('input_credit_send', 2, 1)
        add('input_ready', 3, 0)
        add('input_ready', 4, 1)
        for family in ('range_inference', 'torch_h2d', 'torch_forward', 'torch_d2h'):
            add(family+'_start', 5, first=0, stop=2)
            add(family+'_end', 6, first=0, stop=2)
        add('result_send_start', 7, 0)
        add('result_send_start', 8, 1)
        add('result_send_complete', 10, 1)
        add('ml_step_end', 12)
        value = t.controller_metrics(list(reversed(rows)), 2)
        self.assertEqual(value['exchange_ms'], 9000)
        self.assertEqual(value['inference_end_s']-value['inference_start_s'], 1)
        for event in ('input_listen_start', 'input_ready', 'result_send_complete', 'torch_h2d_end'):
            bad = copy.deepcopy(rows)
            bad.pop(next(i for i, r in enumerate(bad) if r['event'] == event))
            with self.subTest(event=event), self.assertRaises(ValueError):
                t.controller_metrics(bad, 2)
        with self.assertRaises(ValueError):
            t.controller_metrics(rows+[rows[2]], 2)

    def test_pipeline_synthetic(self):
        self.assertEqual(t.pipeline([50, 30, 200, 20, 80], 10), 218)
        self.assertEqual(sum([50, 30, 200, 20, 80]), 380)
        groups = [[50, 30, 200, 20, 80], [0, 0, 300, 0, 0]]
        self.assertEqual(max(range(2), key=lambda i: sum(groups[i])), 0)
        self.assertEqual(max(range(2), key=lambda i: t.pipeline(groups[i], 10)), 1)
        for stages, n in (([1, 2], 10), ([1]*5, 0), ([-1]*5, 10)):
            with self.assertRaises(ValueError):
                t.pipeline(stages, n)

    def test_mapping_nonmodulo(self):
        text = ('Rank 2/3 will be in group 0 order 0\n'
                'Rank 0/3 will be in group 0 order 7\n'
                'Rank 1/3 will be in group 0 order 3\n'
                'Rank 2/3 on its local machine is 0/1 is the GPU controller!\n'
                'Rank 2 of 3 uses GPU index 0\n')
        self.assertEqual(t.mapping(text), {0: [2, 1, 0]})
        with self.assertRaises(ValueError):
            t.mapping(text.replace('Rank 0/3 will', 'Rank 1/3 will'))
        with self.assertRaises(ValueError):
            t.mapping(text.replace('order 7', 'order 3'))

    def test_unsorted_range_pairing_and_missing(self):
        def row(event, time):
            return dict(event=event, time_s=time, range_first_rank=0, range_end_rank=2,
                        sample_start=0, sample_count=10)
        rows = [row('torch_h2d_end', 5), row('torch_h2d_start', 3)]
        self.assertEqual(t.pair_ranges(rows, 'torch_h2d'), [(3, 5, (0, 2, 0, 10))])
        for bad in (rows[:1], rows+rows, [row('torch_h2d_start', 6), rows[0]]):
            with self.assertRaises(ValueError):
                t.pair_ranges(bad, 'torch_h2d')

    def test_real_selection(self):
        runs, selected = t.discover()
        self.assertEqual(len(runs), 12)
        self.assertEqual(len(selected), 8)
        for (hw, mode), run in selected.items():
            self.assertEqual(run['model'], 'watercnn8k')
            self.assertEqual(run['layout'] if 'layout' in run else 'default', 'default')
            self.assertEqual(run['campaign'], t.CAMPAIGNS[0] if mode == 'p2p1' and hw.startswith('24') else t.CAMPAIGNS[1])

    def test_real_exports_and_scopes(self):
        path = t.OUT / 'tables/timeline_workgroup_calls.csv'
        if not path.exists():
            self.skipTest('Run gallery first')
        rows = t.read_csv(path)
        self.assertEqual(len(rows), 100)
        for row in rows:
            self.assertIn(int(row['call']), range(1, 6))
            if row['variant'] == 'default':
                self.assertEqual(row['pipeline_prediction_ms'], '')
                self.assertEqual(row['forward_cuda_ms'], '')
            else:
                self.assertAlmostEqual(float(row['exchange_ms']),
                    (float(row['exchange_end_s'])-float(row['exchange_start_s']))*1000)
        self.assertEqual(len(t.read_csv(t.OUT / 'tables/timeline_trace_exports.csv')), 12)

    def test_common_interval_coverage_and_clocks(self):
        ledger_path = t.OUT / 'tables/timeline_trace_exports.csv'
        if not ledger_path.exists():
            self.skipTest('Run gallery first')
        ledger = t.read_csv(ledger_path)
        aix = [r for r in ledger if r['provider'] == 'AIX']
        self.assertEqual({(r['hardware'], r['variant']) for r in aix},
                         {(hw, mode) for hw in t.HARDWARE for mode in ('default', 'p2p1')})
        for row in ledger:
            metadata = json.loads(Path(row['csv']).with_suffix('.json').read_text())
            self.assertGreater(metadata['clock'][0], 0)
            self.assertEqual(metadata['csv_sha256'], t.sha(row['csv']))
            self.assertEqual(int(row['steady_calls']), 5*int(row['solver_locations']))
            self.assertEqual(int(row['warmup_calls']), int(row['solver_locations']))
            calls = {r['ml_call_index'] for r in t.read_csv(row['csv']) if r['region'] == t.otf.STEADY}
            self.assertEqual(calls, {'2', '3', '4', '5', '6'})
        coverage = t.read_csv(t.OUT / 'tables/timeline_common_coverage.csv')
        self.assertEqual(len(coverage), 24)
        self.assertEqual(sum(r['status'] == 'exported' for r in coverage), 12)

    def test_rank_local_audit_and_maxima(self):
        if not (t.OUT / 'tables/timeline_clock_audit.csv').exists():
            self.skipTest('Run gallery first')
        audit = t.read_csv(t.OUT / 'tables/timeline_clock_audit.csv')
        self.assertEqual(len(audit), 1750)
        for row in audit:
            self.assertEqual(row['applied_offset_s'], '')
            low = max(float(row['controller_credit_s'])-float(row['worker_send_s']),
                      float(row['controller_ready_s'])-float(row['worker_complete_s']),
                      float(row['controller_post_s'])-float(row['worker_received_s']))
            self.assertEqual(float(row['causal_lower_s']), low)
            self.assertAlmostEqual(float(row['hypothetical_ack_output_ms']),
                (float(row['worker_received_s'])+float(row['hypothetical_ack_offset_s'])-float(row['controller_post_s']))*1000)
        metrics = t.read_csv(t.OUT / 'tables/timeline_workgroup_calls.csv')
        calls = t.read_csv(t.OUT / 'aix_pipeline_spans/call_values.csv')
        self.assertEqual(len(calls), 20)
        for row in calls:
            self.assertEqual(row['pipeline_prediction_ms'], '')
            self.assertEqual(row['fullcredit_exchange_ms'], '')
            selected = [r for r in metrics if r['variant'] == 'p2p1' and
                        (r['hardware'], r['call']) == (row['hardware'], row['call'])]
            self.assertEqual(float(row['p2p_controller_exchange_ms']), max(float(r['exchange_ms']) for r in selected))

    def test_real_renderer_contracts(self):
        import sys
        sys.path.insert(0, str(Path.home() / 'Master-Thesis/gitlab/sourcecode'))
        import thesis_style as ts
        ts.apply()
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        renderer = t.load_module(t.RENDERER, 'test_original_terrain')
        sys.path.insert(0, str(t.MMCP / 'scripts'))
        from plot_cmi_aix_terrain_style import stagger_range_labels
        _, selected = t.discover()
        run = selected['96p96c4g', 'p2p1']
        groups = t.mapping((Path(run['run_dir']) / 'driver.log').read_text())
        members = groups[0]
        directory = t.ROOT / 'logs' / ('aix_p2p_timeline_' + run['run_id'])
        rows = renderer.read_events(directory)
        refs = [{k: float(v) if k == 'end_ms' else int(v) if k in ('workgroup', 'call') else v
                 for k, v in r.items()} for r in t.read_csv(t.OUT / 'tables/timeline_collective_intervals.csv')]
        captured = []
        def inspect(fig, path, style):
            self.assertEqual(len(fig.axes), 1)
            self.assertFalse(fig.texts)
            self.assertAlmostEqual(fig.get_figwidth()*25.4, 150)
            ax = fig.axes[0]
            self.assertIn('independent rank-local', ax.get_xlabel())
            self.assertEqual(len(ax.get_yticklabels()), 48)
            self.assertTrue(all(v.get_fontsize() == 5 for v in ax.get_yticklabels()))
            for ref in refs:
                if ref['hardware'] == '96p96c4g' and ref['workgroup'] == 0 and ref['call'] == 5:
                    self.assertLess(ref['end_ms'], ax.get_xlim()[1])
            fig.canvas.draw()
            for legend in fig.legends:
                box = legend.get_window_extent(fig.canvas.get_renderer())
                self.assertGreaterEqual(box.x0, 0)
                self.assertLessEqual(box.x1, fig.bbox.x1)
                self.assertGreater(box.y0, ax.bbox.y1)
            captured.append(path)
            plt.close(fig)
        with patch.object(t, 'save_figure', side_effect=inspect), \
             patch.object(renderer, 'synchronize_step_clocks', side_effect=AssertionError('No ACK alignment')):
            # render_detail overrides the original synchronizer with an identity.
            t.render_detail('96p96c4g', 0, members, rows, refs, t.OUT / 'aix_timelines', ts, plt,
                            renderer, stagger_range_labels)
        self.assertEqual(len(captured), 1)

    def test_all_rendered_formats_fonts_and_dimensions(self):
        metadata = json.loads((t.OUT / 'aix_timelines/metadata.json').read_text())
        self.assertEqual(len(metadata['figures']), 14)
        from PIL import Image
        for name in metadata['figures']:
            base = t.OUT / 'aix_timelines' / name
            for suffix in ('.png', '.pdf', '.pgf'):
                self.assertGreater(base.with_suffix(suffix).stat().st_size, 1000)
            info = subprocess.check_output(['pdfinfo', str(base.with_suffix('.pdf'))], text=True)
            width = float(re.search(r'Page size:\s+([\d.]+)', info)[1])
            self.assertAlmostEqual(width*25.4/72, 150, places=2)
            fonts = subprocess.check_output(['pdffonts', str(base.with_suffix('.pdf'))], text=True)
            self.assertIn('Type 1', fonts)
            self.assertNotIn('Type 3', fonts)
            with Image.open(base.with_suffix('.png')) as image:
                self.assertLess(abs(image.width-150/25.4*300), 2)
        for path, digest in metadata['numeric_sha256'].items():
            self.assertEqual(t.sha(t.OUT / path), digest)

    def test_source_provenance_unchanged(self):
        metadata = json.loads((t.OUT / 'aix_timelines/metadata.json').read_text())
        self.assertTrue(metadata['source_integrity_verified'])
        for path, digest in metadata['sources'].items():
            self.assertEqual(t.sha(path), digest, path)


if __name__ == '__main__':
    unittest.main()
