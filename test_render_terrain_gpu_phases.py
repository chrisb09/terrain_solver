"""Synthetic scope tests and independent checks of the real exported gallery."""
from collections import defaultdict
import itertools
import unittest
from unittest.mock import Mock

import render_terrain_gpu_phases as plot
from analyze_cmi_scorep_profiles import CallTreeNode


def tree(spec):
    nodes = {}
    for i, (name, parent, seconds, visits) in enumerate(spec):
        node = CallTreeNode(i, name, parent)
        node.excl_time = seconds
        node.incl_time = seconds
        node.visits = visits
        nodes[i] = node
    return nodes


class ScopeTests(unittest.TestCase):
    def test_controller_scope_not_all_solver_ranks(self):
        controller = tree([('solver_step_ml_steady', None, 1, 5),
                           ('inferenceDevice', 0, .1, 5),
                           ('aix_controller_inference', 1, 2, 5),
                           ('torchInference::inference', 2, 8, 5),
                           ('gatherInputData', 0, 5, 5)])
        worker = tree([('solver_step_ml_steady', None, 1, 5),
                       ('inferenceDevice', 0, .001, 5),
                       ('gatherInputData', 0, 15, 5)])
        rows, _ = plot.phase_metrics([controller, worker], 'AIX', 5)
        phases = {r['phase']: r for r in rows}
        self.assertEqual(phases['Device (host)']['rank_count'], 1)
        self.assertEqual(phases['Device (host)']['mean_s'], 2)
        self.assertEqual(phases['Gather (host)']['rank_count'], 2)
        self.assertEqual(phases['Gather (host)']['mean_s'], 2)
        self.assertEqual(phases['Scatter (host)']['mean_s'], '')

    def test_nested_send_and_receive_never_double_count(self):
        nodes = tree([('phydll_dl_client', None, 9000, 1),
                      ('dl_send_output', 0, 2, 5), ('dl_send', 1, 3, 5),
                      ('MPI_Send', 2, 5, 5), ('dl_recv', 0, 1, 5),
                      ('dl_frame_copy', 4, 4, 5)])
        rows, ranks = plot.phase_metrics([nodes], 'PHYDLL', 5)
        phases = {r['phase']: r for r in rows}
        self.assertEqual(phases['Output send']['mean_s'], 2)
        self.assertEqual(phases['Output send']['raw_calls'], 5)
        self.assertEqual(phases['Input receive']['mean_s'], 1)
        self.assertEqual(sum(r['raw_exclusive_owned_s'] for r in ranks), 15)

    def test_dl_lifecycle_warmup_cannot_be_subtracted(self):
        nodes = tree([('phydll_dl_client', None, 1, 1), ('dl_torch_forward', 0, 12, 6)])
        with self.assertRaisesRegex(ValueError, 'steady visits'):
            plot.phase_metrics([nodes], 'PHYDLL', 5)

    def test_smartsim_multiple_puts_per_call_and_warmup_exclusion(self):
        nodes = tree([('solver_main_loop', None, 1000, 1),
                      ('solver_step_ml_warmup', 0, 10, 1),
                      ('smartsim_put_tensor', 1, 900, 2),
                      ('solver_step_ml_steady', 0, 10, 5),
                      ('smartsim_put_tensor', 3, 20, 10), ('MPI_Barrier', 4, 5, 10),
                      ('smartsim_run_model', 3, 10, 5)])
        rows, _ = plot.phase_metrics([nodes], 'shared', 5)
        phases = {r['phase']: r for r in rows}
        self.assertEqual(phases['Put']['mean_s'], 5)
        self.assertEqual(phases['Model request']['mean_s'], 2)
        self.assertEqual(phases['Get/unpack']['mean_s'], '')
        self.assertNotIn('CUDA', ' '.join(phases))

    def test_final_sibling_barrier_is_not_get_or_scatter(self):
        nodes = tree([('solver_step_ml_steady', None, 0, 5),
                      ('smartsim_library_static_step', 0, 0, 5),
                      ('smartsim_unpack_tensor', 1, .1, 5),
                      ('MPI_Barrier', 0, 100, 5)])
        rows, _ = plot.phase_metrics([nodes], 'shared', 5)
        self.assertEqual(next(r['mean_s'] for r in rows if r['phase'] == 'Get/unpack'), .02)
        self.assertNotIn('Scatter', ' '.join(r['phase'] for r in rows))

    def test_chunk_visits_normalize_per_step_not_per_request(self):
        nodes = tree([('solver_step_ml_steady', None, 0, 5),
                      ('smartsim_put_tensor', 0, 60, 30),
                      ('smartsim_run_model', 0, 30, 15),
                      ('smartsim_unpack_tensor', 0, 15, 15)])
        rows, _ = plot.phase_metrics([nodes], 'pergpu', 5)
        self.assertEqual([r['mean_s'] for r in rows[:3]], [12, 6, 3])


@unittest.skipUnless((plot.TABLES / 'plot_values.csv').is_file(), 'Run the real renderer first')
class GalleryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.values = plot.read_csv(plot.TABLES / 'plot_values.csv')
        cls.ranks = plot.read_csv(plot.TABLES / 'rank_metrics.csv')
        cls.selected = plot.read_csv(plot.TABLES / 'selected_profiles.csv')

    def test_all_32_unique_baselines(self):
        expected = set(itertools.product(plot.MODELS, plot.HARDWARE, plot.PROVIDERS))
        actual = [(r['model'], r['hardware'], r['provider']) for r in self.selected]
        self.assertEqual(len(actual), 32)
        self.assertEqual(set(actual), expected)
        self.assertTrue(all(r['steady_steps'] == '5' and r['warmup_calls'] == '1' for r in self.selected))
        self.assertEqual(sum(r['traced'] == 'True' for r in self.selected), 1)

    def test_independent_recomputation_of_every_value(self):
        grouped = defaultdict(list)
        for r in self.ranks:
            grouped[tuple(r[k] for k in ('model', 'hardware', 'provider', 'phase'))].append(r)
        for row in self.values:
            ranks = grouped[tuple(row[k] for k in ('model', 'hardware', 'provider', 'phase'))]
            self.assertEqual(len(ranks), int(row['rank_count']))
            self.assertEqual(int(row['denominator']), len(ranks) * int(row['steady_steps']))
            recomputed = sum(float(r['raw_exclusive_owned_s']) for r in ranks) / int(row['denominator'])
            self.assertAlmostEqual(recomputed, float(row['mean_s']), places=12)
            self.assertEqual(row['ci95_low_s'], '')
            self.assertEqual(row['ci95_high_s'], '')
            self.assertEqual(row['independent_captures'], '1')
            if row['provider'] == 'AIX' and row['phase'] == 'Device (host)':
                self.assertEqual(int(row['rank_count']), 4 if '4g' in row['hardware'] else 1)

    def test_real_smartsim_inclusive_anchors_and_no_barrier_leak(self):
        audit = plot.read_csv(plot.TABLES / 'smartsim_scope_audit.csv')
        self.assertEqual(len(audit), 64)
        for row in audit:
            self.assertAlmostEqual(float(row['inclusive_mean_s']), float(row['mean_s']), places=10)
            self.assertEqual(float(row['nested_barrier_s']), 0)
            self.assertEqual(int(row['anchor_rank_count']) * 5, int(row['denominator']))
            self.assertEqual(float(row['calls_per_rank_step']), 2 if row['phase'] == 'Put' else 1)
        diff = plot.read_csv(plot.TABLES / 'scope_audit_numeric_diff.csv')
        self.assertEqual(len(diff), 136)
        self.assertTrue(all(float(r['delta_s']) == 0 for r in diff))

    def test_adjacent_columns_bounds_and_no_titles(self):
        values = [dict(r, mean_s=float(r['mean_s'])) for r in self.values]
        for model, hw, include_pergpu in itertools.product(plot.MODELS, plot.HARDWARE, (True, False)):
            fig = plot.render(values, model, hw, Mock(), include_pergpu=include_pergpu)
            self.assertAlmostEqual(fig.get_figwidth() * 25.4, 150)
            self.assertAlmostEqual(fig.get_figheight() * 25.4, 60)
            for ax in fig.axes:
                self.assertEqual(ax.get_title(), '')
                self.assertAlmostEqual(ax.get_position().height * fig.get_figheight() * 25.4, 37.8)
                for left, right in zip(ax.patches, ax.patches[1:]):
                    self.assertGreater(right.get_x(), left.get_x() + left.get_width())
                self.assertTrue(all(bar.get_y() == 0 for bar in ax.patches))
                labels = [t.get_text() for t in ax.get_legend().get_texts()]
                if ax.get_xlabel() == 'AIx':
                    self.assertIn('Scatter/wait (host)', labels)
                if ax.get_xlabel().startswith('SmartSim'):
                    self.assertIn('Get/unpack (host)', labels)
                    self.assertIn('Model request (host)', labels)
                    self.assertFalse(any('Scatter' in label for label in labels))
                legend = ax.get_legend().get_window_extent(fig._get_renderer())
                self.assertGreater(legend.y0, ax.bbox.y1)
                self.assertLess((legend.y0 - ax.bbox.y1) / fig.dpi * 25.4, 3)
            self.assertEqual(len(fig.axes), 4 if include_pergpu else 3)
            self.assertEqual([ax.get_xlabel() for ax in fig.axes],
                             list(plot.LABELS) if include_pergpu else ['AIx', 'PhyDLL', 'SmartSim'])
            self.assertFalse(any('SS' in text.get_text() for text in fig.findobj()
                                 if hasattr(text, 'get_text')))
            self.assertFalse(fig.texts)

    def test_variant_preserves_shared_values_limits_and_output_names(self):
        values = [dict(r, mean_s=float(r['mean_s'])) for r in self.values]
        saver = Mock()
        for model, hw in itertools.product(plot.MODELS, plot.HARDWARE):
            full = plot.render(values, model, hw, saver)
            reduced = plot.render(values, model, hw, saver, include_pergpu=False)
            for original, omitted in zip(full.axes, reduced.axes):
                self.assertEqual(original.get_ylim(), omitted.get_ylim())
                self.assertEqual([p.get_height() for p in original.patches],
                                 [p.get_height() for p in omitted.patches])
            self.assertEqual(saver.save.call_args.args[1],
                             plot.OUT / f'{model}_{hw}_phases_without_pergpu')

    def test_hashes_dimensions_fonts_and_triplets(self):
        import json
        metadata = json.loads((plot.TABLES / 'metadata.json').read_text())
        self.assertEqual(metadata['missing_aix_cuda_stage_values'], 24)
        missing = plot.read_csv(plot.TABLES / 'missing_cuda.csv')
        self.assertEqual(len(missing), 24)
        self.assertTrue(all(r['mean_s'] == '' and r['status'] == 'unavailable' for r in missing))
        for name, expected in metadata['artifact_sha256'].items():
            self.assertEqual(plot.sha256(name), expected)
        for name, expected in metadata['source_hashes'].items():
            self.assertEqual(plot.sha256(name), expected)
        for extension in ('png', 'pdf', 'pgf'):
            self.assertEqual(len(list(plot.OUT.glob('*.' + extension))), 16)
        self.assertEqual(metadata['figure_triplets'], 16)
        self.assertAlmostEqual(metadata['axes_height_mm'], 37.8)
        for row in plot.read_csv(plot.TABLES / 'render_verification.csv'):
            self.assertAlmostEqual(float(row['pdf_width_mm']), 150, places=2)
            self.assertAlmostEqual(float(row['pdf_height_mm']), 60, places=2)
            self.assertAlmostEqual(float(row['axes_height_mm']), 37.8)
            self.assertEqual(int(row['provider_groups']), 3 if row['figure'].endswith('_without_pergpu') else 4)
            self.assertNotIn('Type 3', row['fonts'])

    def test_pgf_labels_and_authored_width(self):
        for path in plot.OUT.glob('*.pgf'):
            pgf = path.read_text()
            self.assertNotRegex(pgf, r'\bSS\b')
            self.assertIn('SmartSim', pgf)
            if path.stem.endswith('_without_pergpu'):
                self.assertNotIn('SmartSim per-GPU', pgf)
                self.assertNotIn('SmartSim shared', pgf)
            else:
                self.assertIn('SmartSim shared', pgf)
                self.assertIn('SmartSim per-GPU', pgf)


if __name__ == '__main__':
    unittest.main()
