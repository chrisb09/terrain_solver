"""Plot-only checks for disjoint aggregation, scope guards and vertical columns."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from matplotlib.text import Text

from plot_scorep_phases import (MODELS, ORDER, PHASES, column_data, format_ms,
                                plt, render_columns)


def fixture():
    rows = []
    for model in MODELS:
        for cfg in ORDER:
            views = ['application', 'provider_dl' if cfg == 'PHYDLL_CPP' else 'provider']
            for view in views:
                for i, phase in enumerate(PHASES[view] + (
                        ['Other / overhead'] if view == 'provider_dl' else [])):
                    rows.append(dict(model=model, config=cfg, view=view, phase=phase,
                                     scope='dl_steady_count_validated' if view == 'provider_dl' else 'solver_steady',
                                     rep_count='5', mean_ms_per_step=str(i + 1)))
    return rows


class PhasePlotTests(unittest.TestCase):
    def test_disjoint_aggregation_and_aix_scope(self):
        rows = fixture()
        data = column_data(rows)['watercnn']
        self.assertEqual([len(p) for _, _, p in data], [1, 4, 4, 4, 4, 4])
        self.assertEqual(data[0][2], [('Inference', 2)])
        self.assertEqual(data[1][2], [('Input receive', 1), ('Forward', 5),
                                      ('Output send', 8), ('Other', 31)])
        self.assertEqual(data[2][2], [('Put', 3), ('Model request', 4),
                                      ('Retrieval/unpack', 5), ('Other', 9)])
        # Solver inference is deliberately unrelated to the selected DL sum.
        for r in rows:
            if r['config'] == 'PHYDLL_CPP' and r['view'] == 'application':
                r['mean_ms_per_step'] = '999999'
        self.assertEqual(column_data(rows)['watercnn'][1], data[1])

    def test_invalid_inputs(self):
        for field, value in [('mean_ms_per_step', 'nan'), ('mean_ms_per_step', 'inf'),
                             ('mean_ms_per_step', '-1'), ('scope', 'solver_warmup'),
                             ('rep_count', '0'), ('rep_count', '4')]:
            rows = fixture()
            target = next(r for r in rows if r['view'] == 'provider_dl')
            target[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                column_data(rows)
        rows = fixture()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            column_data(rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, 'Missing'):
            column_data(rows[2:])

    def test_tiny_format(self):
        for value, expected in [(0, '0'), (.000001, '0.000001'),
                                (1.234e-12, '0.00000000000123'),
                                (.00999, '0.00999'), (.1, '0.1'),
                                (1.23, '1.23'), (100, '100'),
                                (1234.56, '1,235'), (1000000, '1,000,000')]:
            with self.subTest(value=value):
                self.assertEqual(format_ms(value), expected)

    def test_vertical_render_and_all_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            for stacked, application in [(False, False), (True, False), (True, True)]:
                with patch('plot_scorep_phases.close_figure'):
                    render_columns(fixture(), Path(tmp), stacked, application)
                fig = plt.gcf()
                self.addCleanup(plt.close, fig)
                self.assertEqual(tuple(fig.get_size_inches()), (11, 6))
                self.assertEqual(len(fig.axes), 2)
                self.assertEqual([ax.get_title() for ax in fig.axes], ['[TS] WaterCNN', '[TS] Giant MLP'])
                self.assertEqual(sum('n=5 reps' in t.get_text() for t in fig.texts), int(application))
                if not application:
                    self.assertEqual([t.get_text() for t in fig.texts], ['Library inference phases'])
                    self.assertEqual({t.get_text() for t in fig.legends[0].get_texts()}, {
                        'Inference (AIx library call)',
                        'Input receive (DL-side receive + handling)',
                        'Forward (DL-side model execution)',
                        'Output send (DL-side output transfer)',
                        'Put (client tensor submission)',
                        'Model request (client elapsed time)',
                        'Retrieval/unpack (client retrieval + unpack)',
                        'Other (staging / bookkeeping)',
                    })
                fig.canvas.draw()
                if not application:
                    from matplotlib.patches import FancyBboxPatch
                    borders = [artist for artist in fig.artists if isinstance(artist, FancyBboxPatch)]
                    self.assertEqual(len(borders), 2)
                    self.assertTrue(all(border.get_linestyle() == '--' for border in borders))
                    bounds = fig.legends[0].get_window_extent(fig.canvas.get_renderer())
                    self.assertTrue(fig.bbox.contains(bounds.x0, bounds.y0))
                    self.assertTrue(fig.bbox.contains(bounds.x1, bounds.y1))
                    self.assertLess(bounds.y1, min(ax.get_tightbbox(fig.canvas.get_renderer()).y0
                                                  for ax in fig.axes))
                for ax in fig.axes:
                    expected = 24 if application else 21
                    self.assertEqual(len(ax.patches), expected)
                    self.assertTrue(all(abs(p.get_width() - (.65 if stacked else .18)) < 1e-9
                                        for p in ax.patches))
                    self.assertIn('Mean ms', ax.get_ylabel())
                    self.assertEqual(len(ax.get_xticks()), 6)
                    self.assertEqual(list(ax.get_xticks()), list(range(6)))
                    self.assertTrue(all('_' not in t.get_text() for t in ax.get_xticklabels()))
                    self.assertTrue(all('reps' not in t.get_text() for t in ax.get_xticklabels()))
                    self.assertEqual(len(ax.texts), expected + (6 if stacked else 0))
                    for text in ax.texts + ax.get_yticklabels():
                        self.assertNotRegex(text.get_text(), r'\d[eE][+-]?\d')
                    self.assertEqual(ax.yaxis.get_offset_text().get_text(), '')
                    for text in ax.texts:
                        bounds = text.get_window_extent(fig.canvas.get_renderer())
                        self.assertTrue(fig.bbox.contains(bounds.x0, bounds.y0))
                        self.assertTrue(fig.bbox.contains(bounds.x1, bounds.y1))
                    bounds = [Text.get_window_extent(t, fig.canvas.get_renderer()) for t in ax.texts]
                    for i, box in enumerate(bounds):
                        self.assertFalse(any(box.overlaps(other) for other in bounds[i + 1:]))
                name = 'application_stacked' if application else f'simplified_provider_{"stacked" if stacked else "grouped"}'
                for suffix in ('png', 'pdf'):
                    self.assertGreater((Path(tmp) / f'{name}.{suffix}').stat().st_size, 1000)

    def test_tiny_labels_and_axis_ticks(self):
        rows = fixture()
        for r in rows:
            r['mean_ms_per_step'] = '0.000001'
        with tempfile.TemporaryDirectory() as tmp:
            with patch('plot_scorep_phases.close_figure'):
                render_columns(rows, Path(tmp), True)
            fig = plt.gcf()
            self.addCleanup(plt.close, fig)
            fig.canvas.draw()
            for ax in fig.axes:
                for text in ax.texts + ax.get_yticklabels():
                    self.assertNotRegex(text.get_text(), r'\d[eE][+-]?\d')
                self.assertEqual(ax.yaxis.get_offset_text().get_text(), '')
                bounds = [Text.get_window_extent(t, fig.canvas.get_renderer()) for t in ax.texts]
                for i, box in enumerate(bounds):
                    self.assertTrue(fig.bbox.contains(box.x1, box.y1))
                    self.assertFalse(any(box.overlaps(other) for other in bounds[i + 1:]))


if __name__ == '__main__':
    unittest.main()
