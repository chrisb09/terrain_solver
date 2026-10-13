import unittest
from pathlib import Path
from terrain_cpu_complete_analysis import ORDER, MODELS, coverage, selectors, maxima, OUTPUT, read_csv


class CompleteAnalysisTests(unittest.TestCase):
    def rows(self):
        return [dict(model=m, config=c, rep=str(rep)) for m in MODELS for c in ORDER for rep in range(1, 6)]

    def test_exact_coverage(self):
        coverage(self.rows())

    def test_readiness_collision_rejected(self):
        rows = self.rows()
        for r in rows:
            if r['config'] == 'PHYDLL_TRUE':
                r['config'] = 'PHYDLL_FALSE'
        with self.assertRaises(ValueError):
            coverage(rows)

    def test_missing_rep_rejected(self):
        with self.assertRaises(ValueError):
            coverage(self.rows()[:-1])

    def test_smoke_not_selected(self):
        for kind in ('timing', 'memory', 'performance'):
            self.assertTrue(all('smoke' not in r[0] for r in selectors(kind)))
        self.assertEqual(selectors('timing')[0][3], '4952588')
        self.assertEqual(selectors('memory')[-1][3], '4952823')

    def test_independent_metric_maxima(self):
        rows = self.rows()
        for r in rows:
            r.update(pss_peak_bytes=int(r['rep']), rss_peak_bytes=10-int(r['rep']), memory_json='source'+r['rep'])
        for r in maxima(rows):
            self.assertEqual((r['pss_peak_bytes_rep'], r['rss_peak_bytes_rep'], r['rep_count']), ('5', '1', 5))

    @unittest.skipUnless(OUTPUT.exists(), 'Generated artifact integration check')
    def test_generated_artifacts(self):
        for kind in ('timing', 'memory', 'performance'):
            coverage(read_csv(OUTPUT / (kind+'_replicates.csv')))
        self.assertEqual(len(read_csv(OUTPUT / 'latency_steps.csv')), 700)
        for row in read_csv(OUTPUT / 'phase_means.csv'):
            self.assertEqual((row['rep_count'], row['rank_count'], row['steady_steps']), ('5', '96', '10'))
        for name in ('latency_boxplots', 'memory_peaks', 'application_stacked', 'provider_stacked'):
            for suffix in ('png', 'pdf'):
                self.assertGreater((OUTPUT / (name+'.'+suffix)).stat().st_size, 10000)


if __name__ == '__main__':
    unittest.main()
