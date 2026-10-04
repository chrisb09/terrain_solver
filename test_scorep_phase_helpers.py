"""Regression tests for steady extraction, without builds or external jobs."""
import unittest
import csv
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from extract_scorep_tables import discover, extract


def node(cid, name, parent, incl, excl, visits):
    return SimpleNamespace(id=cid, name=name, parent_id=parent,
                           incl_time=incl, excl_time=excl, visits=visits)


class SteadyPhases(unittest.TestCase):
    def setUp(self):
        self.meta = dict(model="watercnn", config="SS_DEFAULT", rep="1", cubex="synthetic")
        self.tree = {
            0: node(0, "terrain_solver", None, 110, 0, 1),
            1: node(1, "solver_step_ml_warmup", 0, 100, 0, 1),
            2: node(2, "smartsim_put_tensor", 1, 100, 100, 2),
            3: node(3, "solver_step_ml_steady", 0, 10, 1, 10),
            4: node(4, "app_provider_inference", 3, 9, 1, 10),
            5: node(5, "smartsim_put_tensor", 4, 4, 4, 20),
            6: node(6, "smartsim_run_model", 4, 3, 3, 10),
            7: node(7, "app_finalize_output", 4, 1, 1, 10),
        }

    def test_repeated_put_is_not_per_call(self):
        raw, phases = extract([self.tree, self.tree], self.meta)
        put = next(r for r in phases if r['view'] == 'provider' and r['phase'] == 'smartsim_put_tensor')
        self.assertEqual(put['raw_calls'], 40)
        self.assertEqual(put['denominator'], 20)
        self.assertEqual(put['ms_per_step'], 400)
        warmup = next(r for r in raw if 'warmup' in r['callpath'] and r['region'] == 'smartsim_put_tensor')
        self.assertFalse(warmup['selected_steady'])
        self.assertEqual(warmup['raw_inclusive_s'], 200)

    def test_nested_application_no_doublecount(self):
        _, phases = extract([self.tree], self.meta)
        app = {r['phase']: r['ms_per_step'] for r in phases if r['view'] == 'application'}
        self.assertEqual(sum(app.values()), 1000)
        self.assertEqual(app['app_provider_inference'], 800)
        self.assertEqual(app['app_finalize_output'], 100)

    def test_aix_forward_excludes_warmup(self):
        self.meta['config'] = 'AIx'
        self.tree[2].name = self.tree[5].name = 'torchInference::forward'
        _, phases = extract([self.tree], self.meta)
        fwd = next(r for r in phases if r['phase'] == 'torchInference::forward')
        self.assertEqual(fwd['raw_total_s'], 4)
        self.assertEqual(fwd['ms_per_step'], 400)

    def test_dl_send_nested_once_and_missing_h2d_zero(self):
        self.meta['config'] = 'PHYDLL_CPP'
        dl = {0: node(0, 'phydll_dl_client', None, 999, 990, 1),
              1: node(1, 'dl_torch_forward', 0, 5, 5, 10),
              2: node(2, 'dl_send_output', 0, 4, 1, 10),
              3: node(3, 'dl_send', 2, 3, 3, 10)}
        _, phases = extract([self.tree, dl], self.meta)
        values = {r['phase']: r for r in phases if r['view'] == 'provider_dl'}
        self.assertEqual(values['dl_send_output']['raw_total_s'], 4)
        self.assertEqual(values['dl_send_output']['raw_calls'], 10)
        self.assertEqual(values['dl_h2d']['ms_per_step'], 0)
        self.assertEqual(sum(r['ms_per_step'] for r in values.values()), 900)
        dl[1].visits = 11
        with self.assertRaisesRegex(ValueError, 'cannot remove warmup'):
            extract([self.tree, dl], self.meta)

    def test_inconsistent_steps_fail(self):
        with self.assertRaisesRegex(ValueError, 'Inconsistent steady'):
            extract([self.tree], self.meta, steady_steps=11)

    def test_old_new_names_and_manifest_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('campaign_watercnn_rep1_123', 'campaign_watercnn_r2_456'):
                scorep = root / name / 'AIX' / 'scorep'
                scorep.mkdir(parents=True)
                (scorep / 'profile.cubex').touch()
            runs = discover([root], None)
            self.assertEqual({r['rep'] for r in runs}, {'1', '2'})
            manifest = root / 'profiles.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(runs[0]))
                writer.writeheader()
                writer.writerow(runs[0])
            self.assertEqual(discover([], manifest), [runs[0]])
            nested = root / 'job_789' / 'giant' / 'rep3' / 'PHYDLL_CPP' / 'scorep'
            nested.mkdir(parents=True)
            (nested / 'profile.cubex').touch()
            new = discover([root / 'job_789'], None)
            self.assertEqual((new[0]['model'], new[0]['config'], new[0]['rep']),
                             ('giant', 'PHYDLL_CPP', '3'))


class CompletedProfileTables(unittest.TestCase):
    """Audit real extracted data when available; otherwise skip cleanly."""
    def test_completed_profile_algebra(self):
        out = Path(__file__).resolve().parent / 'campaign_figures/steady_phases'
        if not (out / 'raw_callpaths.csv').exists():
            self.skipTest('Run extract_scorep_tables.py to audit completed profiles')
        with (out / 'raw_callpaths.csv').open(newline='') as stream:
            raw = list(csv.DictReader(stream))
        with (out / 'phase_replicates.csv').open(newline='') as stream:
            phases = list(csv.DictReader(stream))
        roots, app = {}, defaultdict(float)
        for r in raw:
            key = (r['model'], r['config'], r['rep'])
            if 'solver_step_ml_warmup' in r['callpath']:
                self.assertEqual(r['selected_steady'], 'False')
            if r['region'] == 'solver_step_ml_steady':
                roots[key] = float(r['raw_inclusive_s'])
        for r in phases:
            key = (r['model'], r['config'], r['rep'])
            seconds = float(r['raw_total_s'])
            self.assertAlmostEqual(float(r['ms_per_step']), seconds * 1000 / int(r['denominator']))
            if r['view'] == 'application':
                app[key] += seconds
            if r['phase'] == 'smartsim_put_tensor':
                self.assertEqual(int(r['raw_calls']), 2 * int(r['denominator']))
            if r['view'] == 'provider_dl':
                self.assertNotIn('phydll_send', r['regions'])
                self.assertNotIn('phydll_recv', r['regions'])
                if r['phase'] == 'dl_send_output':
                    self.assertEqual(int(r['raw_calls']), int(r['denominator']))
        self.assertEqual(set(app), set(roots))
        for key in roots:
            self.assertAlmostEqual(app[key], roots[key], delta=max(1e-5, roots[key] * 1e-5))
        print(f'Completed-profile audit: {len(roots)} application sums; repeated puts and DL send scopes valid')


if __name__ == '__main__':
    unittest.main()
