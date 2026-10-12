import unittest
import pandas as pd
from entry_value import core, eligible, calibrate, decide, label_key, development_gate


class CausalityTests(unittest.TestCase):
    def test_strict_joint_maturity_and_expanding_history(self):
        dates = pd.date_range('2025-01-01', periods=300)
        releases = {d: d.tz_localize(core.ET) + pd.Timedelta(days=14) for d in dates}
        cutoff = releases[dates[-1]]
        self.assertEqual(len(eligible(releases, cutoff)), 299)
        self.assertEqual(len(eligible(releases, cutoff, True)), 252)
        self.assertNotIn(dates[-1], eligible(releases, cutoff))

    def test_calibration_uses_only_old_mature_saved_forecasts(self):
        dates = pd.date_range('2025-01-01', periods=4); history = []; labels = {}
        for i, d in enumerate(dates):
            at = d.tz_localize(core.ET) + pd.Timedelta(hours=9, minutes=25)
            history.append({'date': d, 'at': at, 'raw': {s: {'p': .2 + i * .1} for s in core.ETFS},
                'top_symbol': 'SOXL', 'score_ready': i != 0, 'scores': {'SOXL': .01 * i}})
            for s in core.ETFS:
                labels[label_key(at, s)] = {'target_hit': 1 if s == 'SOXL' else 0, 'net_return': .05}
        p = {'calibration_window_dates': 60, 'minimum_calibration_dates': 2}
        c = calibrate(history, labels, dates[:3], p)
        self.assertEqual(c['probability_dates'], list(dates[:3]))
        self.assertEqual(c['top_dates'], list(dates[1:3]))
        self.assertAlmostEqual(c['probability_correction']['SOXL'], .7)
        self.assertAlmostEqual(c['probability_correction']['SOXS'], -.3)
        labels[label_key(history[-1]['at'], 'SOXL')]['net_return'] = 999.
        self.assertEqual(c, calibrate(history, labels, dates[:3], p))

    def test_positive_probability_does_not_imply_positive_payoff(self):
        raw = {s: {'p': .8, 'win': .05, 'failure': -.30, 'direct': .01} for s in core.ETFS}
        cal = {'probability_correction': {s: 0. for s in core.ETFS}, 'probability_ready': True,
               'top_correction': 0., 'top_ready': True}
        e = decide(raw, cal, 'two_part_expanding')
        self.assertAlmostEqual(e['scores']['SOXL'], -.02)
        self.assertEqual(e['action'], 'CASH')

    def test_probability_calibration_cannot_replace_top_value_calibration(self):
        raw = {s: {'p': .8, 'win': .05, 'failure': -.06, 'direct': .01} for s in core.ETFS}
        c = {'probability_correction': {s: 0. for s in core.ETFS}, 'probability_ready': True,
             'top_correction': None, 'top_ready': False}
        self.assertEqual(decide(raw, c, 'two_part_expanding')['action'], 'CASH')
        c.update(top_correction=0., top_ready=True)
        self.assertEqual(decide(raw, c, 'two_part_expanding')['action'], 'SOXL')
        c.update(probability_ready=False)
        self.assertEqual(decide(raw, c, 'two_part_expanding')['action'], 'CASH')

    def test_empirical_probability_is_bounded(self):
        raw = {s: {'p': .8, 'win': .05, 'failure': -.06, 'direct': .01} for s in core.ETFS}
        c = {'probability_correction': {'SOXL': 1., 'SOXS': -1.}, 'probability_ready': True,
             'top_correction': 0., 'top_ready': True}
        self.assertEqual(decide(raw, c, 'two_part_expanding')['probabilities'], {'SOXL': .999, 'SOXS': .001})

    def test_negative_month_or_cash_cannot_pass_screen(self):
        months = ['2026-01', '2026-02', '2026-03', '2026-04']; p = {'development': {'months': months}}
        def stats(rates, n=20):
            return {'monthly': [{'month': m, 'return': r, 'mark_complete': True} for m, r in zip(months, rates)], 'completed_trades': n}
        self.assertFalse(development_gate(stats([.5, .5, .5, -.01]), True, p)['passed'])
        self.assertFalse(development_gate(stats([0.] * 4, 0), True, p)['passed'])
        self.assertTrue(development_gate(stats([.11] * 4), True, p)['passed'])


if __name__ == '__main__':
    unittest.main()
