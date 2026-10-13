import unittest
import numpy as np
import pandas as pd
from error_audit import actual_class, bootstrap, cohort, decompose, summary, TERMS


class ErrorAuditTests(unittest.TestCase):
    def test_exact_known_class_has_only_amplitude_and_regression_error(self):
        x = decompose([0, 0, 1, 0], [.048, -.064, -.09, 0], -.07, 2, -.11)
        self.assertAlmostEqual(x['total_error'], .04)
        self.assertAlmostEqual(x['regression_adjustment'], .02)
        self.assertEqual(x['category_error'], 0)
        self.assertAlmostEqual(x['payoff_error'], .02)

    def test_probability_of_missing_gap_is_positive_overestimate(self):
        x = decompose([.6, .3, .1, 0], [.05, -.06, -.1, 0], .001, 2, -.1)
        self.assertGreater(x['gap_contribution'], 0)
        self.assertEqual(x['payoff_error'], 0)
        self.assertAlmostEqual(sum(x[k] for k in ['ordinary_contribution', 'gap_contribution', 'expiry_contribution']), x['category_error'])

    def test_algebra_all_classes_and_payoff_translation(self):
        q = [.55, .3, .1, .05]; m = np.array([.049, -.064, -.11, -.01])
        for y in range(4):
            a = decompose(q, m, .012, y, m[y] + .003)
            b = decompose(q, m + .7, .712, y, m[y] + .703)
            for name in TERMS[3:]:
                self.assertAlmostEqual(a[name], b[name])

    def test_invalid_probabilities_rejected(self):
        for q in [[.6, .5, 0, 0], [-.1, .7, .4, 0], [1, 0, float('nan'), 0]]:
            with self.assertRaises(ValueError):
                decompose(q, [.05, -.06, -.1, 0], .01, 0, .05)

    def test_unknown_not_fabricated_as_expiry(self):
        for y, value in [(-1, 0), (4, .05), (0, float('nan'))]:
            with self.assertRaises(ValueError):
                decompose([.6, .3, .1, 0], [.05, -.06, -.1, 0], .01, y, value)

    def test_gap_is_execution_flag_not_loss_threshold(self):
        self.assertEqual(actual_class('risk_stop', True), 2)
        self.assertEqual(actual_class('risk_stop', False), 1)
        self.assertEqual(actual_class('target', False), 0)
        with self.assertRaises(ValueError):
            actual_class('pending', False)

    def test_date_equal_not_clock_count_weighting_and_unknown_retained(self):
        rows = []
        for date, count, actual, known in [('a', 8, .1, True), ('b', 1, -.1, True), ('c', 1, 0, False)]:
            for _ in range(count):
                row = dict(date=date, known=known, paired_ready=True, paired_mature=known, truth=0)
                row.update(decompose([1, 0, 0, 0], [.05, -.06, -.1, 0], .05, 0, actual))
                row.update({f'q{k}': float(k == 0) for k in range(4)})
                row.update({f'm{k}': x for k, x in enumerate([.05, -.06, -.1, 0])})
                rows.append(row)
        s = summary(pd.DataFrame(rows))
        self.assertEqual(s['unknown_rows'], 1)
        self.assertEqual(s['known_dates'], 2)
        self.assertAlmostEqual(s['date_equal_terms']['actual'], 0)
        self.assertAlmostEqual(s['class_statistics']['target']['observed_frequency'], 1)

    def test_empty_bootstrap_draws_remain_unknown(self):
        t = pd.DataFrame([dict(date='day0', known=False, **{k: np.nan for k in TERMS})])
        out = bootstrap(t, ['day0', 'day1'])
        self.assertEqual(out['unknown_draws'], 5000)
        self.assertTrue(all(x is None for x in out['intervals'].values()))

    def test_selection_uses_issued_action_not_outcome(self):
        flags = dict(source_ready=True, paired_ready=True, absolute_pair_accepted=False, absolute_now_accepted=False)
        t = pd.DataFrame([dict(**flags, source_eligible=True, paired_accepted=False, actual=.1),
                          dict(**flags, source_eligible=True, paired_accepted=True, actual=-.1),
                          dict(**flags, source_eligible=False, paired_accepted=False, actual=.2)])
        self.assertEqual(cohort(t, 'paired_accepted').actual.tolist(), [-.1])
        self.assertEqual(cohort(t, 'paired_rejected').actual.tolist(), [.1])


if __name__ == '__main__':
    unittest.main()
