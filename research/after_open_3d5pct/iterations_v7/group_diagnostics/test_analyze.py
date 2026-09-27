import unittest
import numpy as np

from .analyze import factor_stats, past_window, shrink_rate, within_stock_auc


class GroupDiagnosticSemantics(unittest.TestCase):
    def test_future_and_today_do_not_change_past_window(self):
        values = np.arange(40, dtype=float)
        expected = past_window(values, 30)
        changed = np.r_[values, [1000., -1000.]]
        changed[30:] = -42
        np.testing.assert_array_equal(expected, past_window(changed, 30))
        self.assertEqual(expected[-1], 29)

    def test_perfect_inverse_market_relationship_is_high_explanatory_power(self):
        m = np.cumsum(np.r_[0., np.tile([.01, -.005, .015, -.01], 5)])
        result = factor_stats(5-2*m, 4+m)
        self.assertAlmostEqual(result['beta'], -2)
        self.assertAlmostEqual(result['r2'], 1)
        self.assertLess(result['residual_daily_sigma'], 1e-12)

    def test_missing_official_session_is_unknown_not_bridged(self):
        a = np.arange(21, dtype=float)
        a[5] = np.nan
        self.assertIsNone(factor_stats(a, np.arange(21, dtype=float)))
        self.assertIsNone(past_window(np.arange(30), 20))

    def test_absent_training_group_falls_back_to_training_rate(self):
        rate = np.array([.1, .2])
        np.testing.assert_array_equal(shrink_rate(np.empty((0, 2)), rate, 50), rate)

    def test_pooled_stock_identity_does_not_create_within_stock_auc(self):
        # Perfect stock-level discrimination but no within-stock positive/negative pair.
        result = within_stock_auc(np.array([0, 0, 1, 1]), np.array([.1, .1, .9, .9]),
                                  np.array(['A', 'A', 'B', 'B']))
        self.assertIsNone(result['auc'])
        self.assertEqual(result['positive_negative_pairs'], 0)


if __name__ == '__main__':
    unittest.main()
