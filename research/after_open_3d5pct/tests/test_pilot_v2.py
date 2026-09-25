"""Focused guards for the new causal market-path representation."""

import math
import unittest

import numpy as np
import pandas as pd

from research.after_open_3d5pct.pilot_v2_data import _safe_ratio, _sequence


class PilotV2RepresentationTest(unittest.TestCase):
    def test_missing_bar_has_different_mask_from_observed_flat_bar(self):
        idx = pd.date_range("2026-08-03T13:30:00Z", periods=2, freq="5min")
        observed = pd.DataFrame({"open": [100.0, np.nan], "high": [100.0, np.nan],
                                 "low": [100.0, np.nan], "close": [100.0, np.nan],
                                 "volume": [25.0, np.nan]}, index=idx)
        seq = _sequence(observed, observed, observed, 2)
        self.assertEqual(seq[:, 6].tolist(), [1.0, 0.0])
        observed.iloc[1] = [100.0, 100.0, 100.0, 100.0, 0.0]
        flat = _sequence(observed, observed, observed, 2)
        self.assertEqual(flat[:, 6].tolist(), [1.0, 1.0])

    def test_appending_future_bar_does_not_change_prefix_encoding(self):
        idx = pd.date_range("2026-08-03T13:30:00Z", periods=3, freq="5min")
        rows = pd.DataFrame({"open": [100.0, 101.0, 50.0], "high": [101.0, 102.0, 100.0],
                             "low": [99.0, 100.0, 49.0], "close": [101.0, 101.5, 99.0],
                             "volume": [10.0, 20.0, 10000.0]}, index=idx)
        before = _sequence(rows.iloc[:2], rows.iloc[:2], rows.iloc[:2], 3)
        after = _sequence(rows, rows, rows, 3)
        np.testing.assert_array_equal(before[:2], after[:2])

    def test_zero_history_denominator_is_missing(self):
        self.assertTrue(math.isnan(_safe_ratio(100.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
