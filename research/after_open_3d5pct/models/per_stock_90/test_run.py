"""Selection rules that do not read the market archive."""
from __future__ import annotations

import unittest

import numpy as np

from research.after_open_3d5pct.models.per_stock_90.coarse_quartile import top_quartile_mask
from research.after_open_3d5pct.models.per_stock_90.run import choose_candidate, wilson_lower


class QuartileTests(unittest.TestCase):
    def test_quiet_day_is_not_a_buy(self):
        vol = np.array([1.0] * 41)
        self.assertFalse(bool(top_quartile_mask(vol)[-1]))

    def test_spike_after_quiet_history_is_a_buy(self):
        vol = np.array([1.0] * 40 + [2.0])
        self.assertTrue(bool(top_quartile_mask(vol)[-1]))


class ChoiceTests(unittest.TestCase):
    def test_keeps_a_ninety_percent_rule(self):
        chosen = choose_candidate([
            {"family": "lightgbm", "precision": 0.80, "buys": 40, "selectivity": 0.5},
            {"family": "morning_vol", "precision": 0.91, "buys": 12, "selectivity": 0.9},
            {"family": "morning_vol", "precision": 0.95, "buys": 10, "selectivity": 0.95},
        ])
        self.assertEqual(chosen["precision"], 0.95)

    def test_none_when_the_tune_window_misses_ninety(self):
        self.assertIsNone(choose_candidate([
            {"family": "lightgbm", "precision": 0.89, "buys": 30, "selectivity": 0.8},
            {"family": "morning_vol", "precision": 1.0, "buys": 9, "selectivity": 0.95},
        ]))

    def test_wilson_lower_is_below_the_point(self):
        lower = wilson_lower(18, 20)
        self.assertIsNotNone(lower)
        self.assertLess(lower, 0.90)
        self.assertGreater(lower, 0.65)


if __name__ == "__main__":
    unittest.main()
