"""Frozen-rule checks that do not read the market archive."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.pugh_ge5.build_rows import (
    complete_daily, expanding_base, history_floor, label_hits, valid_ohlcv, window_end,
)
from research.after_open_3d5pct.models.pugh_ge5.walk import choose_threshold


class WindowTests(unittest.TestCase):
    def test_exact_horizon(self):
        self.assertEqual(window_end(np.array([5, 5, 5, 5]), 0, 10), 1)

    def test_overshoot_is_immature(self):
        self.assertIsNone(window_end(np.array([5, 30]), 0, 10))

    def test_primary_hit_uses_entry_bar(self):
        high = np.array([10.0, 10.6, 10.0, 10.0])
        minutes = np.array([390, 390, 390, 390])
        # horizon 1170 cannot complete in four 390-minute bars; use a tiny stand-in via window_end
        self.assertEqual(window_end(minutes, 0, 1170), 2)
        labels = label_hits(np.array([10.6] + [10.0] * 233), np.array([5] * 234), 0, 10.0)
        self.assertEqual(labels["y_3d_5pct"], 1.0)
        flat = label_hits(np.array([10.1] * 234), np.array([5] * 234), 0, 10.0)
        self.assertEqual(flat["y_3d_5pct"], 0.0)


class BaseTests(unittest.TestCase):
    def test_future_label_end_is_excluded(self):
        ends = [pd.Timestamp("2026-09-24 11:35")]
        hits = [1.0]
        self.assertTrue(np.isnan(expanding_base(ends * 40, hits * 40, pd.Timestamp("2026-09-24 11:30"), minimum=40)))
        later = expanding_base(ends * 40, hits * 40, pd.Timestamp("2026-09-24 11:35"), minimum=40)
        self.assertEqual(later, 1.0)


class HistoryFloorTests(unittest.TestCase):
    def test_2024_floor_keeps_later_listing(self):
        self.assertEqual(history_floor(None), "2024-01-01")
        self.assertEqual(history_floor("2025-02-24"), "2025-02-24")
        self.assertEqual(history_floor("2023-09-14"), "2024-01-01")

    def test_complete_daily_requires_exact_bars(self):
        bars = pd.DataFrame({
            "session_date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "start": pd.to_datetime(["2024-01-02 09:30", "2024-01-02 09:35", "2024-01-03 09:30"]),
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "volume": [1.0, 2.0, 3.0],
        })
        daily = complete_daily(bars, {"2024-01-02": 2, "2024-01-03": 2})
        self.assertEqual(daily["session_date"].tolist(), ["2024-01-02"])
        self.assertEqual(float(daily["close"].iloc[0]), 11.5)
        self.assertEqual(float(daily["volume"].iloc[0]), 3.0)

    def test_invalid_bar_rejects_low_above_close(self):
        frame = pd.DataFrame({"open": [2.35], "high": [2.35], "low": [2.35], "close": [2.3], "volume": [435.0]})
        self.assertFalse(bool(valid_ohlcv(frame).iloc[0]))


class ThresholdTests(unittest.TestCase):
    def test_fallback_when_too_few_buys(self):
        frame = pd.DataFrame({"excess_score": [0.5], "y_3d_5pct": [1.0], "own_base": [0.4]})
        self.assertEqual(choose_threshold(frame), (0.05, "fallback"))

    def test_tie_keeps_higher_threshold(self):
        frame = pd.DataFrame({
            "excess_score": [0.2] * 8,
            "y_3d_5pct": [1.0] * 8,
            "own_base": [0.5] * 8,
        })
        threshold, rule = choose_threshold(frame)
        self.assertEqual(rule, "selected")
        self.assertEqual(threshold, 0.2)


if __name__ == "__main__":
    unittest.main()
