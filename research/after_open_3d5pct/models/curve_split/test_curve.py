"""Curve geometry and the date split. No market archive."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.curve_split.run import curve_geometry, macro_window, split_role


class GeometryTests(unittest.TestCase):
    def test_rising_line_peaks_at_the_end(self):
        beta, kappa, peak, _trough, fade = curve_geometry(np.linspace(-1, 1, 24))
        self.assertGreater(beta, 0)
        self.assertAlmostEqual(peak, 1.0)
        self.assertAlmostEqual(fade, 0.0)
        self.assertLess(abs(kappa), 1e-6)

    def test_early_spike_is_given_back(self):
        z = np.array([0.0, 3.0] + [0.0] * 22)
        _beta, _kappa, peak, _trough, fade = curve_geometry(z)
        self.assertLess(peak, 0.2)
        self.assertAlmostEqual(fade, 1.0)


class WindowTests(unittest.TestCase):
    def test_window_ends_on_the_decision_and_has_35_points(self):
        ends = pd.date_range("2025-06-02 10:30", periods=40, freq="60min").to_numpy()
        closes = np.arange(40, dtype=float)
        decision = pd.Timestamp(ends[34])
        window = macro_window(ends, closes, decision)
        self.assertIsNotNone(window)
        self.assertEqual(len(window), 35)
        self.assertEqual(window[-1], closes[34])
        self.assertEqual(window[0], closes[0])

    def test_short_history_has_no_window(self):
        ends = pd.date_range("2025-06-02 10:30", periods=10, freq="60min").to_numpy()
        window = macro_window(ends, np.arange(10, dtype=float), pd.Timestamp(ends[-1]))
        self.assertIsNone(window)


class SplitTests(unittest.TestCase):
    def test_2026_is_test_and_the_sets_do_not_meet(self):
        self.assertEqual(split_role("2025-06-30"), "train")
        self.assertEqual(split_role("2025-07-01"), "validation")
        self.assertEqual(split_role("2025-12-31"), "validation")
        self.assertEqual(split_role("2026-01-02"), "test")
        self.assertEqual(split_role("2026-09-18"), "test")
        self.assertIsNone(split_role("2026-09-21"))
        roles = [split_role(day) for day in ("2025-06-30", "2025-07-01", "2026-01-02")]
        self.assertEqual(len(set(roles)), 3)
