"""Morning impulse and the 30-day slope. No market archive."""
from __future__ import annotations

import unittest

import numpy as np

from research.after_open_3d5pct.models.potential_v1.run import impulse_from_path, macro_shape


class ImpulseTests(unittest.TestCase):
    def test_a_straight_rise_keeps_its_move(self):
        closes = np.linspace(100, 103, 24)
        net = closes[-1] / 100 - 1
        impulse = impulse_from_path(100, closes)
        self.assertAlmostEqual(impulse, net, places=6)

    def test_a_choppy_rise_shrinks(self):
        closes = np.array([100, 110, 100, 110, 100, 110], dtype=float)
        net = closes[-1] / 100 - 1
        impulse = impulse_from_path(100, closes)
        self.assertGreater(impulse, 0)
        self.assertLess(impulse, net * 0.5)

    def test_a_straight_drop_stays_negative(self):
        closes = np.linspace(100, 97, 24)
        self.assertLess(impulse_from_path(100, closes), 0)


class MacroTests(unittest.TestCase):
    def test_rising_and_falling_slopes_have_opposite_signs(self):
        rising = macro_shape(np.linspace(100, 130, 30))
        falling = macro_shape(np.linspace(130, 100, 30))
        self.assertIsNotNone(rising)
        self.assertGreater(rising[0], 0)
        self.assertLess(falling[0], 0)

    def test_only_the_last_thirty_closes_enter_and_short_history_stays_empty(self):
        self.assertIsNone(macro_shape(np.linspace(100, 130, 29)))
        older_crash = np.concatenate([np.linspace(200, 50, 10), np.linspace(100, 130, 30)])
        matched = macro_shape(np.linspace(100, 130, 40))
        self.assertAlmostEqual(macro_shape(older_crash)[0], matched[0], places=6)
