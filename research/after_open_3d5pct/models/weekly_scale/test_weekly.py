"""Week order, label purge, and the micro times macro product."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.weekly_scale.run import (
    add_relations,
    burst_from_closes,
    matured_before,
    week_blocks,
)


class WeekTests(unittest.TestCase):
    def test_blocks_are_five_days_and_do_not_share_a_date(self):
        dates = [f"2025-01-{day:02d}" for day in range(1, 13)]
        weeks = week_blocks(dates)
        self.assertEqual(len(weeks), 2)
        self.assertEqual(len(weeks[0]), 5)
        self.assertEqual(set(weeks[0]) & set(weeks[1]), set())

    def test_a_label_that_ends_as_the_next_week_starts_is_out_of_training(self):
        frame = pd.DataFrame({
            "session_date": ["2025-06-02", "2025-06-03"],
            "label_end": [pd.Timestamp("2025-06-05 11:00"), pd.Timestamp("2025-06-09 11:30")],
        })
        kept = matured_before(frame, ["2025-06-02", "2025-06-03"], pd.Timestamp("2025-06-09 11:30"))
        self.assertEqual(list(kept["session_date"]), ["2025-06-02"])


class RelationTests(unittest.TestCase):
    def test_product_uses_the_two_scales(self):
        frame = add_relations(pd.DataFrame({
            "gap": [0.02],
            "macro_beta": [0.5],
            "macro_kappa": [-0.2],
            "burst_5m": [0.01],
            "macro_fade": [0.4],
            "overnight_return": [0.03],
            "premarket_return": [0.01],
            "macro_vol": [0.02],
        }))
        self.assertAlmostEqual(frame["gap_x_beta"].iloc[0], 0.01)
        self.assertAlmostEqual(frame["gap_over_vol"].iloc[0], 1.0)

    def test_burst_is_the_largest_five_minute_step(self):
        closes = np.array([100.0, 100.0, 103.0, 103.0])
        size, where = burst_from_closes(closes)
        self.assertAlmostEqual(size, 0.03)
        self.assertAlmostEqual(where, 1 / 3)
