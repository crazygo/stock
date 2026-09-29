"""Peer average and the eight-week window. No market archive."""
from __future__ import annotations

import unittest

import pandas as pd

from research.after_open_3d5pct.models.break_v1.run import peer_morning, window_days
from research.after_open_3d5pct.models.weekly_scale.run import week_blocks


class PeerTests(unittest.TestCase):
    def test_peer_excludes_the_stock_itself(self):
        frame = pd.DataFrame({
            "session_date": ["2025-06-02", "2025-06-02", "2025-06-02"],
            "morning_return": [0.01, 0.03, 0.05],
        })
        peer = peer_morning(frame)
        self.assertAlmostEqual(peer.iloc[0], 0.04)
        self.assertAlmostEqual(peer.iloc[2], 0.02)


class WindowTests(unittest.TestCase):
    def test_eight_training_weeks_stop_before_the_test_week(self):
        dates = [f"2025-01-{day:02d}" for day in range(1, 61)]
        weeks = week_blocks(dates)
        train, test, validation = window_days(weeks, 8)
        self.assertEqual(len(train), 40)
        self.assertEqual(set(train) & set(test), set())
        self.assertLess(max(train), min(test))
        self.assertLess(max(test), min(validation))
