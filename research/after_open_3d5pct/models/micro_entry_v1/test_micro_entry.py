"""Entry is the next bar. The window's own high is not the label."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.micro_entry_v1.rows import (
    agrees_with_label_hits,
    horizon_last,
    rows_from_bars,
    sliding_window_max,
)
from research.after_open_3d5pct.models.micro_entry_v1.run import first_trigger
from research.after_open_3d5pct.models.pugh_ge5.build_rows import window_end


def _session(day: str, n: int = 78, price: float = 100.0) -> pd.DataFrame:
    start = pd.date_range(pd.Timestamp(f"{day} 09:30"), periods=n, freq="5min")
    return pd.DataFrame({
        "start": start,
        "end": start + pd.Timedelta(minutes=5),
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "minutes": 5,
    })


class HorizonTests(unittest.TestCase):
    def test_uniform_five_minute_bars_match_the_slow_scan(self):
        minutes = np.full(400, 5)
        high = np.full(400, 100.0)
        high[20:30] = 130.0
        self.assertEqual(horizon_last(minutes, 10), window_end(minutes, 10, 1170))
        self.assertTrue(agrees_with_label_hits(minutes, high, 10, 100.0))

    def test_sliding_max_is_the_future_high_from_the_entry_bar(self):
        values = np.array([1, 5, 2, 4], dtype=float)
        found = sliding_window_max(values, 2)
        self.assertTrue(np.isnan(found[0]))
        self.assertEqual(list(found[1:]), [5, 5, 4])


class EntryTests(unittest.TestCase):
    def test_two_hour_windows_enter_at_1130_and_ignore_their_own_spike(self):
        days = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"]
        bars = pd.concat([_session(day) for day in days], ignore_index=True)
        bars.loc[:23, "high"] = 200.0
        bars.loc[258:258 + 233, "high"] = 110.0
        built = rows_from_bars(bars, length=24, step=1, symbol="TEST")
        opening = built.loc[built["session_date"].eq("2026-01-02")].sort_values("decision_at").iloc[0]
        self.assertEqual(pd.Timestamp(opening["decision_at"]), pd.Timestamp("2026-01-02 11:30"))
        self.assertEqual(pd.Timestamp(opening["entry_at"]), pd.Timestamp("2026-01-02 11:30"))
        self.assertEqual(opening["y_3d_5pct"], 0.0)
        later = built.loc[
            built["session_date"].eq("2026-01-07") & built["decision_at"].eq(pd.Timestamp("2026-01-07 11:30"))
        ]
        self.assertEqual(float(later["y_3d_5pct"].iloc[0]), 1.0)

    def test_thirty_minute_windows_start_at_1000_and_skip_today_in_the_slope(self):
        days = pd.bdate_range("2026-01-05", periods=45)
        bars = pd.concat([_session(day.strftime("%Y-%m-%d"), price=100 + index) for index, day in enumerate(days)], ignore_index=True)
        mutated = bars.copy()
        target = days[40].strftime("%Y-%m-%d")
        mutated.loc[mutated["start"].dt.strftime("%Y-%m-%d").eq(target), "close"] = 1.0
        original = rows_from_bars(bars, length=6, step=1, symbol="TEST")
        changed = rows_from_bars(mutated, length=6, step=1, symbol="TEST")
        opening = changed.loc[changed["session_date"].eq(target)].sort_values("decision_at").iloc[0]
        self.assertEqual(pd.Timestamp(opening["decision_at"]), pd.Timestamp(f"{target} 10:00"))
        before = original.loc[original["session_date"].eq(target), "macro_30d_beta"].iloc[0]
        self.assertAlmostEqual(float(opening["macro_30d_beta"]), float(before), places=6)


class TriggerTests(unittest.TestCase):
    def test_the_earlier_window_is_the_entry(self):
        frame = pd.DataFrame({
            "symbol": ["A", "A"],
            "session_date": ["2026-01-02", "2026-01-02"],
            "decision_at": [pd.Timestamp("2026-01-02 11:30"), pd.Timestamp("2026-01-02 11:35")],
            "score": [0.9, 0.1],
            "level": [0.5, 0.5],
            "y_3d_5pct": [0.0, 1.0],
        })
        found = first_trigger(frame)
        self.assertEqual(found["buys"], 1)
        self.assertEqual(found["precision"], 0.0)
