"""Handmade geometry checks. No market data."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.micro_entry_v1.rows import rows_from_bars
from research.after_open_3d5pct.models.micro_entry_v1.tracks.shape.run import (
    bar_returns,
    eighth_returns,
    half_returns,
    peer_impulse,
    quarter_returns,
    window_shape_rows,
)


def _bars(days: list[str], n: int = 78, price: float = 100.0) -> pd.DataFrame:
    frames = []
    for day in days:
        start = pd.date_range(pd.Timestamp(f"{day} 09:30"), periods=n, freq="5min")
        frames.append(pd.DataFrame({
            "start": start,
            "end": start + pd.Timedelta(minutes=5),
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "minutes": 5,
        }))
    return pd.concat(frames, ignore_index=True)


class HalfGapTests(unittest.TestCase):
    def test_known_24_bar_path(self):
        closes = np.full(24, 50.0)
        closes[11] = 110.0
        closes[23] = 99.0
        early, late, gap = half_returns(100.0, closes)
        self.assertAlmostEqual(early, 0.10)
        self.assertAlmostEqual(late, 99.0 / 110.0 - 1.0)
        self.assertAlmostEqual(gap, late - early)
        self.assertAlmostEqual(gap, -0.20)

    def test_non_finite_or_non_positive_is_nan_not_zero(self):
        closes = np.full(24, 100.0)
        closes[11] = 110.0
        closes[23] = 120.0
        for open0, path in (
            (0.0, closes),
            (-1.0, closes),
            (np.nan, closes),
            (100.0, np.array([*closes[:11], 0.0, *closes[12:]])),
            (100.0, np.array([*closes[:11], np.nan, *closes[12:]])),
            (100.0, np.array([*closes[:23], np.nan])),
        ):
            early, late, gap = half_returns(open0, path)
            self.assertTrue(np.isnan(early) and np.isnan(late) and np.isnan(gap))
            self.assertNotEqual(gap, 0.0)

    def test_window_rows_match_the_sliding_identity_and_the_known_path(self):
        bars = _bars(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        bars.loc[0, "open"] = 80.0
        bars.loc[11, "close"] = 88.0
        bars.loc[23, "close"] = 96.8
        bars.loc[12, "close"] = 110.0
        bars.loc[24, "close"] = 121.0
        bars.loc[30, "open"] = 0.0
        bars = bars.drop(index=100).reset_index(drop=True)
        built = rows_from_bars(bars, length=24, step=1, symbol="TEST")
        shape = window_shape_rows(bars, "TEST")
        keys = ["symbol", "session_date", "decision_at"]
        left = built[keys].copy()
        right = shape[keys].copy()
        left["decision_at"] = pd.to_datetime(left["decision_at"])
        right["decision_at"] = pd.to_datetime(right["decision_at"])
        merged = left.merge(right, on=keys, how="outer", indicator=True)
        self.assertTrue((merged["_merge"] == "both").all())
        self.assertEqual(len(merged), len(built))
        self.assertFalse(left.duplicated(keys).any())
        found = shape.copy()
        found["decision_at"] = pd.to_datetime(found["decision_at"])
        first = found.loc[found["decision_at"].eq(pd.Timestamp("2024-01-02 11:30"))].iloc[0]
        self.assertAlmostEqual(float(first["early_return"]), 0.10)
        self.assertAlmostEqual(float(first["late_return"]), 0.10)
        self.assertAlmostEqual(float(first["half_gap"]), 0.0)
        second = found.loc[found["decision_at"].eq(pd.Timestamp("2024-01-02 11:35"))].iloc[0]
        self.assertAlmostEqual(float(second["early_return"]), 0.10)
        self.assertAlmostEqual(float(second["late_return"]), 0.10)
        self.assertAlmostEqual(float(second["half_gap"]), 0.0)
        broken = found.loc[found["decision_at"].eq(pd.Timestamp("2024-01-02 14:00"))].iloc[0]
        self.assertTrue(np.isnan(broken["half_gap"]))
        self.assertNotEqual(broken["half_gap"], 0.0)


class QuarterReturnTests(unittest.TestCase):
    def test_four_contiguous_six_bar_returns_on_one_path(self):
        closes = np.full(24, 40.0)
        closes[5] = 50.0
        closes[11] = 40.0
        closes[17] = 60.0
        closes[23] = 30.0
        q1, q2, q3, q4 = quarter_returns(40.0, closes)
        self.assertAlmostEqual(q1, 50.0 / 40.0 - 1.0)
        self.assertAlmostEqual(q2, 40.0 / 50.0 - 1.0)
        self.assertAlmostEqual(q3, 60.0 / 40.0 - 1.0)
        self.assertAlmostEqual(q4, 30.0 / 60.0 - 1.0)
        bad = closes.copy()
        bad[17] = 0.0
        blank = quarter_returns(40.0, bad)
        self.assertTrue(all(np.isnan(item) for item in blank))
        self.assertFalse(any(item == 0.0 for item in blank))
        missing = closes.copy()
        missing[23] = np.nan
        blank = quarter_returns(40.0, missing)
        self.assertTrue(all(np.isnan(item) for item in blank))
        bars = _bars(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        bars.loc[0, "open"] = 40.0
        bars.loc[5, "close"] = 50.0
        bars.loc[11, "close"] = 40.0
        bars.loc[17, "close"] = 60.0
        bars.loc[23, "close"] = 30.0
        shape = window_shape_rows(bars, "TEST")
        shape["decision_at"] = pd.to_datetime(shape["decision_at"])
        first = shape.loc[shape["decision_at"].eq(pd.Timestamp("2024-01-02 11:30"))].iloc[0]
        self.assertAlmostEqual(float(first["q1"]), 0.25)
        self.assertAlmostEqual(float(first["q2"]), -0.20)
        self.assertAlmostEqual(float(first["q3"]), 0.50)
        self.assertAlmostEqual(float(first["q4"]), -0.50)

    def test_eight_contiguous_three_bar_returns_blank_together(self):
        closes = np.full(24, 10.0)
        points = (12.0, 15.0, 12.0, 18.0, 9.0, 18.0, 15.0, 30.0)
        for index, price in zip((2, 5, 8, 11, 14, 17, 20, 23), points):
            closes[index] = price
        terms = eighth_returns(10.0, closes)
        previous = 10.0
        for term, price in zip(terms, points):
            self.assertAlmostEqual(term, price / previous - 1.0)
            previous = price
        closes[14] = np.nan
        blank = eighth_returns(10.0, closes)
        self.assertEqual(len(blank), 8)
        self.assertTrue(all(np.isnan(item) for item in blank))
        self.assertFalse(any(item == 0.0 for item in blank))


class BarReturnTests(unittest.TestCase):
    def test_flat_then_up_has_the_expected_first_and_last_bar(self):
        closes = np.full(24, 100.0)
        closes[23] = 130.0
        terms = bar_returns(100.0, closes)
        self.assertEqual(len(terms), 24)
        self.assertAlmostEqual(terms[0], 0.0)
        self.assertTrue(all(term == 0.0 for term in terms[1:23]))
        self.assertAlmostEqual(terms[23], 0.30)
        bad = closes.copy()
        bad[10] = 0.0
        blank = bar_returns(100.0, bad)
        self.assertEqual(len(blank), 24)
        self.assertTrue(all(np.isnan(item) for item in blank))
        self.assertFalse(any(item == 0.0 for item in blank))
        bars = _bars(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
        bars.loc[0, "open"] = 100.0
        bars.loc[0:22, "close"] = 100.0
        bars.loc[23, "close"] = 130.0
        shape = window_shape_rows(bars, "TEST")
        shape["decision_at"] = pd.to_datetime(shape["decision_at"])
        first = shape.loc[shape["decision_at"].eq(pd.Timestamp("2024-01-02 11:30"))].iloc[0]
        self.assertAlmostEqual(float(first["b0"]), 0.0)
        self.assertAlmostEqual(float(first["b23"]), 0.30)


class PeerImpulseTests(unittest.TestCase):
    def test_mean_excludes_self_and_a_single_name_is_nan(self):
        decision = pd.to_datetime(
            ["2026-01-02 11:30", "2026-01-02 11:30", "2026-01-02 11:30", "2026-01-02 11:35"]
        )
        impulse = pd.Series([1.0, 3.0, 5.0, 9.0])
        peer = peer_impulse(impulse, decision)
        self.assertTrue(np.allclose(peer[:3], [4.0, 3.0, 2.0]))
        self.assertTrue(np.isnan(peer[3]))
        self.assertNotEqual(peer[3], 0.0)

    def test_nan_self_uses_the_finite_group_and_all_nan_stays_nan(self):
        decision = pd.to_datetime(["2026-01-02 11:30"] * 3 + ["2026-01-02 11:35"] * 2)
        impulse = pd.Series([1.0, np.nan, 5.0, np.nan, np.nan])
        peer = peer_impulse(impulse, decision)
        self.assertAlmostEqual(peer[0], 5.0)
        self.assertAlmostEqual(peer[1], 3.0)
        self.assertAlmostEqual(peer[2], 1.0)
        self.assertTrue(np.isnan(peer[3]) and np.isnan(peer[4]))
        self.assertFalse(np.any(peer[3:] == 0.0))


if __name__ == "__main__":
    unittest.main()
