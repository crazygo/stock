"""The premarket high is not the label. Entry is the 09:30 premarket close."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.potential_v1.run import impulse_from_path
from research.after_open_3d5pct.models.premarket_tail_v1.rows import FEATURES, rows_from_bars
from research.after_open_3d5pct.models.premarket_tail_v1.run import (
    beats_everyday,
    decision_cutoff,
    sign_metrics,
)
from research.after_open_3d5pct.models.weekly_scale.run import matured_before


def _block(start: pd.Timestamp, n: int, session: str, price: float) -> pd.DataFrame:
    starts = pd.date_range(start, periods=n, freq="5min")
    return pd.DataFrame({
        "start": starts,
        "end": starts + pd.Timedelta(minutes=5),
        "session_type": session,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
    })


def _regular(day: str, price: float = 100.0, n: int = 78) -> pd.DataFrame:
    return _block(pd.Timestamp(f"{day} 09:30"), n, "regular", price)


def _overnight(day: str, price: float = 100.0) -> pd.DataFrame:
    return _block(pd.Timestamp(f"{day} 03:30"), 6, "overnight", price)


def _premarket(day: str, price: float = 100.0) -> pd.DataFrame:
    return _block(pd.Timestamp(f"{day} 09:00"), 6, "pre_market", price)


def _panel(days: list[str], price: float = 100.0, regular_n: int = 78) -> pd.DataFrame:
    frames = []
    for day in days:
        frames.extend([_overnight(day, price), _premarket(day, price), _regular(day, price, regular_n)])
    return pd.concat(frames, ignore_index=True)


class LabelTests(unittest.TestCase):
    def test_premarket_spike_does_not_flip_the_label_and_entry_is_the_tail_close(self):
        days = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        bars = _panel(days)
        tail = bars["end"].eq(pd.Timestamp("2026-01-06 09:30"))
        bars.loc[tail, "close"] = 101.0
        bars.loc[tail, "high"] = 120.0
        regular = bars["session_type"].eq("regular") & bars["start"].dt.strftime("%Y-%m-%d").eq("2026-01-06")
        bars.loc[regular, "open"] = 102.0
        bars.loc[regular, "high"] = 104.0
        built = rows_from_bars("TEST", bars)
        row = built.loc[built["session_date"].eq("2026-01-06")].iloc[0]
        self.assertEqual(row["y_3d_5pct"], 0.0)
        self.assertEqual(row["entry"], 101.0)
        self.assertEqual(row["regular_open"], 102.0)
        self.assertNotEqual(row["entry"], row["regular_open"])

    def test_first_regular_bar_touch_labels_one(self):
        days = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        bars = _panel(days, price=100.0)
        first = bars["start"].eq(pd.Timestamp("2026-01-06 09:30"))
        bars.loc[first, "high"] = 105.0
        built = rows_from_bars("TEST", bars)
        row = built.loc[built["session_date"].eq("2026-01-06")].iloc[0]
        self.assertEqual(row["entry"], 100.0)
        self.assertEqual(row["y_3d_5pct"], 1.0)
        self.assertEqual(pd.Timestamp(row["label_end"]), pd.Timestamp("2026-01-08 16:00"))

    def test_horizon_is_234_regular_bars_and_the_bar_after_it_does_not_count(self):
        days = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"]
        bars = _panel(days, price=100.0)
        later = bars["start"].eq(pd.Timestamp("2026-01-09 09:30"))
        bars.loc[later, "high"] = 150.0
        built = rows_from_bars("TEST", bars)
        row = built.loc[built["session_date"].eq("2026-01-06")].iloc[0]
        self.assertEqual(row["y_3d_5pct"], 0.0)
        self.assertEqual(pd.Timestamp(row["label_end"]), pd.Timestamp("2026-01-08 16:00"))
        short = _panel(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])
        short = pd.concat([
            short.loc[~((short["session_type"].eq("regular")) & (short["start"].dt.strftime("%Y-%m-%d").eq("2026-01-08")))],
            _regular("2026-01-08", n=77),
        ], ignore_index=True)
        self.assertFalse((rows_from_bars("TEST", short)["session_date"] == "2026-01-06").any())

    def test_last_bar_of_the_horizon_does_count(self):
        days = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        bars = _panel(days, price=100.0)
        last = bars["start"].eq(pd.Timestamp("2026-01-08 15:55"))
        bars.loc[last, "high"] = 105.0
        row = rows_from_bars("TEST", bars).loc[lambda frame: frame["session_date"].eq("2026-01-06")].iloc[0]
        self.assertEqual(row["y_3d_5pct"], 1.0)


class FeatureTests(unittest.TestCase):
    def test_todays_regular_close_is_not_a_feature_and_post_market_is_outside_the_path(self):
        days = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        bars = _panel(days, price=100.0)
        night = bars["session_type"].eq("overnight") & bars["start"].dt.strftime("%Y-%m-%d").eq("2026-01-06")
        bars.loc[night, "close"] = 103.0
        original = rows_from_bars("TEST", bars)
        changed = bars.copy()
        regular_today = changed["session_type"].eq("regular") & changed["start"].dt.strftime("%Y-%m-%d").eq("2026-01-06")
        changed.loc[regular_today, "close"] = 1.0
        post = _block(pd.Timestamp("2026-01-05 18:00"), 1, "post_market", 1.0)
        with_post = pd.concat([changed, post], ignore_index=True)
        rebuilt = rows_from_bars("TEST", with_post)
        left = original.loc[original["session_date"].eq("2026-01-06")].iloc[0]
        right = rebuilt.loc[rebuilt["session_date"].eq("2026-01-06")].iloc[0]
        for name in FEATURES:
            self.assertTrue(np.isfinite(left[name]))
            self.assertAlmostEqual(float(left[name]), float(right[name]), places=8)
        self.assertNotIn("regular_open", FEATURES)

    def test_sunday_evening_joins_monday_and_a_short_night_stays_blank(self):
        days = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
        bars = _panel(days, price=100.0)
        sunday = _block(pd.Timestamp("2026-01-04 20:00"), 1, "overnight", 80.0)
        bars = pd.concat([bars, sunday], ignore_index=True)
        row = rows_from_bars("TEST", bars).loc[lambda frame: frame["session_date"].eq("2026-01-05")].iloc[0]
        self.assertAlmostEqual(float(row["overnight_return"]), 100.0 / 80.0 - 1.0, places=8)
        self.assertGreater(int(row["n_overnight"]), 6)
        sunday = bars["start"].eq(pd.Timestamp("2026-01-04 20:00"))
        monday_night = bars["session_type"].eq("overnight") & bars["start"].dt.strftime("%Y-%m-%d").eq("2026-01-05")
        thin = bars.loc[~sunday].drop(index=bars.loc[monday_night].index[:1])
        blank = rows_from_bars("TEST", thin).loc[lambda frame: frame["session_date"].eq("2026-01-05")].iloc[0]
        self.assertTrue(np.isnan(blank["overnight_return"]))
        self.assertEqual(int(blank["n_overnight"]), 5)

    def test_path_impulse_matches_the_shared_function(self):
        days = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        bars = _panel(days, price=100.0)
        bars.loc[bars["end"].eq(pd.Timestamp("2026-01-06 09:30")), "close"] = 110.0
        row = rows_from_bars("TEST", bars).loc[lambda frame: frame["session_date"].eq("2026-01-06")].iloc[0]
        night = _overnight("2026-01-06")
        pre = _premarket("2026-01-06")
        pre.loc[pre.index[-1], "close"] = 110.0
        closes = pd.concat([night, pre], ignore_index=True).sort_values("end")["close"].to_numpy(float)
        self.assertAlmostEqual(float(row["path_impulse"]), float(impulse_from_path(100.0, closes)), places=8)

    def test_missing_tail_or_regular_open_bar_skips_the_day(self):
        days = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
        bars = _panel(days)
        no_tail = bars.loc[~bars["end"].eq(pd.Timestamp("2026-01-06 09:30"))].copy()
        self.assertFalse((rows_from_bars("TEST", no_tail)["session_date"] == "2026-01-06").any())
        no_open = bars.loc[~bars["start"].eq(pd.Timestamp("2026-01-06 09:30"))].copy()
        self.assertFalse((rows_from_bars("TEST", no_open)["session_date"] == "2026-01-06").any())


class RuleTests(unittest.TestCase):
    def test_purge_keeps_wednesday_and_drops_thursday_and_friday(self):
        frame = pd.DataFrame({
            "session_date": ["2026-09-16", "2026-09-17", "2026-09-18"],
            "label_end": [
                pd.Timestamp("2026-09-18 16:00"),
                pd.Timestamp("2026-09-21 16:00"),
                pd.Timestamp("2026-09-22 16:00"),
            ],
            "y_3d_5pct": [1.0, 1.0, 1.0],
        })
        kept = matured_before(frame, ["2026-09-16", "2026-09-17", "2026-09-18"], decision_cutoff(["2026-09-21"]))
        self.assertEqual(list(kept["session_date"]), ["2026-09-16"])

    def test_sign_rule_ignores_missing_and_flat_impulse(self):
        frame = pd.DataFrame({
            "path_impulse": [np.nan, 0.1, -0.2, 0.0],
            "y_3d_5pct": [1.0, 1.0, 0.0, 1.0],
        })
        stats = sign_metrics(frame)
        self.assertEqual(stats["buys"], 1)
        self.assertEqual(stats["precision"], 1.0)

    def test_everyday_beat_needs_fifteen_buys_and_a_strict_lift(self):
        self.assertFalse(beats_everyday({"buys": 15, "precision": 0.5, "base_rate": 0.5}))
        self.assertFalse(beats_everyday({"buys": 14, "precision": 0.9, "base_rate": 0.4}))
        self.assertTrue(beats_everyday({"buys": 15, "precision": 0.51, "base_rate": 0.5}))
        self.assertFalse(beats_everyday({"buys": 0, "precision": None, "base_rate": 0.4}))


if __name__ == "__main__":
    unittest.main()
