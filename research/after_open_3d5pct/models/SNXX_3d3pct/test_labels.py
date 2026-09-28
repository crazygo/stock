"""Boundary checks for the hourly 1170-minute window."""
from datetime import datetime, timedelta
from unittest import TestCase, main

import pandas as pd

from research.after_open_3d5pct.models.SNXX_3d3pct.labels import build_rows, horizon_slice, regular_bars


def _session(day: datetime, high: float = 10, close_1130: float = 10) -> list[dict]:
    rows = []
    clock = datetime(day.year, day.month, day.day, 9, 30)
    for step in (60, 60, 60, 60, 60, 60, 30, 60):
        price = close_1130 if clock.hour == 11 and clock.minute == 30 else 10
        rows.append({
            "time_key": clock.strftime("%Y-%m-%d %H:%M:%S"),
            "open": 10.0, "high": 10.4, "low": 9.8, "close": price, "volume": 100.0 + clock.hour,
        })
        clock += timedelta(minutes=step)
    return rows


class HourlyWindowTest(TestCase):
    def test_window_ends_on_third_session_1030_bar(self):
        start = datetime(2026, 8, 24)
        frame = pd.DataFrame(sum((_session(start + timedelta(days=i)) for i in range(4)), []))
        regular = regular_bars(frame)
        pos = int(regular.index[regular["start"].eq(pd.Timestamp("2026-08-24 11:30:00"))][0])
        window = horizon_slice(regular, pos)
        self.assertEqual(str(window["start"].iloc[-1]), "2026-08-27 10:30:00")
        self.assertNotIn("2026-08-27 11:30:00", set(window["start"].astype(str)))
        self.assertEqual(int(window["minutes"].sum()), 1170)

    def test_missing_final_bar_is_immature(self):
        start = datetime(2026, 8, 24)
        raw = sum((_session(start + timedelta(days=i)) for i in range(4)), [])
        raw = [r for r in raw if not r["time_key"].startswith("2026-08-27")]
        regular = regular_bars(pd.DataFrame(raw))
        pos = int(regular.index[regular["start"].eq(pd.Timestamp("2026-08-24 11:30:00"))][0])
        self.assertIsNone(horizon_slice(regular, pos))

    def test_entry_bar_close_does_not_change_features(self):
        days = [datetime(2026, 6, 1) + timedelta(days=i) for i in range(30)]
        days = [d for d in days if d.weekday() < 5][:12]
        watched = days[8].strftime("%Y-%m-%d")
        base = pd.DataFrame(sum((_session(d) for d in days), []))
        changed = base.copy()
        changed.loc[changed["time_key"].eq(watched + " 11:30:00"), "close"] = 999
        left, _ = build_rows(base)
        right, _ = build_rows(changed)
        self.assertIn(watched, set(left.session_date))
        cols = [c for c in left.columns if c.startswith(("r_", "morning", "overnight", "range", "volume"))]
        a = left.loc[left.session_date.eq(watched), cols].iloc[0]
        b = right.loc[right.session_date.eq(watched), cols].iloc[0]
        self.assertTrue((a.to_numpy() == b.to_numpy()).all())


if __name__ == "__main__":
    main()
