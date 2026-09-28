"""The paired window ends at 11:35 three sessions later and ignores the entry bar's close."""
from datetime import datetime, timedelta
from unittest import TestCase, main

import pandas as pd

from research.after_open_3d5pct.models.SOXS_SOXX_3d5pct.labels import build_rows, horizon_slice, regular_bars


def _frame(symbol: str, start: datetime, sessions: int, entry_close: float = 10.0) -> pd.DataFrame:
    rows = []
    for offset in range(sessions):
        day = start + timedelta(days=offset)
        clock = datetime(day.year, day.month, day.day, 9, 30)
        for _ in range(78):
            end = clock + timedelta(minutes=5)
            close = entry_close if clock.hour == 11 and clock.minute == 35 else 10.0
            rows.append({
                "session_type": "regular",
                "start_at_et": clock.strftime("%Y-%m-%dT%H:%M:%S-04:00"),
                "end_at_et": end.strftime("%Y-%m-%dT%H:%M:%S-04:00"),
                "open": 10.0, "high": 10.4, "low": 9.8, "close": close, "volume": 100.0 + clock.hour,
            })
            clock = end
    return pd.DataFrame(rows)


class PairLabelTest(TestCase):
    def test_window_is_234_regular_bars(self):
        frame = _frame("SOXX", datetime(2026, 8, 24), 4)
        bars = regular_bars(frame)
        pos = int(bars.index[bars["start"].eq(pd.Timestamp("2026-08-24 11:35:00"))][0])
        window = horizon_slice(bars, pos)
        self.assertEqual(len(window), 234)
        self.assertEqual(int(window["minutes"].sum()), 1170)
        self.assertEqual(str(window["end"].iloc[-1]), "2026-08-27 11:35:00")

    def test_entry_close_is_not_a_feature(self):
        block = datetime(2026, 6, 8)
        base_x = _frame("SOXS", block, 8)
        base_q = _frame("SOXX", block, 8)
        changed = base_x.copy()
        changed.loc[changed["start_at_et"].str.startswith("2026-06-12T11:35"), "close"] = 999
        left, _ = build_rows(base_x, base_q)
        right, _ = build_rows(changed, base_q)
        self.assertIn("2026-06-12", set(left.session_date))
        cols = [c for c in left.columns if c.startswith(("SOXS_", "SOXX_", "morning_", "r12_"))]
        a = left.loc[left.session_date.eq("2026-06-12"), cols].iloc[0]
        b = right.loc[right.session_date.eq("2026-06-12"), cols].iloc[0]
        self.assertTrue((a.to_numpy() == b.to_numpy()).all())


if __name__ == "__main__":
    main()
