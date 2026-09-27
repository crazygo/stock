"""Synthetic checks for the separate manual-action contract."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from research.after_open_3d5pct.policy_replay_v1 import (
    PolicyConfig, instant, replay, simulate_entry, summarize,
)


ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def fixture(days=6):
    starts = []
    origin = datetime(2026, 9, 21, 9, 30, tzinfo=ET)
    for day in range(days):
        at = origin + timedelta(days=day)
        for j in range(78):
            starts.append((at + timedelta(minutes=5*j)).astimezone(UTC))
    bars = {at: {"session_type": "regular", "price_basis": "NONE",
                 "open": 100., "high": 100.5, "low": 99.5, "close": 100.}
            for at in starts}
    return starts, bars


def decision(day, hour, probability=.35, target=1):
    cutoff = datetime(2026, 9, day, *map(int, hour.split(':')), tzinfo=ET).astimezone(UTC)
    return {"sample_id": f"ABC|{cutoff.isoformat()}", "symbol": "ABC",
            "session_date": cutoff.astimezone(ET).date().isoformat(),
            "cutoff_et": hour, "cutoff_at": cutoff.isoformat(),
            "decision_at": (cutoff + timedelta(seconds=30)).isoformat(),
            "probability": probability, "target": target}


class ReplayTests(unittest.TestCase):
    def test_delayed_first_open_and_touch_ambiguity(self):
        starts, bars = fixture()
        entry = instant("2026-09-21T11:35:00-04:00")
        bars[entry]["high"] = 106
        bars[entry]["low"] = 97
        out = simulate_entry(decision(21, "11:30"), bars, starts,
                             PolicyConfig(), datetime(2026, 10, 1, tzinfo=UTC))
        self.assertEqual(out["entry_at"], entry.isoformat())
        self.assertEqual(out["status"], "target_touch_proxy")
        self.assertEqual(out["exit_price_proxy"], 105)
        self.assertEqual(out["holding_regular_minutes"], 5)
        self.assertTrue(out["mae_order_uncertain"])
        self.assertAlmostEqual(out["mae_lower_bound"], -.03)
        self.assertEqual(out["mae_upper_bound"], 0)
        self.assertLess(out["net_return"], .05)

    def test_three_cutoffs_threshold_and_cross_date_held_state(self):
        starts, bars = fixture()
        hit = instant("2026-09-23T10:00:00-04:00")
        bars[hit]["high"] = 105
        rows = [decision(21, "11:30", .349), decision(21, "12:30", .9),
                decision(21, "13:30", .35), decision(22, "11:30", .99),
                decision(23, "13:30", .5)]
        ledger, trades = replay(rows, {"ABC": bars}, starts, PolicyConfig(),
                                datetime(2026, 10, 1, tzinfo=UTC))
        self.assertEqual([r["action"] for r in ledger],
                         ["below_threshold", "scored_only", "recommend_entry",
                          "observe_held", "recommend_entry"])
        self.assertEqual(len(trades), 2)
        counts = summarize(rows, ledger, trades)
        self.assertEqual(counts["predictions"], 4)
        self.assertEqual(counts["recommendations"], 2)
        self.assertEqual(counts["mature"], 4)
        self.assertEqual(counts["pool_hit"], 4)

    def test_missing_entry_and_path_never_choose_later_success(self):
        starts, bars = fixture()
        row = decision(21, "11:30")
        entry = instant("2026-09-21T11:35:00-04:00")
        bars.pop(entry)
        out = simulate_entry(row, bars, starts, PolicyConfig(),
                             datetime(2026, 10, 1, tzinfo=UTC))
        self.assertEqual(out["status"], "entry_bar_missing_or_invalid")
        bars[entry] = {"session_type": "regular", "price_basis": "NONE",
                       "open": 100., "high": 100.5, "low": 99.5, "close": 100.}
        missing = instant("2026-09-22T11:00:00-04:00")
        bars.pop(missing)
        later_hit = instant("2026-09-23T10:00:00-04:00")
        bars[later_hit]["high"] = 106
        out = simulate_entry(row, bars, starts, PolicyConfig(),
                             datetime(2026, 10, 1, tzinfo=UTC))
        self.assertEqual(out["status"], "indeterminate_missing_path")
        self.assertNotIn("target_hit", out)

    def test_pending_and_denominators(self):
        starts, bars = fixture()
        row = decision(21, "11:30", target=None)
        out = simulate_entry(row, bars, starts, PolicyConfig(),
                             datetime(2026, 9, 21, 18, tzinfo=UTC))
        self.assertEqual(out["status"], "pending_window")
        ledger, trades = replay([row], {"ABC": bars}, starts, PolicyConfig(),
                                datetime(2026, 9, 21, 18, tzinfo=UTC))
        counts = summarize([row], ledger, trades)
        self.assertEqual(counts["predictions"], 1)
        self.assertEqual(counts["mature"], 0)
        self.assertEqual(counts["pending_or_missing_target"], 1)
        self.assertEqual(counts["resolved_operations"], 0)

    def test_time_mismatch_and_policy_version_guard(self):
        starts, bars = fixture()
        row = decision(21, "11:30")
        row["decision_at"] = row["cutoff_at"]
        self.assertEqual(simulate_entry(row, bars, starts, PolicyConfig(),
                                        datetime(2026, 10, 1, tzinfo=UTC))["status"],
                         "decision_time_mismatch")
        with self.assertRaises(ValueError):
            PolicyConfig(cutoffs_et=("11:30", "12:30", "13:30"))


if __name__ == "__main__":
    unittest.main()
