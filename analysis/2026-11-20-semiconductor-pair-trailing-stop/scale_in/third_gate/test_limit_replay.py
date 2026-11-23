"""Synthetic path and accounting tests independent of market replay results."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest

from limit_replay import run


FLAT = (100, 100, 100, 100)
PATHS = ("OLHC", "OHLC")


def account(bars, path="OLHC", signals=(2,), **kwargs):
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    times = [start+timedelta(minutes=5*i) for i in range(len(bars))]
    data = dict(tape={"SOXL":bars}, times=times, days=["2026-09-01"]*len(times))
    lots = [dict(stage=3, cycle=k+1, symbol="SOXL", decision_i=i-1, entry_i=i) for k,i in enumerate(signals)]
    return run(SimpleNamespace(data=data), path, lots, **kwargs)


class LimitReplayTests(unittest.TestCase):
    def test_opening_improvement_and_target_use_actual_fill(self):
        for path in PATHS:
            r = account([FLAT, FLAT, (95,104,94,101)], path)
            t = r["trades"][0]
            self.assertEqual((t["entry_price"], t["fill_leg"], t["reason"]), (95,"O","target"))
            self.assertAlmostEqual(t["exit_price"], 95*1.08)
            self.assertAlmostEqual(r["net"], 43)

    def test_touch_fills_and_pre_fill_high_cannot_trigger_exit(self):
        bars = [FLAT, (104,120,100,120), (102,110,100,100.5)]
        low_first = account(bars)
        self.assertEqual(low_first["events"][0]["limit_price"], 100)
        self.assertEqual(low_first["trades"][0]["reason"], "target")
        high_first = account(bars, "OHLC")
        self.assertEqual(high_first["trades"][0]["fill_leg"], "H-L")
        self.assertEqual(high_first["trades"][0]["reason"], "open")
        self.assertEqual(high_first["fees"], 2.5)
        self.assertAlmostEqual(high_first["net"], .5)

    def test_missed_limit_expires_even_if_next_open_below_limit(self):
        for path in PATHS:
            bars = [FLAT, (104,121,100,120), (105,109,100.01,103), (89,101,88,100)]
            r = account(bars, path)
            self.assertEqual((r["triggers"],r["unfilled"],r["buys"],r["fees"]), (1,1,0,0))
            self.assertEqual(r["events"][0]["status"], "unfilled")

    def test_stop_on_same_segment_after_fill(self):
        for path in PATHS:
            r = account([FLAT,FLAT,(103,105,95,98)], path)
            t = r["trades"][0]
            self.assertEqual((t["entry_i"],t["exit_i"],t["exit_price"]), (2,2,97))
            self.assertAlmostEqual(r["net"], -23)

    def test_later_gap_stop_crosses_at_its_open(self):
        for path in PATHS:
            r = account([FLAT]*3+[(90,94,89,91)], path)
            t = r["trades"][0]
            self.assertEqual((t["exit_i"],t["exit_price"],t["exit_at_open"]), (3,90,True))

    def test_trailing_stop_can_close_with_positive_net(self):
        for path in PATHS:
            r = account([FLAT]*3+[(106,106,102,103)], path)
            self.assertEqual((r["wins"],r["targets"],r["trades"][0]["reason"]), (1,0,"stop"))
            self.assertAlmostEqual(r["net"],6*(106*.97-100)-5)

    def test_account_cash_and_exclusive_position(self):
        r = account([FLAT]*6, signals=(2,5))
        self.assertEqual([e["status"] for e in r["events"]], ["filled","occupied"])
        r = account([FLAT]*3, initial=602)
        self.assertEqual((r["cash_blocked"],r["fees"],r["open"]), (1,0,0))
        r = account([FLAT]*5+[(96,97,96,96)], signals=(2,5))
        self.assertEqual([e["status"] for e in r["events"]], ["filled","filled"])
        self.assertEqual((r["sells"],r["open"]), (1,1))

    def test_tape_suffix_cannot_change_a_past_order(self):
        bars = [FLAT, FLAT, (102,105,99,100)]
        early = account(bars)
        later = account(bars+[(90,100,88,94)])
        self.assertEqual(early["events"], later["events"])


if __name__ == "__main__":
    unittest.main()
