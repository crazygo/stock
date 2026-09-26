import copy
import unittest
from pathlib import Path
import json
import numpy as np
import pandas as pd
from research.strategy_group_lab.data import aggregate, snapshot_segments, split_masks
from research.strategy_group_lab.portfolio import simulate
from research.strategy_group_lab.evaluate import route, metrics

CFG = json.loads((Path(__file__).parents[1]/"configs/v1.json").read_text())

class TimeContractTests(unittest.TestCase):
    def test_future_and_late_duplicate_cannot_change_features(self):
        sessions, records = [], []
        for d in pd.bdate_range("2026-01-05", periods=12):
            day = d.strftime("%Y-%m-%d")
            opening = pd.Timestamp(day+" 09:30", tz="America/New_York").tz_convert("UTC")
            closing = opening+pd.Timedelta(minutes=390)
            sessions.append({"session_date": day, "open_at": opening.isoformat(), "close_at": closing.isoformat()})
            for at in pd.date_range(opening, closing, freq="5min", inclusive="left"):
                end = at+pd.Timedelta(minutes=5)
                records.append({"_start": at, "_end": end, "_available": end+pd.Timedelta(seconds=1),
                                "open": 100., "high": 101., "low": 99., "close": 100., "volume": 1000.,
                                "turnover": 100000., "price_basis": "NONE"})
        raw = pd.DataFrame(records)
        cutoff = pd.Timestamp(sessions[-1]["open_at"])+pd.Timedelta(minutes=120)
        a = snapshot_segments(raw, {}, 11, sessions, cutoff)
        future = raw.iloc[[0]].copy()
        future["_available"] = cutoff+pd.Timedelta(days=1)
        future[["open", "high", "low", "close"]] = [500, 900, 400, 800]
        modified = raw.copy()
        modified.loc[modified._end > cutoff, ["open", "high", "low", "close"]] *= 10
        b = snapshot_segments(pd.concat([modified, future]), {}, 11, sessions, cutoff)
        self.assertEqual(a[1], b[1])
        for x, y in zip(a[0], b[0]):
            np.testing.assert_array_equal(x[0], y[0])
            np.testing.assert_array_equal(x[1], y[1])

    def test_coarse_tail_retains_available_information(self):
        a = np.tile([100, 102, 99, 101, 10, 1000], (24, 1)).astype(float)
        mask = np.ones(24, bool)
        mask[-1] = False
        z, m, duration = aggregate(a, mask, 12)
        self.assertEqual(duration, [60, 55])
        self.assertTrue(m.all())
        self.assertEqual(z[1, 4], 110)
        mask[15] = False
        self.assertFalse(aggregate(a, mask, 12)[1][1])

    def test_purge_depends_on_outcome_availability(self):
        rows = pd.DataFrame({"date": ["2026-04-10", "2026-04-30", "2026-05-20", "2026-06-25", "2026-07-01"],
                             "cutoff_at": [f"{d}T15:30:00+00:00" for d in ["2026-04-10", "2026-04-30", "2026-05-20", "2026-06-25", "2026-07-01"]]})
        labels = pd.DataFrame({"status": ["mature"]*5, "label_available_at":
                              [f"{d}T16:00:00+00:00" for d in ["2026-04-30", "2026-05-02", "2026-06-20", "2026-07-10", "2026-08-01"]]})
        fit, cal, test = split_masks(rows, labels, CFG["folds"][0])
        self.assertEqual(np.flatnonzero(fit).tolist(), [0])
        self.assertEqual(np.flatnonzero(cal).tolist(), [2])
        self.assertEqual(np.flatnonzero(test).tolist(), [4])

class PortfolioTests(unittest.TestCase):
    def fixture(self):
        grid = pd.DataFrame({"start": pd.date_range("2026-07-01T13:30:00Z", periods=78, freq="5min")})
        grid["end"] = grid.start+pd.Timedelta(minutes=5)
        grid["date"] = "2026-07-01"
        rows = pd.DataFrame({"row_id": [0, 1], "symbol": ["A", "B"], "date": ["2026-07-01"]*2, "pos": [25, 25]})
        labels = pd.DataFrame({"row_id": [0, 1], "expectation_id": ["d3_r5"]*2, "status": ["mature"]*2,
                               "reason": [None]*2, "hit": [1, 1], "hit_minutes": [5, 5]})
        signals = pd.DataFrame({"row_id": [0, 1], "p": [.8, .6], "binding_id": ["a", "b"], "expectation_id": ["d3_r5"]*2})
        paths = {s: np.tile([100, 105, 99, 100, 1, 1], (78, 1)) for s in ["A", "B"]}
        return grid, rows, labels, signals, paths

    def test_cost_and_cash_conservation(self):
        grid, rows, labels, signals, paths = self.fixture()
        result, detail = simulate(signals, rows, labels, paths, grid, CFG, end_date="2026-07-01")
        self.assertEqual(result["trade_n"], 2)
        self.assertEqual(detail["trades"][0]["shares"], 99)
        expected_cost = 2*99*205*.0006
        self.assertAlmostEqual(result["cost"], expected_cost)
        self.assertAlmostEqual(result["end_balance"], 100000+2*99*5-expected_cost)
        self.assertEqual(result["open_positions"], 0)

    def test_missing_selected_outcome_not_replaced_by_future_winner(self):
        grid, rows, labels, signals, paths = self.fixture()
        cfg = copy.deepcopy(CFG)
        cfg["portfolio"]["top_k"] = 1
        labels.loc[0, "status"] = "pending"
        result, detail = simulate(signals, rows, labels, paths, grid, cfg, end_date="2026-07-01")
        self.assertIsNone(result["net_return"])
        self.assertEqual(result["trade_n"], 0)
        self.assertEqual(detail["rejections"][0]["symbol"], "A")

    def test_duplicate_signals_do_not_double_buy(self):
        grid, rows, labels, signals, paths = self.fixture()
        signals = pd.concat([signals, signals.iloc[[0]]], ignore_index=True)
        result, detail = simulate(signals, rows, labels, paths, grid, CFG, end_date="2026-07-01")
        self.assertEqual(result["trade_n"], 2)
        self.assertEqual(len({t["symbol"] for t in detail["trades"]}), 2)

class RouterTests(unittest.TestCase):
    def test_empty_calendar_blocks_are_not_independent_evidence(self):
        dates = pd.bdate_range("2026-07-01", periods=39).strftime("%Y-%m-%d").tolist()
        f = pd.DataFrame({"date": dates[-17:], "symbol": ["A"]*17, "status": ["mature"]*17,
                          "p": [.2]*17, "raw_p": [.2]*17, "b0": [.5]*17, "hit": [0]*17})
        result = metrics(f, dates, 3, CFG)
        self.assertIsNone(result["p_value"])
        self.assertEqual(result["ci_status"], "insufficient_calendar_blocks")

    def test_unmatured_future_labels_never_change_week_selection(self):
        cfg = copy.deepcopy(CFG)
        cfg["router"].update(minimum_candidates=1, minimum_dates=1)
        cfg["folds"] = [{"outer_start": "2026-07-01", "outer_end": "2026-07-17"}]
        grid = pd.DataFrame({"date": pd.bdate_range("2026-06-01", "2026-07-17").strftime("%Y-%m-%d")})
        frame = pd.DataFrame({"row_id": [0, 1], "p": [.8, .8], "b0": [.5, .5], "hit": [1, 0],
                              "date": ["2026-07-01", "2026-07-06"], "status": ["mature"]*2,
                              "label_available_at": ["2026-07-03T20:00:01Z", "2026-07-20T20:00:01Z"],
                              "net_proxy": [.05, -.5], "holding_minutes_proxy": [390, 390]})
        bindings = {"a": {"algorithm": "lgbm", "expectation": "d3_r5"}}
        _, schedule1 = route({"a": frame}, bindings, grid, cfg)
        frame.loc[1, ["hit", "net_proxy"]] = [1, .9]
        _, schedule2 = route({"a": frame}, bindings, grid, cfg)
        self.assertEqual(schedule1, schedule2)
        self.assertEqual(schedule1[0]["selected"], [])
        self.assertEqual(schedule1[1]["selected"][0]["binding_id"], "a")

if __name__ == "__main__":
    unittest.main()
