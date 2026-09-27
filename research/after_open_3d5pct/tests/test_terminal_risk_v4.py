"""Contract tests for full-window no-stop terminal valuation."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct.terminal_risk_v4 import (
    evaluate_complete_path, full_sample_es_loss, mixed_expected_net, net_return)
from research.after_open_3d5pct.train_terminal_risk_v4 import select_cross_section
from research.after_open_3d5pct.hourly_v4_visibility import visible_versions_v4
from research.after_open_3d5pct.run_once_terminal_risk_v4 import run_once


class TerminalRiskV4Tests(unittest.TestCase):
    def setUp(self):
        self.grid = pd.date_range("2026-01-05T14:30:00Z", periods=234, freq="5min")
        self.end = self.grid[-1] + pd.Timedelta(minutes=5)
        self.path = pd.DataFrame({
            "start_at": self.grid,
            "end_at": self.grid + pd.Timedelta(minutes=5),
            "available_at": self.grid + pd.Timedelta(minutes=5, seconds=1),
            "session_type": "regular", "price_basis": "NONE",
            "open": np.full(234, 100.0), "high": np.full(234, 101.0),
            "low": np.full(234, 99.0), "close": np.full(234, 100.0),
            "volume": np.full(234, 1000.0)})

    def score(self, frame=None, as_of=None):
        return evaluate_complete_path(self.path if frame is None else frame,
                                      entry_at=self.grid[0], entry_price=100.0,
                                      label_end_at=self.end, expected_starts=self.grid,
                                      as_of=self.end + pd.Timedelta(seconds=1)) if as_of is None else \
            evaluate_complete_path(self.path if frame is None else frame,
                                   entry_at=self.grid[0], entry_price=100.0,
                                   label_end_at=self.end, expected_starts=self.grid,
                                   as_of=as_of)

    def test_deep_interim_loss_followed_by_touch_is_success_without_penalty(self):
        f = self.path.copy()
        f.loc[20, ["open", "low", "close"]] = [75., 70., 75.]
        f.loc[20, "high"] = 100.
        f.loc[100, "high"] = 105.
        result = self.score(f)
        self.assertEqual((result["status"], result["target"]), ("mature", 1))
        self.assertAlmostEqual(result["gross_evaluation_return"], .05)
        self.assertAlmostEqual(result["net_evaluation_return"], float(net_return(.05, .0006, .0006)))

    def test_no_touch_positive_terminal_valuation(self):
        f = self.path.copy()
        f.loc[233, ["high", "close"]] = [103., 102.]
        result = self.score(f)
        self.assertEqual(result["target"], 0)
        self.assertAlmostEqual(result["gross_evaluation_return"], .02)
        self.assertGreater(result["net_evaluation_return"], 0)

    def test_no_touch_negative_terminal_valuation(self):
        f = self.path.copy()
        f.loc[233, ["low", "close"]] = [94., 95.]
        result = self.score(f)
        self.assertEqual(result["target"], 0)
        self.assertAlmostEqual(result["gross_evaluation_return"], -.05)

    def test_post_touch_twenty_percent_or_collapse_does_not_change_result(self):
        first = self.path.copy()
        first.loc[25, "high"] = 105.
        second = first.copy()
        second.loc[180, ["open", "high", "low", "close"]] = [120., 130., 10., 12.]
        self.assertEqual(self.score(first)["net_evaluation_return"],
                         self.score(second)["net_evaluation_return"])

    def test_full_window_maturity_missing_and_future_version(self):
        early = self.score(as_of=self.end - pd.Timedelta(seconds=1))
        self.assertEqual((early["status"], early["reason"]), ("pending", "full_window_not_elapsed"))
        missing = self.score(self.path.drop(index=100))
        self.assertEqual(missing["reason"], "missing_regular_bar")
        f = self.path.copy()
        f.loc[40, "available_at"] = self.end + pd.Timedelta(days=1)
        self.assertEqual(self.score(f)["reason"], "missing_regular_bar")
        future = self.path.iloc[[40]].copy()
        future.loc[:, "available_at"] = self.end + pd.Timedelta(days=1)
        future.loc[:, "high"] = 999.
        combined = pd.concat([self.path, future], ignore_index=True)
        self.assertEqual(self.score(combined)["target"], self.score()["target"])

    def test_mixed_basis_and_conflicting_same_version_rejected(self):
        adjusted = self.path.copy()
        adjusted.loc[10, "price_basis"] = "QFQ"
        self.assertEqual(self.score(adjusted)["reason"], "missing_or_mixed_price_basis")
        duplicate = self.path.iloc[[10]].copy()
        duplicate.loc[:, "high"] = 104.
        self.assertEqual(self.score(pd.concat([self.path, duplicate]))["reason"],
                         "conflicting_duplicate_version")

    def test_mixture_cost_and_full_sample_tail_denominator(self):
        p = np.array([0., .5, 1.])
        got = mixed_expected_net(p, np.array([.02, -.02, -.5]),
                                 touch_gross=.05, buy_cost=.0006, sell_cost=.0006)
        touch = float(net_return(.05, .0006, .0006))
        self.assertAlmostEqual(got[0], float(net_return(.02, .0006, .0006)))
        self.assertAlmostEqual(got[1], .5 * touch + .5 * float(net_return(-.02, .0006, .0006)))
        self.assertAlmostEqual(got[2], touch)
        # A 5% tail over 21 *full* trades contains the worst trade and 5% of
        # the second worst, regardless of which trades touched.
        returns = np.array([-.2, -.1] + [.05] * 19)
        self.assertAlmostEqual(full_sample_es_loss(returns, .05), (.2 + .05 * .1) / 1.05)

    def test_selection_is_within_each_date_hour(self):
        frame = pd.DataFrame({"session_date": ["2026-01-05"] * 4 + ["2026-01-06"] * 4,
                              "cutoff_et": ["10:30"] * 8,
                              "symbol": list("ABCDEFGH"),
                              "p_touch": [.9, .8, .7, .6, .5, .4, .3, .2],
                              "expected_net": [.01] * 8,
                              "failure_q10_net": [-.05] * 8})
        scenario = {"opportunity_budget_fraction": .25, "predicted_expected_net_min": 0,
                    "predicted_failure_q10_net_min": -.15}
        chosen = select_cross_section(frame, scenario)
        self.assertEqual(frame.loc[chosen, "symbol"].tolist(), ["A", "E"])

    def test_observed_receipt_version_beats_assumed_null_but_late_or_future_does_not(self):
        cutoff = pd.Timestamp("2026-09-25T14:30:00Z")
        knowledge = cutoff + pd.Timedelta(seconds=20)
        old = self.path.iloc[[0]].copy()
        old.loc[:, "start_at"] = cutoff - pd.Timedelta(minutes=5)
        old.loc[:, "end_at"] = cutoff
        old.loc[:, "available_at"] = cutoff + pd.Timedelta(seconds=1)
        old["received_at"] = pd.to_datetime(pd.Series([pd.NaT], index=old.index), utc=True)
        observed = old.copy()
        observed.loc[:, "close"] = 100.5
        observed.loc[:, "high"] = 101.5
        observed.loc[:, "received_at"] = cutoff + pd.Timedelta(seconds=5)
        late = observed.copy()
        late.loc[:, "close"] = 100.8
        late.loc[:, "received_at"] = cutoff + pd.Timedelta(seconds=21)
        future = observed.copy()
        future.loc[:, "start_at"] = cutoff
        future.loc[:, "end_at"] = cutoff + pd.Timedelta(minutes=5)
        future.loc[:, "close"] = 999.
        raw = pd.concat([old, observed, late, future], ignore_index=True)
        visible = visible_versions_v4(raw, cutoff, knowledge)
        self.assertEqual(len(visible), 1)
        self.assertAlmostEqual(float(visible.iloc[0].close), 100.5)
        raw.loc[3, "high"] = 9999.
        self.assertAlmostEqual(float(visible_versions_v4(raw, cutoff, knowledge).iloc[0].close), 100.5)

    def test_future_asof_rejected_before_model_or_snapshot_access(self):
        with self.assertRaisesRegex(ValueError, "future_requested_at"):
            run_once(Path("/missing/model"), Path("/missing/output"),
                     datetime.now(timezone.utc) + timedelta(days=1),
                     historical_parameter=True)


if __name__ == "__main__":
    unittest.main()
