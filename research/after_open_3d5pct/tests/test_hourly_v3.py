from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo
import json
import unittest

import numpy as np
import pandas as pd

from research.after_open_3d5pct.hourly_v3_time import resolve_time
from research.after_open_3d5pct.hourly_v3_snapshot import source_watermark, visible_versions
from research.after_open_3d5pct.hourly_v3_data import build_bundle
from research.after_open_3d5pct.train_hourly_v3 import _select_threshold
from research.after_open_3d5pct.evaluate_forward_hourly_v3 import evaluate_row, REPO
from research.after_open_3d5pct.run_once_v3 import classify_action

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


class HourlyTimeTest(unittest.TestCase):
    def setUp(self):
        self.calendar = {"calendar_id": "fixture", "sessions": [
            {"session_date": "2026-09-24", "open_at": "2026-09-24T13:30:00+00:00",
             "close_at": "2026-09-24T20:00:00+00:00"},
            {"session_date": "2026-11-27", "open_at": "2026-11-27T14:30:00+00:00",
             "close_at": "2026-11-27T18:00:00+00:00"}]}

    def test_only_floors_to_past_supported_cutoff(self):
        a = resolve_time("2026-09-24T14:50:00-04:00", self.calendar)
        b = resolve_time("2026-09-24T13:12:00-04:00", self.calendar)
        self.assertEqual(a.effective_cutoff, "2026-09-24T18:30:00+00:00")
        self.assertEqual(b.effective_cutoff, "2026-09-24T16:30:00+00:00")
        self.assertEqual(a.state, "expired")

    def test_before_first_and_closed_never_cross_day(self):
        early = resolve_time("2026-09-24T10:00:00-04:00", self.calendar)
        closed = resolve_time("2026-09-25T11:30:00-04:00", self.calendar)
        half = resolve_time("2026-11-27T15:00:00-05:00", self.calendar)
        self.assertIsNone(early.effective_cutoff)
        self.assertEqual(early.next_legal_cutoff, "2026-09-24T14:30:00+00:00")
        self.assertIsNone(closed.effective_cutoff)
        self.assertEqual(half.effective_cutoff, "2026-11-27T17:30:00+00:00")
        self.assertEqual(half.state, "expired")

    def test_requires_explicit_timezone(self):
        with self.assertRaises(ValueError):
            resolve_time("2026-09-24T14:50:00", self.calendar)

    def test_cross_year_never_reuses_previous_session_signal(self):
        calendar = {"calendar_id": "cross_year", "sessions": [
            {"session_date": "2026-12-31", "open_at": "2026-12-31T14:30:00+00:00",
             "close_at": "2026-12-31T21:00:00+00:00"},
            {"session_date": "2027-01-04", "open_at": "2027-01-04T14:30:00+00:00",
             "close_at": "2027-01-04T21:00:00+00:00"}]}
        before = resolve_time("2027-01-04T10:00:00-05:00", calendar)
        after = resolve_time("2027-01-04T13:12:00-05:00", calendar)
        self.assertIsNone(before.effective_cutoff)
        self.assertEqual(after.effective_cutoff, "2027-01-04T17:30:00+00:00")


class AsOfSourceTest(unittest.TestCase):
    def test_future_and_late_revision_do_not_change_view(self):
        end = pd.Timestamp("2026-09-24T17:30:00Z")
        base = pd.DataFrame([{"start_at": end-pd.Timedelta(minutes=5), "end_at": end,
                              "available_at": end+pd.Timedelta(seconds=1),
                              "received_at": pd.NaT, "session_date": "2026-09-24",
                              "session_type": "regular", "close": 100.0}])
        altered = pd.concat([base, pd.DataFrame([
            {**base.iloc[0].to_dict(), "close": 999.0, "received_at": end+pd.Timedelta(hours=1)},
            {**base.iloc[0].to_dict(), "start_at": end, "end_at": end+pd.Timedelta(minutes=5),
             "available_at": end+pd.Timedelta(minutes=5, seconds=1), "close": 500.0}])], ignore_index=True)
        deadline = end+pd.Timedelta(seconds=30)
        self.assertEqual(visible_versions(base, end, deadline).close.tolist(),
                         visible_versions(altered, end, deadline).close.tolist())
        self.assertEqual(source_watermark(altered, "2026-09-24", end, deadline)["status"],
                         "complete_assumed_availability")

    def test_no_future_labels_required_for_hourly_features(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            days = pd.bdate_range("2026-09-10", periods=11).strftime("%Y-%m-%d").tolist()
            calendar = {"calendar_id": "fixture", "sessions": []}
            source = root/"bars"
            for day in days:
                op = datetime.fromisoformat(day).replace(hour=9, minute=30, tzinfo=ET).astimezone(UTC)
                cl = op+timedelta(minutes=390)
                calendar["sessions"].append({"session_date": day, "open_at": op.isoformat(),
                                             "close_at": cl.isoformat()})
            (root/"calendar.json").write_text(json.dumps(calendar))
            universe = {"universe_mode": "current_universe_retrospective",
                        "members": [{"symbol": "AAA", "role": "candidate", "industry_proxy": "SEC"}]}
            (root/"universe.json").write_text(json.dumps(universe))
            rows = []
            for day in days:
                op = pd.Timestamp(datetime.fromisoformat(day).replace(hour=9, minute=30, tzinfo=ET))
                for i in range(78):
                    start = op+pd.Timedelta(minutes=5*i)
                    end = start+pd.Timedelta(minutes=5)
                    rows.append({"start_at": start.tz_convert("UTC").isoformat(),
                                 "end_at": end.tz_convert("UTC").isoformat(),
                                 "available_at": (end+pd.Timedelta(seconds=1)).tz_convert("UTC").isoformat(),
                                 "session_type": "regular", "price_basis": "NONE",
                                 "open": 100+i/100, "high": 100.1+i/100,
                                 "low": 99.9+i/100, "close": 100.05+i/100, "volume": 1000,
                                 "session_date": day})
            for symbol in ("AAA", "QQQ", "SEC"):
                p = source/symbol/"2026.parquet"
                p.parent.mkdir(parents=True)
                pd.DataFrame(rows).to_parquet(p, index=False)
            config = {"source_dir": "bars", "universe": "universe.json", "calendar": "calendar.json",
                      "corporate_actions_dir": "actions", "source_start_date": days[0],
                      "source_end_date": days[-1], "history_sessions": 10,
                      "decision_hours_et": ["10:30", "15:30"], "signal_latency_seconds": 30,
                      "target_return": 0.05, "feature_availability": "fixture_assumption"}
            cutoff = pd.Timestamp(datetime.fromisoformat(days[-1]).replace(hour=10, minute=30, tzinfo=ET)).tz_convert("UTC")
            before = build_bundle(root, config, include_labels=False, only_date=days[-1],
                                  as_of=cutoff+pd.Timedelta(seconds=30))
            self.assertEqual(len(before.rows), 1)
            self.assertIsNone(before.rows.iloc[0].target)
            for symbol in ("AAA", "QQQ", "SEC"):
                p = source/symbol/"2026.parquet"
                f = pd.read_parquet(p)
                late = f[(f.session_date == days[-1]) & (f.end_at == cutoff.isoformat())].iloc[0].copy()
                late["close"] = 9999.0
                late["received_at"] = (cutoff+pd.Timedelta(hours=1)).isoformat()
                f = pd.concat([f, pd.DataFrame([late])], ignore_index=True)
                f.loc[f.session_date == days[-1], "high"] = f.loc[f.session_date == days[-1], "high"]*2
                # Only future bars are altered; restore the current cutoff prefix.
                f.loc[(f.session_date == days[-1]) & (f.end_at <= cutoff.isoformat()), "high"] /= 2
                f.to_parquet(p, index=False)
            after = build_bundle(root, config, include_labels=False, only_date=days[-1],
                                 as_of=cutoff+pd.Timedelta(seconds=30))
            cols = [c for c in before.rows if c.startswith(("h_", "a_", "b_", "ab_"))]
            pd.testing.assert_frame_equal(before.rows[cols], after.rows[cols])
            np.testing.assert_array_equal(before.b_seq, after.b_seq)


class ThresholdTest(unittest.TestCase):
    def test_no_forced_candidate(self):
        config = {"threshold_grid": [0.3, 0.4], "threshold_min_recommendations": 100,
                  "threshold_min_precision_lift_absolute": 0.05}
        result = _select_threshold(config, np.zeros(20), np.full(20, 0.9))
        self.assertIsNone(result["threshold"])

    def test_long_acquisition_expires_action_but_historical_stays_research(self):
        kw = {"complete": True, "score": .8, "resolution_state": "aligned",
              "generated_at": datetime(2026, 9, 24, 18, 40, tzinfo=UTC),
              "expires_at": datetime(2026, 9, 24, 18, 37, tzinfo=UTC),
              "received": True, "universe_known": True,
              "manually_excluded": False, "threshold": .3}
        self.assertEqual(classify_action(**kw, historical=False)[0], "expired")
        self.assertEqual(classify_action(**kw, historical=True)[0], "historical_research_only")
        before_expiry = dict(kw, generated_at=datetime(2026, 9, 24, 18, 31, tzinfo=UTC),
                             universe_known=False)
        self.assertEqual(classify_action(**before_expiry, historical=False)[0], "data_unverified")
        no_threshold = dict(before_expiry, universe_known=True, threshold=None)
        self.assertEqual(classify_action(**no_threshold, historical=False)[0], "no_action")


class ForwardMaturityTest(unittest.TestCase):
    def test_pending_missing_and_matured_have_distinct_denominators(self):
        with TemporaryDirectory(dir=REPO/"research/after_open_3d5pct/runs") as temp:
            source = Path(temp)
            days = ["2026-09-21", "2026-09-22", "2026-09-23", "2026-09-24"]
            sessions, bars = [], []
            for day in days:
                op = datetime.fromisoformat(day).replace(hour=9, minute=30, tzinfo=ET).astimezone(UTC)
                sessions.append({"session_date": day, "open_at": op.isoformat(),
                                 "close_at": (op+timedelta(minutes=390)).isoformat()})
                for i in range(78):
                    start = op+timedelta(minutes=5*i)
                    bars.append({"start_at": start.isoformat(),
                                 "available_at": (start+timedelta(minutes=5, seconds=1)).isoformat(),
                                 "session_type": "regular", "open": 100.0,
                                 "high": 106.0 if day == days[1] and i == 2 else 101.0,
                                 "low": 99.0})
            p = source/"AAA"/"2026.parquet"
            p.parent.mkdir()
            pd.DataFrame(bars).to_parquet(p, index=False)
            row = {"symbol": "AAA", "sample_id": "fixture", "status": "research_candidate",
                   "effective_cutoff": "2026-09-21T14:30:00+00:00", "predictions": {"raw_model_output": .6}}
            calendar = {"sessions": sessions}
            pending = evaluate_row(row, calendar, source, datetime(2026, 9, 22, tzinfo=UTC))
            self.assertEqual(pending["label_status"], "pending")
            mature = evaluate_row(row, calendar, source, datetime(2026, 9, 25, tzinfo=UTC))
            self.assertEqual(mature["label_status"], "matured_assumed_historical_availability")
            self.assertEqual(mature["target"], 1)
            f = pd.read_parquet(p).drop(index=200)
            f.to_parquet(p, index=False)
            missing = evaluate_row(row, calendar, source, datetime(2026, 9, 25, tzinfo=UTC))
            self.assertEqual(missing["label_status"], "missing_regular_bar_or_not_yet_available")


if __name__ == "__main__":
    unittest.main()
