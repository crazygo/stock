"""Causal historical quality cohort and label-version boundary tests."""

from __future__ import annotations

import unittest

import pandas as pd

from research.after_open_3d5pct.quality_pool_v5 import HOURS, build_quality_features


CONFIG = {"lookback_official_prior_sessions": 63,
          "expected_distinct_cutoffs_et": list(HOURS),
          "min_complete_mature_stock_dates": 40,
          "strict_historical_rate_gt": .6,
          "recent_prior_windows_sessions": [21, 42],
          "recent_prior_min_complete_dates": [10, 20]}


class QualityPoolV5Tests(unittest.TestCase):
    def setUp(self):
        self.sessions = pd.bdate_range("2026-01-02", periods=66).strftime("%Y-%m-%d").tolist()
        self.decision = pd.Timestamp(self.sessions[64] + "T20:30:30Z")
        self.query = pd.DataFrame([{"sample_id": "Q", "symbol": "XYZ",
                                    "session_date": self.sessions[64],
                                    "decision_at": self.decision}])
        items = []
        # Exactly 40 of the preceding 63 official sessions have six complete
        # historical labels; 24 good dates/16 bad dates = exactly 60%.
        for j, day in enumerate(self.sessions[1:41]):
            for hour in HOURS:
                end = pd.Timestamp(day + "T20:00:00Z") + pd.Timedelta(days=3)
                items.append({"sample_id": f"XYZ|{day}|{hour}", "symbol": "XYZ",
                              "session_date": day, "cutoff_et": hour,
                              "target": 1 if j < 24 else 0, "status": "mature",
                              "label_end_at": end,
                              "label_available_at": end + pd.Timedelta(seconds=1),
                              "label_received_at": pd.NaT})
        self.labels = pd.DataFrame(items)
        self.labels["label_received_at"] = pd.to_datetime(self.labels.label_received_at, utc=True)

    def feature(self, labels=None):
        return build_quality_features(self.query, self.labels if labels is None else labels,
                                      self.sessions, CONFIG).iloc[0]

    def test_exactly_sixty_is_excluded_and_one_extra_hit_is_admitted(self):
        row = self.feature()
        self.assertEqual(row.h_quality_mature_days_63, 40)
        self.assertEqual(row.h_quality_rate_63, .6)
        self.assertFalse(row.quality_eligible)
        changed = self.labels.copy()
        changed.loc[changed.index[-1], "target"] = 1
        self.assertTrue(self.feature(changed).quality_eligible)

    def test_lookback_uses_official_sessions_not_extra_old_success(self):
        old = self.labels.iloc[:6].copy()
        old["session_date"] = self.sessions[0]
        old["target"] = 1
        old["sample_id"] = [f"old|{h}" for h in HOURS]
        row = self.feature(pd.concat([self.labels, old], ignore_index=True))
        self.assertEqual(row.h_quality_rate_63, .6)
        self.assertEqual(row.h_quality_mature_days_63, 40)

    def test_incomplete_duplicate_cutoff_and_pending_date_do_not_count(self):
        missing = self.labels.drop(self.labels.index[0]).copy()
        duplicate = missing.iloc[[0]].copy()
        duplicate["sample_id"] = "duplicate_other_sample"
        duplicate["label_received_at"] = self.decision - pd.Timedelta(seconds=1)
        replaced = pd.concat([missing, duplicate], ignore_index=True)
        row = self.feature(replaced)
        self.assertEqual(row.h_quality_mature_days_63, 39)
        self.assertEqual(row.quality_status, "insufficient_history")
        pending = self.labels.copy()
        pending.loc[0, ["status", "target", "label_available_at"]] = ["pending", None, pd.NaT]
        self.assertEqual(self.feature(pending).h_quality_mature_days_63, 39)

    def test_identical_transport_duplicate_is_idempotent_but_conflict_fails_closed(self):
        duplicate = self.labels.iloc[[0]].copy()
        duplicate["sample_id"] = "duplicate_same_version"
        pd.testing.assert_series_equal(self.feature(pd.concat([self.labels, duplicate], ignore_index=True)),
                                       self.feature())
        duplicate["target"] = 0
        with self.assertRaisesRegex(ValueError, "conflicting same-time"):
            self.feature(pd.concat([self.labels, duplicate], ignore_index=True))

    def test_late_revision_or_extra_hour_does_not_rewrite_past(self):
        base = self.feature()
        revision = self.labels.iloc[[0]].copy()
        revision["target"] = 0
        revision["label_received_at"] = self.decision + pd.Timedelta(seconds=1)
        late = pd.concat([self.labels, revision], ignore_index=True)
        row = self.feature(late)
        pd.testing.assert_series_equal(row, base)
        extra = self.labels.iloc[[0]].copy()
        extra["cutoff_et"] = "16:30"
        extra["label_received_at"] = self.decision + pd.Timedelta(seconds=1)
        row = self.feature(pd.concat([self.labels, extra], ignore_index=True))
        pd.testing.assert_series_equal(row, base)

    def test_late_sixth_hour_does_not_change_any_past_feature(self):
        missing = self.labels.iloc[1:].copy()
        before = self.feature(missing)
        sixth = self.labels.iloc[[0]].copy()
        sixth["label_received_at"] = self.decision + pd.Timedelta(seconds=1)
        after = self.feature(pd.concat([missing, sixth], ignore_index=True))
        pd.testing.assert_series_equal(after, before)

    def test_full_history_future_date_and_unavailable_label_do_not_enter(self):
        future = self.labels.iloc[:6].copy()
        future["session_date"] = self.sessions[65]
        future["sample_id"] = [f"future|{h}" for h in HOURS]
        future["target"] = 1
        self.assertEqual(self.feature(pd.concat([self.labels, future], ignore_index=True)).h_quality_rate_63, .6)
        late = self.labels.copy()
        late.loc[0, "label_available_at"] = self.decision + pd.Timedelta(seconds=1)
        self.assertEqual(self.feature(late).h_quality_mature_days_63, 39)


if __name__ == "__main__":
    unittest.main()
