from dataclasses import replace
from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest

from research.after_open_3d5pct.baseline import fit_base_rate
from research.after_open_3d5pct.config import load_config
from research.after_open_3d5pct.contracts import Bar, Sample, Session
from research.after_open_3d5pct.features import opening_features
from research.after_open_3d5pct.labeling import make_label, select_entry
from research.after_open_3d5pct.smoke import run_smoke
from research.after_open_3d5pct.splits import purged_fold
from research.after_open_3d5pct.timeaxis import ET, add_regular_minutes, intervals


def at(day, clock):
    return datetime.fromisoformat(f"{day}T{clock}:00").replace(tzinfo=ET)


def calendar(days, early_close=None):
    return [Session(at(d, "09:30"), at(d, (early_close or {}).get(d, "16:00"))) for d in days]


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config()
        cls.sessions = calendar(["2026-09-24", "2026-09-25", "2026-09-28", "2026-09-29", "2026-09-30"])
        cls.bars = []
        for session in cls.sessions:
            for start, end in intervals(session.open_at, session.close_at, cls.sessions, 5):
                cls.bars.append(Bar("TEST", start, end, end + timedelta(seconds=1),
                                    100, 101, 99, 100, 1000, "fixture"))
        cls.cutoff = at("2026-09-24", "11:30")
        cls.decision = cls.cutoff + timedelta(seconds=30)
        cls.entry = select_entry("TEST", cls.decision, cls.bars, cls.sessions, cls.config)
        cls.end = at("2026-09-29", "11:35")
        cls.as_of = cls.end + timedelta(seconds=1)

    def label(self, bars=None, as_of=None):
        return make_label(self.entry, self.bars if bars is None else bars,
                          self.sessions, self.as_of if as_of is None else as_of, self.config)

    def features(self, bars=None):
        return opening_features("TEST", self.cutoff, self.decision,
                                self.bars if bars is None else bars, self.sessions, self.config)

    def test_entry_respects_manual_delay_and_is_not_signal_close(self):
        self.assertEqual(self.entry.start_at, at("2026-09-24", "11:35"))
        self.assertGreater(self.entry.start_at, self.decision + timedelta(seconds=120))

    def test_missing_entry_does_not_skip_to_next_bar(self):
        with self.assertRaisesRegex(ValueError, "entry bar"):
            select_entry("TEST", self.decision, [b for b in self.bars if b != self.entry],
                         self.sessions, self.config)

    def test_entry_never_rolls_overnight(self):
        with self.assertRaisesRegex(ValueError, "same-session"):
            select_entry("TEST", at("2026-09-24", "15:59"), self.bars, self.sessions, self.config)

    def test_horizon_crosses_weekend_in_market_minutes(self):
        self.assertEqual(add_regular_minutes(self.entry.start_at, 1170, self.sessions), self.end)

    def test_holiday_and_early_close_consume_actual_minutes(self):
        sessions = calendar(["2026-11-25", "2026-11-27", "2026-11-30", "2026-12-01"],
                            {"2026-11-27": "13:00"})
        self.assertEqual(add_regular_minutes(at("2026-11-25", "11:35"), 1170, sessions),
                         at("2026-12-01", "14:35"))

    def test_dst_preserves_et_clock_not_fixed_utc_offset(self):
        sessions = calendar(["2026-10-30", "2026-11-02", "2026-11-03", "2026-11-04"])
        start = at("2026-10-30", "11:35")
        end = add_regular_minutes(start, 1170, sessions)
        self.assertEqual(end, at("2026-11-04", "11:35"))
        self.assertNotEqual(start.utcoffset(), end.utcoffset())

    def test_unknown_calendar_tail_rejected(self):
        with self.assertRaisesRegex(ValueError, "cover"):
            add_regular_minutes(self.entry.start_at, 1170, self.sessions[:2])

    def test_naive_timestamp_and_invalid_ohlc_rejected(self):
        with self.assertRaises(ValueError):
            replace(self.entry, start_at=self.entry.start_at.replace(tzinfo=None))
        with self.assertRaises(ValueError):
            replace(self.entry, low=102)

    def test_feature_schedule_excludes_opening_hour(self):
        cutoff = at("2026-09-24", "10:30")
        with self.assertRaisesRegex(ValueError, "schedule"):
            opening_features("TEST", cutoff, cutoff + timedelta(seconds=30),
                             self.bars, self.sessions, self.config)

    def test_feature_prefix_is_invariant_to_appended_mutated_future(self):
        prefix = [b for b in self.bars if b.end_at <= self.cutoff]
        future = [replace(b, high=10000, volume=99999999) for b in self.bars if b.end_at > self.cutoff]
        self.assertEqual(self.features(prefix), self.features(prefix + future))
        self.assertEqual(self.features()["volume_since_open"], 24_000)

    def test_delayed_feature_bar_is_not_backfilled(self):
        bars = [replace(b, available_at=self.decision + timedelta(minutes=1))
                if b.end_at == self.cutoff else b for b in self.bars]
        with self.assertRaisesRegex(ValueError, "late"):
            self.features(bars)

    def test_feature_missing_or_duplicate_prefix_rejected(self):
        for bars in (self.bars[1:], self.bars + [self.bars[0]]):
            with self.subTest(count=len(bars)), self.assertRaises(ValueError):
                self.features(bars)

    def test_full_negative_is_mature_with_mfe_and_mae(self):
        label = self.label()
        self.assertEqual((label.status, label.hit), ("mature", 0))
        self.assertAlmostEqual(label.mfe, 0.01)
        self.assertAlmostEqual(label.mae, -0.01)
        self.assertIsNone(label.hit_minutes_upper)

    def test_first_hit_time_is_interval_not_exact_timestamp(self):
        bars = [replace(b, high=105, low=94) if b.start_at == self.entry.end_at else b for b in self.bars]
        label = self.label(bars)
        self.assertEqual((label.hit, label.hit_minutes_lower, label.hit_minutes_upper), (1, 5, 10))
        self.assertAlmostEqual(label.mfe, 0.05)
        self.assertAlmostEqual(label.mae, -0.06)

    def test_pending_early_hit_never_becomes_training_positive(self):
        bars = [replace(b, high=108) if b.start_at == self.entry.end_at else b for b in self.bars]
        label = self.label(bars, self.entry.end_at + timedelta(minutes=10))
        self.assertEqual((label.status, label.hit, label.mfe), ("pending", None, None))

    def test_horizon_finished_but_last_bar_not_available_is_pending(self):
        label = self.label(as_of=self.end)
        self.assertEqual((label.status, label.reason), ("pending", "outcome_data_not_available"))

    def test_missing_label_bar_is_unknown_even_after_an_early_hit(self):
        bars = [replace(b, high=108) if b.start_at == self.entry.end_at else b
                for b in self.bars if b.start_at != self.entry.end_at + timedelta(minutes=5)]
        label = self.label(bars)
        self.assertEqual((label.status, label.hit), ("insufficient_data", None))

    def test_duplicate_label_bar_is_not_silently_deduplicated(self):
        self.assertEqual(self.label(self.bars + [self.entry]).status, "insufficient_data")

    def test_post_horizon_high_and_premarket_high_are_excluded(self):
        pre = Bar("TEST", at("2026-09-25", "08:00"), at("2026-09-25", "08:05"),
                  at("2026-09-25", "08:06"), 100, 200, 99, 100, 1000, "fixture")
        bars = [replace(b, high=200) if b.start_at >= self.end else b for b in self.bars] + [pre]
        self.assertEqual(self.label(bars).hit, 0)

    def test_mixed_price_basis_rejected(self):
        bars = [replace(b, price_basis="different") if b.start_at == self.entry.end_at else b for b in self.bars]
        with self.assertRaisesRegex(ValueError, "basis"):
            self.label(bars)

    def test_partial_boundary_cannot_use_full_bar(self):
        with self.assertRaisesRegex(ValueError, "finer"):
            intervals(self.entry.start_at, self.end + timedelta(minutes=1), self.sessions, 5)

    def test_semantic_change_requires_version(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            replace(self.config, target_return=0.08)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.start = at("2026-09-21", "00:00")
        self.end = at("2026-09-24", "00:00")
        self.as_of = at("2026-10-01", "00:00")

    def sample(self, key, decision, end, available=None, target=1):
        return Sample(key, key.split("|")[0], decision, end, available or end, "mature", target)

    def test_purge_uses_label_end_and_actual_availability(self):
        earlier = at("2026-09-17", "11:30")
        ended = at("2026-09-18", "16:00")
        samples = [self.sample("OK", earlier, ended),
                   self.sample("OVERLAP", earlier, at("2026-09-21", "11:30")),
                   self.sample("LATE", earlier, ended, self.start)]
        fold = purged_fold(samples, self.start, self.end, self.as_of)
        self.assertEqual([s.sample_id for s in fold.train], ["OK"])
        self.assertEqual(set(fold.excluded), {"OVERLAP", "LATE"})

    def test_same_date_all_symbols_and_hours_stay_in_validation(self):
        samples = [self.sample(f"{symbol}|{hour}", at("2026-09-21", hour), at("2026-09-25", "16:00"))
                   for symbol in ("A", "B") for hour in ("11:30", "14:30")]
        fold = purged_fold(samples, self.start, self.end, self.as_of)
        self.assertEqual((len(fold.train), len(fold.validation)), (0, 4))

    def test_intraday_split_rejected(self):
        with self.assertRaisesRegex(ValueError, "midnights"):
            purged_fold([], self.start + timedelta(hours=12), self.end, self.as_of)

    def test_pending_and_validation_outcome_never_affect_train_rate(self):
        train = self.sample("TRAIN", at("2026-09-14", "11:30"), at("2026-09-17", "11:30"), target=0)
        val = self.sample("VAL", at("2026-09-21", "11:30"), at("2026-09-24", "11:30"))
        pending = Sample("PENDING", "A", at("2026-09-18", "11:30"), at("2026-09-23", "11:30"),
                         None, "pending", None)
        before = purged_fold([train, val, pending], self.start, self.end, self.as_of)
        after = purged_fold([train, replace(val, target=0), pending], self.start, self.end, self.as_of)
        self.assertEqual(fit_base_rate(before.train), fit_base_rate(after.train))
        self.assertIn("PENDING", before.excluded)

    def test_pending_cannot_carry_positive_or_negative_label(self):
        with self.assertRaisesRegex(ValueError, "unmatured"):
            Sample("BAD", "A", at("2026-09-18", "11:30"), at("2026-09-23", "11:30"),
                   None, "pending", 0)


class SmokeTests(unittest.TestCase):
    def test_end_to_end_artifacts_are_auditable_and_synthetic(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            manifest = run_smoke(output, load_config())
            self.assertEqual(manifest["data_kind"], "synthetic")
            self.assertEqual(manifest["samples"], 240)
            self.assertGreater(manifest["label_status_counts"]["pending"], 0)
            folds = json.loads((output / "folds.json").read_text())
            self.assertEqual(len(folds), 2)
            for fold in folds:
                self.assertTrue(fold["train_ids"])
                self.assertTrue(fold["validation_ids"])
                self.assertTrue(set(fold["train_ids"]).isdisjoint(fold["validation_ids"]))
                self.assertTrue(any("purged" in reason for reason in fold["excluded"].values()))
            self.assertTrue((output / "predictions.jsonl").read_text())
            with self.assertRaisesRegex(ValueError, "already exists"):
                run_smoke(output, load_config())


if __name__ == "__main__":
    unittest.main()
