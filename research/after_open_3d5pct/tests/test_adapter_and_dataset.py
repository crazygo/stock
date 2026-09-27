"""Unit tests for research v2 real data adapter and dataset builder."""

from datetime import datetime, timezone
import unittest
from zoneinfo import ZoneInfo

from research.after_open_3d5pct.adapter import load_calendar_sessions, load_universe, load_research_bars
from research.after_open_3d5pct.config import load_config
from research.after_open_3d5pct.dataset import build_dataset
from research.after_open_3d5pct.splits import purged_fold
from research.after_open_3d5pct.baseline import fit_base_rate, score_constant
from research.after_open_3d5pct.timeaxis import ET

UTC = ZoneInfo("UTC")


class AdapterAndDatasetTests(unittest.TestCase):

    def test_calendar_sessions_loading(self):
        sessions = load_calendar_sessions()
        self.assertGreater(len(sessions), 180)
        # Check sorted and non-overlapping
        for i in range(1, len(sessions)):
            self.assertLess(sessions[i - 1].close_at, sessions[i].open_at)
        # Check 2026-07-03 Independence day observed is closed or early close
        july3 = [s for s in sessions if s.open_at.astimezone(ET).strftime("%Y-%m-%d") == "2026-07-03"]
        self.assertEqual(len(july3), 0, "July 3 2026 is Independence Day observed holiday and must be closed")

    def test_universe_loading(self):
        candidates = load_universe(role="candidate")
        benchmarks = load_universe(role="benchmark")
        self.assertEqual(len(candidates), 101)
        self.assertIn("AAPL", candidates)
        self.assertIn("NVDA", candidates)
        self.assertIn("QQQ", benchmarks)
        self.assertIn("SPY", benchmarks)
        self.assertIn("SOXX", benchmarks)
        self.assertEqual(len(set(candidates).intersection(set(benchmarks))), 0)

    def test_real_dataset_generation_and_splits(self):
        sessions = load_calendar_sessions()
        config = load_config()
        # Test on AAPL
        bars_dict = load_research_bars(["AAPL"], interval="5m", price_basis="NONE", session="ALL", year=2026)
        self.assertIn("AAPL", bars_dict)
        self.assertGreater(len(bars_dict["AAPL"]), 1000)

        ds = build_dataset(
            symbols=["AAPL"],
            bars_by_symbol=bars_dict,
            sessions=sessions,
            config=config,
            as_of=datetime(2026, 10, 1, tzinfo=UTC)
        )

        samples = ds["samples"]
        self.assertGreater(len(samples), 50)
        for s in samples:
            self.assertTrue(s.sample_id.startswith("AAPL|"))
            if s.label_status == "mature":
                self.assertIn(s.target, (0, 1))

        # Test fold split
        val_start = datetime(2026, 9, 1, 0, 0, tzinfo=ET)
        val_end = datetime(2026, 9, 15, 0, 0, tzinfo=ET)
        as_of = datetime(2026, 10, 1, 0, 0, tzinfo=ET)

        fold = purged_fold(samples, val_start, val_end, as_of)
        self.assertGreater(len(fold.train), 0)
        self.assertGreater(len(fold.validation), 0)

        # Baseline evaluation
        train_prob = fit_base_rate(fold.train)
        metrics = score_constant(train_prob, fold.validation)
        self.assertIn("brier", metrics)
        self.assertIn("log_loss", metrics)
        self.assertGreaterEqual(metrics["brier"], 0.0)


if __name__ == "__main__":
    unittest.main()
