import os
import tempfile
import unittest
from pathlib import Path

from screener import Bar, ScreenConfig, evaluate_symbol, load_env_file, write_outputs


def make_bar(day: int, close: float, intraday_half_range: float = 0.006) -> Bar:
    return Bar(
        session=f"2026-07-{day:02d}",
        open=close,
        high=close * (1 + intraday_half_range),
        low=close * (1 - intraday_half_range),
        close=close,
        volume=1_000_000,
    )


class EvaluateSymbolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ScreenConfig(min_avg_dollar_volume=1_000_000)

    def test_matches_seven_session_drop_then_seven_session_base(self) -> None:
        drop = [100.0, 96.0, 92.0, 88.0, 84.0, 82.0, 80.5, 79.5]
        flat = [79.8, 79.3, 80.1, 79.6, 80.0, 79.7, 80.2]
        bars = [make_bar(index + 1, close) for index, close in enumerate(drop + flat)]
        result = evaluate_symbol("TEST", "Synthetic", bars, self.config)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.drop_pct, -20.5)
        self.assertEqual(result.as_of, "2026-07-15")

    def test_rejects_fourteen_bars_because_it_has_only_six_plus_seven_intervals(self) -> None:
        closes = [100, 95, 90, 86, 83, 81, 79.5] + [79.7] * 7
        bars = [make_bar(index + 1, close) for index, close in enumerate(closes)]
        self.assertIsNone(evaluate_symbol("SHORT", "Too Short", bars, self.config))

    def test_rejects_continued_downtrend_masquerading_as_a_base(self) -> None:
        drop = [100.0, 96.0, 92.0, 88.0, 84.0, 82.0, 80.5, 79.5]
        flat = [79.0, 78.4, 77.8, 77.2, 76.6, 76.0, 75.4]
        bars = [make_bar(index + 1, close) for index, close in enumerate(drop + flat)]
        self.assertIsNone(evaluate_symbol("DOWN", "Still Falling", bars, self.config))

    def test_rejects_illiquid_stock(self) -> None:
        config = ScreenConfig(min_avg_dollar_volume=100_000_000)
        drop = [100.0, 96.0, 92.0, 88.0, 84.0, 82.0, 80.5, 79.5]
        flat = [79.8, 79.3, 80.1, 79.6, 80.0, 79.7, 80.2]
        bars = [make_bar(index + 1, close) for index, close in enumerate(drop + flat)]
        self.assertIsNone(evaluate_symbol("THIN", "Illiquid", bars, config))


class EnvFileTests(unittest.TestCase):
    def test_loads_env_without_overwriting_existing_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("NEW_TEST_KEY=from_file\nEXISTING_TEST_KEY=from_file\n", encoding="utf-8")
            os.environ.pop("NEW_TEST_KEY", None)
            os.environ["EXISTING_TEST_KEY"] = "from_environment"
            try:
                load_env_file(path)
                self.assertEqual(os.environ["NEW_TEST_KEY"], "from_file")
                self.assertEqual(os.environ["EXISTING_TEST_KEY"], "from_environment")
            finally:
                os.environ.pop("NEW_TEST_KEY", None)
                os.environ.pop("EXISTING_TEST_KEY", None)


class OutputTests(unittest.TestCase):
    def test_writes_daily_and_stable_latest_reports(self) -> None:
        config = ScreenConfig(min_avg_dollar_volume=1_000_000)
        drop = [100.0, 96.0, 92.0, 88.0, 84.0, 82.0, 80.5, 79.5]
        flat = [79.8, 79.3, 80.1, 79.6, 80.0, 79.7, 80.2]
        bars = [make_bar(index + 1, close) for index, close in enumerate(drop + flat)]
        candidate = evaluate_symbol("TEST", "Synthetic", bars, config)
        assert candidate is not None
        counts = {"universe": 5000, "complete_bar_history": 4900, "matched": 1}

        with tempfile.TemporaryDirectory() as directory:
            paths = write_outputs(Path(directory), "2026-07-15", [candidate], config, counts)
            for path in paths:
                self.assertTrue(path.exists())
            markdown = (Path(directory) / "2026-07-15" / "screen.md").read_text(encoding="utf-8")
            self.assertIn("# 跌后平台候选 — 2026-07-15", markdown)
            self.assertIn("**TEST**", markdown)
            self.assertIn("## 二次研究任务", markdown)
            self.assertEqual(markdown, (Path(directory) / "latest.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
