import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from analyze_apr_may_peak import group_index, learn_drawdown_groups
from backfill_market_data import backfill_range


class FakeClient:
    def __init__(self, available):
        self.available = set(available)
        self.calls = []

    def fetch_grouped_day(self, session):
        self.calls.append(session.isoformat())
        return {"AAA": object()} if session.isoformat() in self.available else {}


class HistoricalWaveTests(unittest.TestCase):
    def test_group_learning_finds_ordered_breaks(self):
        values = [
            0.03, 0.04, 0.05, 0.06, 0.07,
            0.16, 0.18, 0.19, 0.21, 0.22,
            0.35, 0.37, 0.39, 0.41, 0.44,
            0.61, 0.64, 0.69, 0.72, 0.80,
        ]
        result = learn_drawdown_groups(values)
        self.assertGreaterEqual(result["k"], 3)
        self.assertEqual(result["breaks"], sorted(result["breaks"]))
        groups = [group_index(x, result["breaks"]) for x in values]
        self.assertLess(groups[0], groups[-1])

    def test_backfill_skips_existing_and_weekends(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_dir = root / "market_data" / "us"
            archive_dir.mkdir(parents=True)
            (archive_dir / "manifest.json").write_text(
                json.dumps({
                    "archives": {
                        "2026-07-24": {"path": "2026-07-24/grouped.json.gz"}
                    }
                }),
                encoding="utf-8",
            )
            client = FakeClient({"2026-07-27"})

            # Patch the archive writer at module level so this remains a unit test.
            import backfill_market_data
            original = backfill_market_data.archive_grouped_day
            written = []
            try:
                backfill_market_data.archive_grouped_day = lambda **kwargs: written.append(kwargs["session"])
                result = backfill_range(
                    client=client,
                    start=date(2026, 7, 24),
                    end=date(2026, 7, 27),
                    cache_dir=root / "cache",
                    archive_dir=archive_dir,
                    min_results=1,
                )
            finally:
                backfill_market_data.archive_grouped_day = original

            self.assertEqual(client.calls, ["2026-07-27"])
            self.assertEqual(written, ["2026-07-27"])
            self.assertEqual(result["counts"]["already_archived"], 1)
            self.assertEqual(result["counts"]["archived"], 1)


if __name__ == "__main__":
    unittest.main()
