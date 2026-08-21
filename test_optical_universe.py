import json
import tempfile
import unittest
from pathlib import Path

from optical_universe import load_universe, select_companies, write_outputs


class OpticalUniverseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = load_universe(Path(__file__).with_name("optical_universe.json"))

    def test_default_strict_contains_expected_names(self):
        rows = select_companies(self.payload["companies"], market="both", include_broad=False, include_watch=False)
        symbols = {(row["market"], row["symbol"]) for row in rows}
        for expected in {
            ("US", "AAOI"), ("US", "LITE"), ("US", "FN"), ("US", "CIEN"), ("US", "CLFD"),
            ("HK", "03308"), ("HK", "01191"), ("HK", "01879"), ("HK", "06869"),
            ("HK", "01617"), ("HK", "09963"),
        }:
            self.assertIn(expected, symbols)
        self.assertNotIn(("HK", "06166"), symbols)
        self.assertNotIn(("US", "COHR"), symbols)

    def test_quantified_strict_shares_clear_threshold(self):
        rows = select_companies(self.payload["companies"], market="both", include_broad=False, include_watch=False)
        for row in rows:
            share = row.get("optical_revenue_pct")
            if share is not None:
                self.assertGreater(share, 50.0, row["symbol"])

    def test_market_filter(self):
        rows = select_companies(self.payload["companies"], market="HK", include_broad=True, include_watch=True)
        self.assertTrue(rows)
        self.assertTrue(all(row["market"] == "HK" for row in rows))

    def test_outputs_are_machine_and_human_readable(self):
        rows = select_companies(self.payload["companies"], market="US", include_broad=False, include_watch=False)
        for row in rows:
            row.update({"price": None, "currency": None, "exchange": None, "quote_time_utc": None, "quote_status": "skipped", "quote_error": None})
        with tempfile.TemporaryDirectory() as tmp:
            json_path, csv_path, md_path = write_outputs(Path(tmp), self.payload, rows)
            self.assertTrue(json.loads(json_path.read_text(encoding="utf-8"))["companies"])
            self.assertIn("Optical-primary stock universe", md_path.read_text(encoding="utf-8"))
            self.assertIn("symbol", csv_path.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
