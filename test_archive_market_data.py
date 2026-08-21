import gzip
import json
import tempfile
import unittest
from pathlib import Path

from archive_market_data import (
    ArchiveError,
    archive_all_cached_days,
    archive_grouped_day,
    infer_session,
)


class ArchiveMarketDataTests(unittest.TestCase):
    def _write_payload(self, path: Path, tickers: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "OK",
            "adjusted": True,
            "queryCount": len(tickers),
            "resultsCount": len(tickers),
            "results": [
                {"T": ticker, "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100}
                for ticker in tickers
            ],
        }
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    def test_archives_exact_cached_payload_and_updates_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            archive_dir = root / "market_data" / "us"
            source = cache_dir / "bars" / "2026-08-20.json"
            self._write_payload(source, ["AAA", "BBB"])

            first = archive_grouped_day(
                session="2026-08-20",
                cache_dir=cache_dir,
                archive_dir=archive_dir,
                min_results=2,
            )
            archive_path = archive_dir / "2026-08-20" / "grouped.json.gz"
            compressed_once = archive_path.read_bytes()
            self.assertEqual(gzip.decompress(compressed_once), source.read_bytes())
            self.assertEqual(first["results_count"], 2)
            self.assertEqual(first["ticker_count"], 2)

            second = archive_grouped_day(
                session="2026-08-20",
                cache_dir=cache_dir,
                archive_dir=archive_dir,
                min_results=2,
            )
            self.assertEqual(compressed_once, archive_path.read_bytes())
            self.assertEqual(first["sha256_uncompressed"], second["sha256_uncompressed"])

            manifest = json.loads((archive_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(list(manifest["archives"]), ["2026-08-20"])
            self.assertEqual(
                manifest["archives"]["2026-08-20"]["path"],
                "2026-08-20/grouped.json.gz",
            )

    def test_rejects_partial_market_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "cache" / "bars" / "2026-08-20.json"
            self._write_payload(source, ["AAA"])
            with self.assertRaises(ArchiveError):
                archive_grouped_day(
                    session="2026-08-20",
                    cache_dir=root / "cache",
                    archive_dir=root / "market_data" / "us",
                    min_results=2,
                )

    def test_backfills_valid_cache_and_skips_partial_old_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            archive_dir = root / "market_data" / "us"
            self._write_payload(cache_dir / "bars" / "2026-08-18.json", ["AAA", "BBB"])
            self._write_payload(cache_dir / "bars" / "2026-08-19.json", ["AAA"])
            self._write_payload(cache_dir / "bars" / "2026-08-20.json", ["AAA", "BBB"])

            result = archive_all_cached_days(
                cache_dir=cache_dir,
                archive_dir=archive_dir,
                min_results=2,
                required_session="2026-08-20",
            )
            self.assertEqual(result["sessions"], ["2026-08-18", "2026-08-20"])
            self.assertEqual(result["archived_count"], 2)
            self.assertEqual(result["skipped"][0]["session"], "2026-08-19")
            manifest = json.loads((archive_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(list(manifest["archives"]), ["2026-08-18", "2026-08-20"])

    def test_required_session_must_be_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_payload(root / "cache" / "bars" / "2026-08-20.json", ["AAA"])
            with self.assertRaises(ArchiveError):
                archive_all_cached_days(
                    cache_dir=root / "cache",
                    archive_dir=root / "market_data" / "us",
                    min_results=2,
                    required_session="2026-08-20",
                )

    def test_infers_session_from_latest_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "latest.json"
            path.write_text(json.dumps({"as_of": "2026-08-20"}), encoding="utf-8")
            self.assertEqual(infer_session(path), "2026-08-20")


if __name__ == "__main__":
    unittest.main()
