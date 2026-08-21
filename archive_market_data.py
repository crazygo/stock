#!/usr/bin/env python3
"""Persist cached Massive grouped daily U.S. stock responses into Git history.

The screener already caches raw `/v2/aggs/grouped/...` responses under
`.cache/massive/bars/YYYY-MM-DD.json`. GitHub Actions cache is only an execution
optimization, so this script creates a durable, compressed archive that can be
used for future backtests without re-downloading the same market days.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class ArchiveError(RuntimeError):
    """Raised when a market-day cache cannot be safely archived."""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArchiveError(f"Missing input file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArchiveError(f"Invalid JSON in {path}: {exc}") from exc


def infer_session(results_json: Path) -> str:
    payload = _load_json(results_json)
    session = payload.get("as_of") if isinstance(payload, Mapping) else None
    if not isinstance(session, str) or len(session) != 10:
        raise ArchiveError(f"Could not infer YYYY-MM-DD as_of from {results_json}")
    return session


def validate_grouped_payload(payload: Any, *, min_results: int) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ArchiveError("Grouped-day payload must be a JSON object")
    status = payload.get("status")
    if status not in {None, "OK", "DELAYED"}:
        raise ArchiveError(f"Unexpected Massive status: {status}")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ArchiveError("Grouped-day payload has no results array")
    if len(results) < min_results:
        raise ArchiveError(
            f"Grouped-day payload looks incomplete: {len(results)} rows < {min_results}"
        )
    ticker_count = sum(
        1 for item in results if isinstance(item, Mapping) and isinstance(item.get("T"), str)
    )
    if ticker_count < min_results:
        raise ArchiveError(
            f"Grouped-day payload has too few ticker rows: {ticker_count} < {min_results}"
        )
    return {
        "status": status,
        "results_count": len(results),
        "ticker_count": ticker_count,
        "adjusted": payload.get("adjusted"),
        "query_count": payload.get("queryCount"),
    }


def archive_grouped_day(
    *,
    session: str,
    cache_dir: Path,
    archive_dir: Path,
    min_results: int = 1000,
) -> dict[str, Any]:
    source_path = cache_dir / "bars" / f"{session}.json"
    try:
        source_bytes = source_path.read_bytes()
    except FileNotFoundError as exc:
        raise ArchiveError(f"No cached grouped-day response for {session}: {source_path}") from exc

    try:
        payload = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveError(f"Cached grouped-day response is invalid JSON: {source_path}") from exc

    stats = validate_grouped_payload(payload, min_results=min_results)
    compressed = gzip.compress(source_bytes, compresslevel=9, mtime=0)
    digest = hashlib.sha256(source_bytes).hexdigest()

    relative_path = Path(session) / "grouped.json.gz"
    target_path = archive_dir / relative_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists() or target_path.read_bytes() != compressed:
        target_path.write_bytes(compressed)

    manifest_path = archive_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        if not isinstance(manifest, dict):
            raise ArchiveError(f"Invalid archive manifest: {manifest_path}")
    else:
        manifest = {
            "version": 1,
            "market": "us_stocks_grouped_daily",
            "source": "Massive /v2/aggs/grouped/locale/us/market/stocks/{date}",
            "archives": {},
        }

    archives = manifest.setdefault("archives", {})
    if not isinstance(archives, dict):
        raise ArchiveError(f"Invalid archives map in {manifest_path}")
    entry = {
        "path": relative_path.as_posix(),
        "sha256_uncompressed": digest,
        "uncompressed_bytes": len(source_bytes),
        "compressed_bytes": len(compressed),
        **stats,
    }
    archives[session] = entry
    manifest["archives"] = {key: archives[key] for key in sorted(archives)}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return {"session": session, "archive": str(target_path), "manifest": str(manifest_path), **entry}


def archive_all_cached_days(
    *,
    cache_dir: Path,
    archive_dir: Path,
    min_results: int = 1000,
    required_session: str | None = None,
) -> dict[str, Any]:
    bars_dir = cache_dir / "bars"
    source_paths = sorted(bars_dir.glob("????-??-??.json")) if bars_dir.exists() else []
    if not source_paths:
        raise ArchiveError(f"No cached grouped-day files found in {bars_dir}")

    archived: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for source_path in source_paths:
        session = source_path.stem
        try:
            archived.append(
                archive_grouped_day(
                    session=session,
                    cache_dir=cache_dir,
                    archive_dir=archive_dir,
                    min_results=min_results,
                )
            )
        except ArchiveError as exc:
            skipped.append({"session": session, "reason": str(exc)})

    archived_sessions = {item["session"] for item in archived}
    if required_session and required_session not in archived_sessions:
        skipped_reason = next(
            (item["reason"] for item in skipped if item["session"] == required_session),
            "required session was not present in cache",
        )
        raise ArchiveError(f"Required session {required_session} was not archived: {skipped_reason}")
    if not archived:
        raise ArchiveError("No valid full-market sessions were archived")

    return {
        "archived_count": len(archived),
        "sessions": sorted(archived_sessions),
        "skipped": skipped,
        "manifest": str(archive_dir / "manifest.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", help="Required/latest market session YYYY-MM-DD")
    parser.add_argument("--results-json", type=Path, default=Path("results/latest.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/massive"))
    parser.add_argument("--archive-dir", type=Path, default=Path("market_data/us"))
    parser.add_argument(
        "--all-cached",
        action="store_true",
        help="Archive every valid grouped-day response already present in the Actions cache",
    )
    parser.add_argument(
        "--min-results",
        type=int,
        default=1000,
        help="Safety floor to avoid committing a partial market response",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        required_session = args.session or infer_session(args.results_json)
        if args.all_cached:
            result = archive_all_cached_days(
                cache_dir=args.cache_dir,
                archive_dir=args.archive_dir,
                min_results=args.min_results,
                required_session=required_session,
            )
        else:
            result = archive_grouped_day(
                session=required_session,
                cache_dir=args.cache_dir,
                archive_dir=args.archive_dir,
                min_results=args.min_results,
            )
    except (ArchiveError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
