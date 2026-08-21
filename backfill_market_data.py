#!/usr/bin/env python3
"""Backfill durable U.S. grouped daily market archives from Massive.

The existing daily screener already knows how to download one grouped U.S.
stock market session. This command extends that mechanism across an explicit
historical range and persists each valid trading day through
archive_market_data.py. Existing archive dates are skipped, so reruns are safe.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from archive_market_data import ArchiveError, archive_grouped_day
from screener import MassiveClient, ScreenerError, load_env_file


def _load_archived_sessions(archive_dir: Path) -> set[str]:
    manifest_path = archive_dir / "manifest.json"
    if not manifest_path.exists():
        return set()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    archives = payload.get("archives") if isinstance(payload, Mapping) else None
    return set(archives) if isinstance(archives, Mapping) else set()


def backfill_range(
    *,
    client: MassiveClient,
    start: date,
    end: date,
    cache_dir: Path,
    archive_dir: Path,
    min_results: int = 1000,
) -> dict[str, Any]:
    if end < start:
        raise ValueError("end must be on or after start")

    archived_sessions = _load_archived_sessions(archive_dir)
    counts = {
        "calendar_days": 0,
        "weekdays": 0,
        "already_archived": 0,
        "archived": 0,
        "no_market_data": 0,
    }
    archived: list[str] = []
    skipped_no_data: list[str] = []

    cursor = start
    while cursor <= end:
        counts["calendar_days"] += 1
        if cursor.weekday() >= 5:
            cursor += timedelta(days=1)
            continue
        counts["weekdays"] += 1
        session = cursor.isoformat()
        if session in archived_sessions:
            counts["already_archived"] += 1
            cursor += timedelta(days=1)
            continue

        bars = client.fetch_grouped_day(cursor)
        if not bars:
            counts["no_market_data"] += 1
            skipped_no_data.append(session)
            cursor += timedelta(days=1)
            continue

        archive_grouped_day(
            session=session,
            cache_dir=cache_dir,
            archive_dir=archive_dir,
            min_results=min_results,
        )
        archived_sessions.add(session)
        archived.append(session)
        counts["archived"] += 1
        cursor += timedelta(days=1)

    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "counts": counts,
        "archived_sessions": archived,
        "no_market_data_sessions": skipped_no_data,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Inclusive YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="Inclusive YYYY-MM-DD")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/massive"))
    parser.add_argument("--archive-dir", type=Path, default=Path("market_data/us"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--api-key-env", default="MASSIVE_API_KEY")
    parser.add_argument("--request-delay", type=float, default=12.2)
    parser.add_argument("--min-results", type=int, default=1000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        load_env_file(args.env_file)
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ScreenerError(f"Set {args.api_key_env} before running backfill")
        client = MassiveClient(
            api_key=api_key,
            cache_dir=args.cache_dir,
            request_delay=args.request_delay,
            max_retries=8,
        )
        result = backfill_range(
            client=client,
            start=start,
            end=end,
            cache_dir=args.cache_dir,
            archive_dir=args.archive_dir,
            min_results=args.min_results,
        )
    except (ArchiveError, ScreenerError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
