#!/usr/bin/env python3
"""Build the SEC-style ticker->CIK cache from Massive's bulk ticker directory.

GitHub-hosted runners may be blocked from www.sec.gov/files/company_tickers.json.
The same CIK identifier is present in Massive's paginated All Tickers endpoint,
which takes only ~6 requests for active U.S. common stocks. We write the small
mapping in the shape expected by enrich_industry_profit.SecClient.ticker_map so
all subsequent SIC/companyfacts requests remain sourced from data.sec.gov.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from screener import MassiveClient, ScreenerError, load_env_file


def build_map(client: MassiveClient) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    next_url: str | None = "/v3/reference/tickers"
    params: Mapping[str, Any] | None = {
        "market": "stocks",
        "locale": "us",
        "active": "true",
        "type": "CS",
        "limit": 1000,
        "sort": "ticker",
        "order": "asc",
    }
    index = 0
    while next_url:
        payload = client._request_json(next_url, params)
        params = None
        for item in payload.get("results") or []:
            if not isinstance(item, Mapping):
                continue
            ticker = item.get("ticker")
            cik = item.get("cik")
            if not ticker or cik in {None, ""}:
                continue
            try:
                cik_int = int(str(cik))
            except ValueError:
                continue
            result[str(index)] = {
                "cik_str": cik_int,
                "ticker": str(ticker),
                "title": item.get("name") or ticker,
            }
            index += 1
        next_url = payload.get("next_url")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".cache/sec/company_tickers.json"))
    parser.add_argument("--massive-cache", type=Path, default=Path(".cache/massive"))
    parser.add_argument("--request-delay", type=float, default=12.2)
    args = parser.parse_args(argv)

    load_env_file(Path(".env"))
    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        print("error: MASSIVE_API_KEY is required")
        return 2
    try:
        client = MassiveClient(api_key, args.massive_cache, request_delay=args.request_delay)
        mapping = build_map(client)
        if len(mapping) < 3000:
            raise ScreenerError(f"Ticker/CIK mapping looks incomplete: {len(mapping)}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(mapping, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    except (OSError, ScreenerError) as exc:
        print(f"error: {exc}")
        return 2
    print(json.dumps({"mapped_active_common_stocks": len(mapping), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
