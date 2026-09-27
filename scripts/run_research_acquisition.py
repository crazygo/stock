#!/usr/bin/env python3
"""Batch execution pipeline for research v2 market data acquisition.

Provides automated batching and checkpointing for acquiring:
- 5m / 60m unadjusted (NONE) full-session (ALL) bars
- Corporate action rehab records (get_rehab)

Usage:
    # Run full QQQ candidate universe in steady batches:
    python3 scripts/run_research_acquisition.py --all

    # Run specific tickers:
    python3 scripts/run_research_acquisition.py --symbols MSFT GOOGL AMZN

    # Resume from checkpoint or run dry-run check:
    python3 scripts/run_research_acquisition.py --all --dry-run
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.fetch_research_data import ResearchDataFetcher


def load_universe_symbols() -> list[str]:
    flat_u = ROOT / "market_data" / "universe" / "qqq_retrospective_v1.json"
    nested_u = ROOT / "market_data" / "research_v2" / "universe" / "qqq_retrospective_v1.json"
    u_file = flat_u if flat_u.exists() else nested_u
    if not u_file.exists():
        raise FileNotFoundError(f"Universe file not found: {u_file}. Run scripts/build_universe_metadata.py first.")
    with open(u_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [m["symbol"] for m in data.get("members", [])]


def run_batch_acquisition(
    symbols: list[str],
    start_date: str = "2026-01-02",
    end_date: str = "2026-09-24",
    interval: str = "5m",
    price_basis: str = "NONE",
    session: str = "ALL",
    dry_run: bool = False
):
    fetcher = ResearchDataFetcher()
    year = int(start_date.split("-")[0])

    print(f"\n==================================================")
    print(f" Research v2 Market Data Acquisition Pipeline")
    print(f"==================================================")
    print(f" Target Symbols    : {len(symbols)} tickers")
    print(f" Interval / Basis  : {interval} / {price_basis} ({session})")
    print(f" Date Range        : {start_date} to {end_date}")
    print(f" Dry Run Mode      : {dry_run}")
    print(f" Storage Directory : {fetcher.root}")
    print(f"==================================================\n")

    completed = 0
    skipped = 0
    failed = 0

    try:
        for idx, sym in enumerate(symbols, 1):
            local_path = fetcher.get_local_path(sym, interval, price_basis, session, year)
            if local_path.exists():
                try:
                    df_existing = pd.read_parquet(local_path)
                    if (df_existing["session_date"].min() <= start_date and
                        df_existing["session_date"].max() >= end_date):
                        print(f"[{idx}/{len(symbols)}] [Skip/Cached] {sym} fully covers {start_date}..{end_date} ({len(df_existing)} bars)")
                        skipped += 1
                        continue
                    else:
                        print(f"[{idx}/{len(symbols)}] [Partial Cache] {sym} covers {df_existing['session_date'].min()}..{df_existing['session_date'].max()}, extending to {start_date}..{end_date}")
                except Exception:
                    pass

            if dry_run:
                print(f"[{idx}/{len(symbols)}] [Dry-Run] Would fetch {sym} ({start_date} -> {end_date})")
                completed += 1
                continue

            print(f"[{idx}/{len(symbols)}] [Fetching] {sym} ...")
            # 1. Fetch corporate actions
            fetcher.fetch_corporate_actions(sym)

            # 2. Fetch 5m K-lines with full pagination
            res = fetcher.fetch_kline(
                symbol=sym,
                start_date=start_date,
                end_date=end_date,
                interval="5m",
                price_basis=price_basis,
                session=session
            )

            # 3. Fetch 60m K-lines if not present
            local_60m = fetcher.get_local_path(sym, "60m", price_basis, session, year)
            if not local_60m.exists():
                fetcher.fetch_kline(
                    symbol=sym,
                    start_date=start_date,
                    end_date=end_date,
                    interval="60m",
                    price_basis=price_basis,
                    session=session
                )

            if res is not None:
                completed += 1
            else:
                failed += 1

    finally:
        fetcher.close()

    print(f"\nPipeline Finished: Completed={completed}, Skipped(Cached)={skipped}, Failed={failed}\n")


def main():
    parser = argparse.ArgumentParser(description="Batch acquisition runner for research v2")
    parser.add_argument("--all", action="store_true", help="Fetch entire QQQ universe")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to fetch")
    parser.add_argument("--start", default="2026-01-02", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-09-24", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="5m", choices=["5m", "60m", "day"], help="K-line interval")
    parser.add_argument("--price-basis", default="NONE", choices=["NONE", "QFQ"], help="Price basis")
    parser.add_argument("--session", default="ALL", choices=["ALL", "RTH"], help="Session coverage")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without requesting")
    args = parser.parse_args()

    if args.all:
        symbols = load_universe_symbols()
    elif args.symbols:
        symbols = [s.upper().replace("US.", "") for s in args.symbols]
    else:
        parser.error("Must specify either --all or --symbols")

    run_batch_acquisition(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
        price_basis=args.price_basis,
        session=args.session,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
