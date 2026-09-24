#!/usr/bin/env python3
"""Fetch full 2026 hourly (60m) extended-session data for QQQ constituents into Parquet.

Design Principles:
1. Safe Rate Limits: 1.0s - 1.5s delay between symbols (well below Futu's 60 req / 30s limit).
2. Resumable & Idempotent: Checks existing market_data/us_60m/<TICKER>/2026.parquet before requesting.
3. Full Sessions: extended_time=True, session=Session.ALL (overnight 20-04, pre 04-09:30, regular, post 16-20).
4. Direct Parquet Storage: Saves atomic ZSTD-compressed Parquet files directly.
5. NO Auto-Push: Data is saved strictly locally and NEVER automatically pushed to R2.

Usage:
    python3 scripts/fetch_qqq_hourly_parquet.py [--delay 1.2] [--dry-run] [--symbols AAPL NVDA]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_QQQ_FILE = ROOT / "analysis" / "qqq_constituents.json"
MARKET_DATA_DIR = ROOT / "market_data" / "us_60m"

FLOAT_COLS = [
    "open", "high", "low", "close", "volume", "turnover",
    "pe_ratio", "turnover_rate", "change_rate", "last_close"
]


def load_qqq_universe(qqq_path: Path = DEFAULT_QQQ_FILE) -> List[str]:
    """Load QQQ constituent ticker list."""
    if not qqq_path.exists():
        raise FileNotFoundError(f"QQQ constituents file not found: {qqq_path}")
    data = json.loads(qqq_path.read_text(encoding="utf-8"))
    tickers = [t.upper().strip() for t in data.get("tickers", []) if t.isalpha()]
    return sorted(set(tickers))


def check_local_status(ticker: str, start_date: str, end_date: str) -> Tuple[bool, Optional[pd.DataFrame]]:
    """Check if local 2026.parquet already completely covers the requested window."""
    parquet_p = MARKET_DATA_DIR / ticker / "2026.parquet"
    if not parquet_p.exists():
        return False, None
    try:
        df = pd.read_parquet(parquet_p)
        if df.empty or "time_key" not in df.columns:
            return False, df
        first_ts = str(df["time_key"].min())[:10]
        last_ts = str(df["time_key"].max())[:10]
        if first_ts <= start_date and last_ts >= end_date:
            return True, df
        return False, df
    except Exception:
        return False, None


def save_parquet(ticker: str, df: pd.DataFrame) -> Path:
    """Save DataFrame to market_data/us_60m/<TICKER>/2026.parquet using atomic replacement."""
    target_dir = MARKET_DATA_DIR / ticker
    target_dir.mkdir(parents=True, exist_ok=True)
    parquet_p = target_dir / "2026.parquet"

    # Type sanitation
    df["time_key"] = df["time_key"].astype(str)
    for c in FLOAT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    # Sort & deduplicate
    df = df.sort_values("time_key").drop_duplicates(subset=["time_key"], keep="last")

    temp_p = parquet_p.with_suffix(".parquet.tmp")
    df.to_parquet(temp_p, engine="pyarrow", compression="zstd", compression_level=7)
    temp_p.replace(parquet_p)
    return parquet_p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default="2026-01-01", help="Start date YYYY-MM-DD (default: 2026-01-01)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today NY)")
    parser.add_argument("--symbols", nargs="*", help="Optional custom ticker subset (default: all QQQ)")
    parser.add_argument("--delay", type=float, default=1.2, help="Polite delay between symbol requests in seconds (default: 1.2s)")
    parser.add_argument("--page-delay", type=float, default=0.25, help="Delay between pagination requests in seconds (default: 0.25s)")
    parser.add_argument("--dry-run", action="store_true", help="Print symbols and local cache status without making API requests")
    parser.add_argument("--host", default="127.0.0.1", help="Futu OpenD host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=11111, help="Futu OpenD port (default: 11111)")
    args = parser.parse_args()

    today_ny = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    end_date = args.end or today_ny
    start_date = args.start

    all_symbols = load_qqq_universe()
    if args.symbols:
        custom_set = {s.upper().strip() for s in args.symbols}
        targets = [s for s in all_symbols if s in custom_set]
    else:
        targets = all_symbols

    print(f"=== QQQ Hourly Parquet Fetcher ===")
    print(f"  Target Window  : {start_date} -> {end_date}")
    print(f"  Constituents   : {len(targets)} symbols")
    print(f"  Request Delay  : {args.delay}s (Page delay: {args.page_delay}s)")
    print(f"  Output Format  : Parquet (ZSTD)")
    print(f"  Output Base    : {MARKET_DATA_DIR}")

    # Step 1: Scan local status
    already_covered = []
    need_fetch = []
    for t in targets:
        covered, _ = check_local_status(t, start_date, end_date)
        if covered:
            already_covered.append(t)
        else:
            need_fetch.append(t)

    print(f"\n--- Pre-flight Local Status ---")
    print(f"  Already Up to Date : {len(already_covered)} symbols (will be skipped)")
    print(f"  Need Fetch / Update: {len(need_fetch)} symbols")

    if args.dry_run:
        print("\n[Dry-run] Need fetch tickers:")
        for idx, t in enumerate(need_fetch, 1):
            print(f"  {idx:3d}. {t}")
        return 0

    if not need_fetch:
        print("\nAll QQQ constituents are already 100% up to date in local Parquet!")
        return 0

    # Step 2: Initialize Futu OpenD
    try:
        import futu
    except ImportError:
        print("Error: futu-api is required. Please install futu-api.", file=sys.stderr)
        return 1

    futu.SysConfig.enable_proto_encrypt(False)
    ctx = futu.OpenQuoteContext(host=args.host, port=args.port)

    # Inspect Quota
    ret, quota_info = ctx.get_history_kl_quota(get_detail=True)
    if ret == futu.RET_OK:
        used, total, detail = quota_info
        consumed_codes = {item["code"] for item in detail} if isinstance(detail, list) else set()
        print(f"\nFutu Quota: used={used}/{total}, currently consumed symbols={len(consumed_codes)}")
    else:
        consumed_codes = set()
        print(f"\n[Warning] Could not inspect quota details: {quota_info}")

    success_count = 0
    failed_count = 0
    total_new_bars = 0
    start_total_t = time.time()

    try:
        for idx, ticker in enumerate(need_fetch, 1):
            code = f"US.{ticker}"
            covered, local_df = check_local_status(ticker, start_date, end_date)
            if covered:
                print(f"[{idx}/{len(need_fetch)}] {ticker}: Already up to date. Skipping.")
                continue

            # Polite throttling delay
            time.sleep(args.delay)

            # Paginated fetch with session=ALL and extended_time=True
            page_key = None
            page_records = []
            page_num = 0
            has_error = False

            while True:
                if page_num > 0:
                    time.sleep(args.page_delay)

                ret, df, next_key = ctx.request_history_kline(
                    code=code,
                    start=start_date,
                    end=end_date,
                    ktype=futu.KLType.K_60M,
                    autype=futu.AuType.QFQ,
                    fields=futu.KL_FIELD.ALL,
                    max_count=1000,
                    page_req_key=page_key,
                    extended_time=True,
                    session=futu.Session.ALL,
                )

                if ret != futu.RET_OK:
                    msg = str(df)
                    print(f"[{idx}/{len(need_fetch)}] {ticker} FAILED: {msg[:200]}", file=sys.stderr)
                    has_error = True
                    if any(w in msg.lower() for w in ("frequency", "rate limit", "频率", "quota", "额度")):
                        print("Stopping execution to respect Futu frequency / quota limits.", file=sys.stderr)
                        return 2
                    break

                if df is not None and not df.empty:
                    page_records.append(df)

                page_num += 1
                if not next_key:
                    break
                page_key = next_key

            if has_error or not page_records:
                failed_count += 1
                continue

            new_df = pd.concat(page_records, ignore_index=True)
            if local_df is not None and not local_df.empty:
                combined_df = pd.concat([local_df, new_df], ignore_index=True)
            else:
                combined_df = new_df

            saved_path = save_parquet(ticker, combined_df)
            total_new_bars += len(new_df)
            success_count += 1
            print(f"[{idx}/{len(need_fetch)}] {ticker}: +{len(new_df)} bars (Total {len(combined_df)} in {saved_path.name})")

    finally:
        ctx.close()

    elapsed = time.time() - start_total_t
    print(f"\n=== Run Summary ===")
    print(f"  Successfully Fetched : {success_count}/{len(need_fetch)} symbols")
    print(f"  Failed / Skipped     : {failed_count} symbols")
    print(f"  New Bars Collected   : {total_new_bars}")
    print(f"  Total Time Taken     : {elapsed:.2f}s")
    print(f"  Storage Target       : market_data/us_60m/<TICKER>/2026.parquet")
    print(f"\n[Next Step Reminder]")
    print(f"  Data is saved strictly locally. To upload to Cloudflare R2:")
    print(f"    python3 scripts/r2_sync.py push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
