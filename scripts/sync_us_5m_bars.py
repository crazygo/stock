#!/usr/bin/env python3
"""Standard US Equity 5-Minute (5m) K-Line Synchronization Engine.

Strictly follows market_data/us_5m/AGENTS.md:
- Target path: market_data/us_5m/<SYMBOL>/2026.parquet
- Format: Apache Parquet (ZSTD compression, level 7)
- K-line: 5-minute bars (ktype=ft.KLType.K_5M)
- Price basis: Unadjusted raw prices (autype=ft.AuType.NONE, price_basis="NONE")
- Session coverage: Full 24-hour sessions (extended_time=True: regular, pre_market, post_market, overnight)
- Deduplication: Merges seamlessly with existing bars, deduplicating by start_at (UTC)
- Pacing: >= 1.0s delay between symbols to strictly avoid Futu OpenD rate-limiting

Usage Examples:
    # 1. Incremental sync for specific tickers from Futu OpenD
    python3 scripts/sync_us_5m_bars.py --symbols AAPL NVDA CRDO

    # 2. Sync all 138 universe tickers (QQQ + Favorites + 0086 Holdings + ETFs)
    python3 scripts/sync_us_5m_bars.py --all-universe

    # 3. Pull 5m data from Cloudflare R2 first
    python3 scripts/sync_us_5m_bars.py --r2-pull --symbols AAPL NVDA

    # 4. Dry run to inspect latest bar timestamps without writing
    python3 scripts/sync_us_5m_bars.py --symbols CRDO --dry-run
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Optional Futu OpenD import
try:
    import futu as ft
    FUTU_AVAILABLE = True
except ImportError:
    FUTU_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.fetch_research_data import normalize_kline_dataframe
from scripts.r2_client import R2Client

M5_DIR = ROOT / "market_data" / "us_5m"
QQQ_FILE = ROOT / "analysis" / "qqq_constituents.json"

# Standard 138-ticker universe
FAVORITES_TICKERS = {
    'WDC', 'STX', 'TER', 'PLTR', 'NBIS', 'AVGO', 'AAOZ', 'CRWV', 'AMAT', 'LITX',
    'CBRS', 'RMBS', 'IONL', 'LRNZ', 'AAOI', 'VRT', 'AIPO', 'CIEN', 'SOXS', 'MU',
    'CRDO', 'AMD', 'SMTC', 'ARM', 'TXG', 'TWST', 'SDGR', 'QCOM', 'COHR', 'ALAB',
    'LITE', 'MRVL', 'LIFE', 'NOK', 'KOD'
}

HOLDINGS_0086 = {
    'SNXX', 'RKLB', 'NXT', 'NVTS', 'INTC', 'GOOG', 'FN', 'COHR', 'AVGO', 'AMZN',
    'AIPO', 'CRDO', 'MRVL', 'AMAT', 'HLTH', 'CIEN', 'VRT', 'LITX', 'SEDG', 'BBC', 'AXTI'
}

CORE_ETFS = {
    'QQQ', 'SPY', 'DIA', 'SOXX', 'SMH', 'SOXL', 'SOXS', 'IGV', 'SPCX', 'XLU'
}

SEMI_HARDWARE = {
    'NVDA', 'AVGO', 'AMD', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'MU', 'INTC', 'ASML',
    'KLAC', 'MRVL', 'ADI', 'NXPI', 'MCHP', 'MPWR', 'TER', 'RMBS', 'SMTC', 'ALAB',
    'SNDK', 'COHR', 'LITE', 'CRDO', 'VRT', 'NVTS', 'AXTI', 'AAOI', 'AAOZ', 'CBRS',
    'WDC', 'STX', 'SOXX', 'SMH', 'SOXL', 'SOXS', 'FN', 'NXT'
}


def get_default_universe() -> List[str]:
    """Retrieve the full 138-ticker universe."""
    tickers: Set[str] = set()
    if QQQ_FILE.exists():
        try:
            tickers.update(json.loads(QQQ_FILE.read_text())['tickers'])
        except Exception:
            pass
    tickers.update(FAVORITES_TICKERS)
    tickers.update(HOLDINGS_0086)
    tickers.update(CORE_ETFS)
    tickers.update(SEMI_HARDWARE)
    return sorted(tickers)


def get_local_latest_date(symbol: str, year: int = 2026) -> Optional[str]:
    """Get the latest session_date or time_key from local 5m parquet file."""
    p = M5_DIR / symbol.upper() / f"{year}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p, columns=["time_key"])
        if not df.empty:
            max_t = str(df["time_key"].max())
            return max_t[:10]
    except Exception:
        pass
    return None


def sync_from_r2(symbol: str, year: int = 2026, client: Optional[R2Client] = None) -> bool:
    """Pull 5m parquet file from Cloudflare R2 bucket."""
    client = client or R2Client()
    key = f"us_5m/{symbol.upper()}/{year}.parquet"
    local_p = M5_DIR / symbol.upper() / f"{year}.parquet"

    try:
        data = client.get_object(key)
        if data:
            local_p.parent.mkdir(parents=True, exist_ok=True)
            local_p.write_bytes(data)
            print(f"[{symbol:6s}] 📥 Downloaded {len(data)/1024:.1f} KB from R2 ({key})")
            return True
        else:
            print(f"[{symbol:6s}] ⚠️ Not found in R2 ({key})")
            return False
    except Exception as e:
        print(f"[{symbol:6s}] ❌ R2 error: {e}")
        return False


def sync_from_futu(
    ctx: ft.OpenQuoteContext,
    symbol: str,
    start_date: str,
    end_date: str,
    dry_run: bool = False,
    year: int = 2026
) -> Tuple[bool, int, str]:
    """Fetch incremental 5m bars from Futu OpenD and merge into local parquet."""
    code = f"US.{symbol.upper()}"
    local_path = M5_DIR / symbol.upper() / f"{year}.parquet"

    try:
        ret, raw_df, _ = ctx.request_history_kline(
            code=code,
            start=start_date,
            end=end_date,
            ktype=ft.KLType.K_5M,
            autype=ft.AuType.NONE,
            max_count=1000,
            extended_time=True,
        )

        if ret != ft.RET_OK:
            return False, 0, f"OpenD error: {raw_df}"

        if raw_df is None or raw_df.empty:
            return True, 0, "No new bars returned"

        # Normalize with timezone-aware columns
        norm_df = normalize_kline_dataframe(raw_df, symbol.upper(), "5m", "NONE")

        # Merge with existing
        if local_path.exists():
            existing_df = pd.read_parquet(local_path)
            merged_df = pd.concat([existing_df, norm_df], ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset=["start_at"]).sort_values("start_at").reset_index(drop=True)
        else:
            merged_df = norm_df

        new_bars_count = len(norm_df)
        total_bars = len(merged_df)
        max_time = str(merged_df["time_key"].max())

        if not dry_run:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(merged_df)
            pq.write_table(table, local_path, compression="zstd", compression_level=7)

        return True, new_bars_count, f"Merged {new_bars_count} bars (Total: {total_bars}, Latest: {max_time})"

    except Exception as e:
        return False, 0, f"Exception: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Sync US 5m K-line Parquet files from Futu OpenD / Cloudflare R2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to sync (e.g. AAPL NVDA CRDO)")
    parser.add_argument("--all-universe", action="store_true", help="Sync all 138 universe tickers")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD). If omitted, auto-detected from local parquet")
    parser.add_argument("--end", default=date.today().strftime("%Y-%m-%d"), help="End date (YYYY-MM-DD, default: today)")
    parser.add_argument("--r2-pull", action="store_true", help="Download missing/remote 5m parquet from Cloudflare R2")
    parser.add_argument("--dry-run", action="store_true", help="Query without writing to disk")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between symbol requests in seconds (default: 1.0s)")
    parser.add_argument("--host", default="127.0.0.1", help="Futu OpenD host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=11111, help="Futu OpenD port (default: 11111)")

    args = parser.parse_args()

    # Determine symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols]
    elif args.all_universe or not any([args.symbols, args.r2_pull]):
        symbols = get_default_universe()
        print(f"Targeting full universe: {len(symbols)} tickers")
    else:
        symbols = get_default_universe()

    print("=" * 70)
    print(f"🚀 US Equity 5m K-Line Sync Engine")
    print(f"   Target Tickers : {len(symbols)} symbols")
    print(f"   Date Range     : {args.start or '(auto-detected)'} -> {args.end}")
    print(f"   Mode           : {'R2 Pull' if args.r2_pull else 'Futu OpenD Sync'}{' (DRY RUN)' if args.dry_run else ''}")
    print("=" * 70)

    # R2 Pull Mode
    if args.r2_pull:
        client = R2Client()
        success = 0
        for s in symbols:
            ok = sync_from_r2(s, client=client)
            if ok:
                success += 1
        print(f"\n✨ R2 Pull Finished: {success}/{len(symbols)} files synchronized.")
        return

    # Futu OpenD Mode
    if not FUTU_AVAILABLE:
        print("❌ Error: futu-api is not installed. Run `pip install futu-api` first.")
        sys.exit(1)

    ft.SysConfig.enable_proto_encrypt(False)
    print(f"Connecting to Futu OpenD at {args.host}:{args.port}...")
    try:
        ctx = ft.OpenQuoteContext(host=args.host, port=args.port)
    except Exception as e:
        print(f"❌ Failed to connect to Futu OpenD: {e}")
        sys.exit(1)

    success_cnt = 0
    err_cnt = 0
    total_new_bars = 0
    t0 = time.time()

    try:
        for idx, symbol in enumerate(symbols, 1):
            # Auto-detect start date if not provided
            if args.start:
                s_date = args.start
            else:
                last_d = get_local_latest_date(symbol)
                if last_d:
                    # Sync from 2 days before last known date to ensure no intraday gaps
                    dt = datetime.strptime(last_d, "%Y-%m-%d") - timedelta(days=2)
                    s_date = dt.strftime("%Y-%m-%d")
                else:
                    # Default to 30 days ago
                    s_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            ok, new_bars, msg = sync_from_futu(
                ctx=ctx,
                symbol=symbol,
                start_date=s_date,
                end_date=args.end,
                dry_run=args.dry_run
            )

            status_icon = "✅" if ok else "❌"
            print(f"[{idx:3d}/{len(symbols)}] {status_icon} {symbol:6s} ({s_date} ~ {args.end}): {msg}")

            if ok:
                success_cnt += 1
                total_new_bars += new_bars
            else:
                err_cnt += 1

            # Frequency pacing
            time.sleep(args.delay)

    finally:
        ctx.close()

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"✨ Sync completed in {elapsed:.1f}s!")
    print(f"   Successful: {success_cnt}/{len(symbols)}")
    print(f"   Errors    : {err_cnt}")
    print(f"   New Bars  : {total_new_bars} bars added")
    print("=" * 70)


if __name__ == "__main__":
    main()
