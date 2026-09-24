#!/usr/bin/env python3
"""Fetch market data with hierarchical fallback strategy using Parquet format.

Strategy:
  1. Check local Parquet cache (market_data/us_60m/<SYMBOL>/2026.parquet)
  2. If missing or incomplete, check and download Parquet from Cloudflare R2
  3. If still missing from R2, fallback to Futu OpenD (futud)
  4. IMPORTANT: Fetched Futu data is saved locally only; NEVER automatically
     synced to R2. Manual sync must be triggered via `scripts/r2_sync.py push`.

Examples:
    # Fetch data for AAPL and NVDA
    python3 scripts/fetch_market_data.py --symbols AAPL NVDA

    # Fetch for specific date range
    python3 scripts/fetch_market_data.py --symbols TSLA --start 2026-09-01 --end 2026-09-24
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.r2_client import R2Client


FLOAT_COLS = ["open", "high", "low", "close", "volume", "turnover", "pe_ratio", "turnover_rate", "change_rate", "last_close"]


def load_local_bars(ticker: str, year: int = 2026, base_dir: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Load local Parquet bars (or fallback to gzip json)."""
    target_dir = base_dir or (ROOT / "market_data" / "us_60m")
    parquet_p = target_dir / ticker.upper() / f"{year}.parquet"
    if parquet_p.exists():
        try:
            return pd.read_parquet(parquet_p)
        except Exception as e:
            print(f"[Warning] Failed to read Parquet {parquet_p}: {e}", file=sys.stderr)

    # Fallback to json.gz
    gz_p = target_dir / ticker.upper() / f"{year}.json.gz"
    if gz_p.exists():
        try:
            with gzip.open(gz_p, "rt", encoding="utf-8") as f:
                d = json.load(f)
            bars = d.get("bars", [])
            if bars:
                df = pd.DataFrame(bars)
                return df
        except Exception:
            pass

    return None


def save_local_bars(ticker: str, df: pd.DataFrame, year: int = 2026, base_dir: Optional[Path] = None) -> Path:
    """Save bars locally in standard Parquet format (ZSTD compressed)."""
    target_dir = base_dir or (ROOT / "market_data" / "us_60m")
    ticker_dir = target_dir / ticker.upper()
    ticker_dir.mkdir(parents=True, exist_ok=True)
    parquet_p = ticker_dir / f"{year}.parquet"

    # Normalize types
    if "time_key" in df.columns:
        df["time_key"] = df["time_key"].astype(str)
    for c in FLOAT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    # Sort by time
    if "time_key" in df.columns:
        df = df.sort_values("time_key").drop_duplicates(subset=["time_key"], keep="last")

    temp_p = parquet_p.with_suffix(".parquet.tmp")
    df.to_parquet(temp_p, engine="pyarrow", compression="zstd")
    temp_p.replace(parquet_p)
    return parquet_p


def try_download_from_r2(ticker: str, year: int = 2026, r2_client: Optional[R2Client] = None, base_dir: Optional[Path] = None) -> bool:
    """Check R2 for Parquet file and download if exists."""
    client = r2_client or R2Client()
    remote_key = f"market_data/us_60m/{ticker.upper()}/{year}.parquet"
    target_dir = base_dir or (ROOT / "market_data" / "us_60m")
    dest_path = target_dir / ticker.upper() / f"{year}.parquet"

    head = client.head_object(remote_key)
    if head is not None:
        print(f"  [R2 Hit] Found {remote_key} on R2 ({head['size']} bytes). Downloading to local...")
        client.get_object(remote_key, dest_path)
        return True

    return False


def fetch_from_futu(
    ticker: str,
    start_date: str,
    end_date: str,
    futu_ctx: Any,
) -> pd.DataFrame:
    """Fetch 60m bars from Futu OpenD."""
    import futu
    print(f"  [Futu Fetch] Pulling {ticker} ({start_date} -> {end_date}) from Futu OpenD...")
    code = f"US.{ticker.upper()}"
    ret, df, _ = futu_ctx.request_history_kline(
        code=code,
        start=start_date,
        end=end_date,
        ktype=futu.KLType.K_60M,
        autype=futu.AuType.QFQ,
        fields=futu.KL_FIELD.ALL,
        max_count=1000,
        extended_time=True,
        session=futu.Session.ALL,
    )
    if ret != futu.RET_OK:
        raise RuntimeError(f"Futu OpenD request failed for {code}: {df}")

    if df is None or df.empty:
        return pd.DataFrame()

    return df


def ensure_symbol_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    r2_client: Optional[R2Client] = None,
    futu_ctx: Optional[Any] = None,
) -> Dict[str, Any]:
    """Hierarchical fetch for a single symbol: Local Parquet -> R2 Parquet -> Futu OpenD."""
    ticker = ticker.upper().removeprefix("US.")
    today_ny = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    req_start = start_date or "2026-01-01"
    req_end = end_date or today_ny
    year = int(req_end[:4])

    print(f"\nProcessing [{ticker}] (Target window: {req_start} -> {req_end}):")

    # Step 1: Check Local Cache (Parquet)
    df = load_local_bars(ticker, year=year)
    if df is not None and not df.empty and "time_key" in df.columns:
        first_ts = str(df["time_key"].min())[:10]
        last_ts = str(df["time_key"].max())[:10]
        if first_ts <= req_start and last_ts >= req_end:
            print(f"  [Local Hit] Up to date in local Parquet ({len(df)} bars, {first_ts} ~ {last_ts}).")
            return {"source": "local", "ticker": ticker, "bars_count": len(df), "status": "cached"}

    # Step 2: Check Cloudflare R2 (Parquet)
    print(f"  [Local Miss / Incomplete] Checking Cloudflare R2 for {ticker} Parquet...")
    r2_downloaded = try_download_from_r2(ticker, year=year, r2_client=r2_client)
    if r2_downloaded:
        df = load_local_bars(ticker, year=year)
        if df is not None and not df.empty and "time_key" in df.columns:
            first_ts = str(df["time_key"].min())[:10]
            last_ts = str(df["time_key"].max())[:10]
            if first_ts <= req_start and last_ts >= req_end:
                print(f"  [R2 Complete] Retrieved from R2 Parquet ({len(df)} bars, {first_ts} ~ {last_ts}).")
                return {"source": "r2", "ticker": ticker, "bars_count": len(df), "status": "downloaded_r2"}

    # Step 3: Fallback to Futu OpenD
    print(f"  [R2 Miss / Incomplete] Falling back to Futu OpenD...")
    owns_futu_ctx = False
    if futu_ctx is None:
        try:
            import futu
            futu.SysConfig.enable_proto_encrypt(False)
            futu_ctx = futu.OpenQuoteContext(host="127.0.0.1", port=11111)
            owns_futu_ctx = True
        except Exception as e:
            raise RuntimeError(f"Could not connect to Futu OpenD at 127.0.0.1:11111: {e}") from e

    try:
        futu_df = fetch_from_futu(ticker, req_start, req_end, futu_ctx)
        if df is not None and not df.empty:
            merged_df = pd.concat([df, futu_df], ignore_index=True)
        else:
            merged_df = futu_df

        saved_path = save_local_bars(ticker, merged_df, year=year)
        added_bars = len(futu_df)
        total_bars = len(merged_df)

        print(f"  [Futu Saved] Fetched {added_bars} bars. Saved to {saved_path.name} (Total: {total_bars} bars).")
        print(f"  [Manual Sync Notice] Parquet saved to local disk only. NOT automatically pushed to R2.")
        print(f"    -> Run `python3 scripts/r2_sync.py push --symbols {ticker}` when ready to sync.")

        return {
            "source": "futu",
            "ticker": ticker,
            "added_bars": added_bars,
            "total_bars": total_bars,
            "status": "saved_local_only",
        }
    finally:
        if owns_futu_ctx and futu_ctx:
            futu_ctx.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", required=True, help="Stock tickers to ensure (e.g. AAPL NVDA)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (default: 2026-01-01)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (default: today NY)")
    args = parser.parse_args()

    r2_client = R2Client()
    futu_ctx = None
    try:
        import futu
        futu.SysConfig.enable_proto_encrypt(False)
        futu_ctx = futu.OpenQuoteContext(host="127.0.0.1", port=11111)
    except Exception:
        pass

    results = []
    try:
        for symbol in args.symbols:
            res = ensure_symbol_data(
                symbol,
                start_date=args.start,
                end_date=args.end,
                r2_client=r2_client,
                futu_ctx=futu_ctx,
            )
            results.append(res)
    finally:
        if futu_ctx:
            futu_ctx.close()

    print("\n=== Fetch Summary ===")
    for r in results:
        print(f"  {r['ticker']}: source={r['source']}, status={r['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
