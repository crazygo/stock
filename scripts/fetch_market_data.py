#!/usr/bin/env python3
"""Fetch market data with hierarchical fallback strategy.

Strategy:
  1. Check local cache (market_data/us_60m/<SYMBOL>/...)
  2. If missing or incomplete, check and download from Cloudflare R2
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
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.r2_client import R2Client


def load_local_bars(ticker: str, year: int = 2026, base_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load local gzip json bars for a ticker."""
    target_dir = base_dir or (ROOT / "market_data" / "us_60m")
    file_path = target_dir / ticker.upper() / f"{year}.json.gz"
    if not file_path.exists():
        return None
    try:
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to read local {file_path}: {e}", file=sys.stderr)
        return None


def save_local_bars(ticker: str, payload: Dict[str, Any], year: int = 2026, base_dir: Optional[Path] = None) -> Path:
    """Save bars locally in gzip json format."""
    target_dir = base_dir or (ROOT / "market_data" / "us_60m")
    file_path = target_dir / ticker.upper() / f"{year}.json.gz"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = file_path.with_suffix(".json.gz.tmp")
    with gzip.open(temp_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    temp_path.replace(file_path)
    return file_path


def try_download_from_r2(ticker: str, year: int = 2026, r2_client: Optional[R2Client] = None, base_dir: Optional[Path] = None) -> bool:
    """Check R2 and download if exists."""
    client = r2_client or R2Client()
    remote_key = f"market_data/us_60m/{ticker.upper()}/{year}.json.gz"
    target_dir = base_dir or (ROOT / "market_data" / "us_60m")
    dest_path = target_dir / ticker.upper() / f"{year}.json.gz"

    head = client.head_object(remote_key)
    if head is None:
        return False

    print(f"  [R2 Hit] Found {remote_key} on R2 ({head['size']} bytes). Downloading to local...")
    client.get_object(remote_key, dest_path)
    return True


def fetch_from_futu(
    ticker: str,
    start_date: str,
    end_date: str,
    futu_ctx: Any,
) -> List[Dict[str, Any]]:
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
        return []

    records = df.to_dict(orient="records")
    clean_bars = []
    for r in records:
        clean = {k: (None if str(v) == "nan" else v) for k, v in r.items()}
        clean_bars.append(clean)
    return clean_bars


def ensure_symbol_data(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    r2_client: Optional[R2Client] = None,
    futu_ctx: Optional[Any] = None,
) -> Dict[str, Any]:
    """Hierarchical fetch for a single symbol: Local -> R2 -> Futu OpenD."""
    ticker = ticker.upper().removeprefix("US.")
    today_ny = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    req_start = start_date or "2026-01-01"
    req_end = end_date or today_ny
    year = int(req_end[:4])

    print(f"\nProcessing [{ticker}] (Target window: {req_start} -> {req_end}):")

    # Step 1: Check Local Cache
    data = load_local_bars(ticker, year=year)
    if data:
        bars = data.get("bars", [])
        timestamps = [b["time_key"] for b in bars if "time_key" in b]
        if timestamps:
            first_ts = timestamps[0][:10]
            last_ts = timestamps[-1][:10]
            if first_ts <= req_start and last_ts >= req_end:
                print(f"  [Local Hit] Up to date locally ({len(bars)} bars, {first_ts} ~ {last_ts}).")
                return {"source": "local", "ticker": ticker, "bars_count": len(bars), "status": "cached"}

    # Step 2: Check Cloudflare R2
    print(f"  [Local Miss / Incomplete] Checking Cloudflare R2...")
    r2_downloaded = try_download_from_r2(ticker, year=year, r2_client=r2_client)
    if r2_downloaded:
        data = load_local_bars(ticker, year=year)
        if data:
            bars = data.get("bars", [])
            timestamps = [b["time_key"] for b in bars if "time_key" in b]
            if timestamps:
                first_ts = timestamps[0][:10]
                last_ts = timestamps[-1][:10]
                if first_ts <= req_start and last_ts >= req_end:
                    print(f"  [R2 Complete] Retrieved from R2 ({len(bars)} bars, {first_ts} ~ {last_ts}).")
                    return {"source": "r2", "ticker": ticker, "bars_count": len(bars), "status": "downloaded_r2"}

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
        futu_bars = fetch_from_futu(ticker, req_start, req_end, futu_ctx)
        existing_bars = data.get("bars", []) if data else []
        existing_keys = {b.get("time_key") for b in existing_bars}

        added = 0
        for b in futu_bars:
            if b.get("time_key") not in existing_keys:
                existing_bars.append(b)
                existing_keys.add(b.get("time_key"))
                added += 1

        existing_bars.sort(key=lambda x: str(x.get("time_key", "")))
        payload = {
            "ticker": ticker,
            "year": year,
            "interval": "60m",
            "start": req_start,
            "end": req_end,
            "adjustment": "qfq",
            "extended_time": True,
            "session": "ALL",
            "bars": existing_bars,
        }
        save_local_bars(ticker, payload, year=year)
        print(f"  [Futu Saved] Fetched {len(futu_bars)} bars (+{added} new). Total local bars: {len(existing_bars)}.")
        print(f"  [Manual Sync Notice] Data saved to local disk only. NOT automatically pushed to R2.")
        print(f"    -> Run `python3 scripts/r2_sync.py push --symbols {ticker}` when ready to sync.")

        return {
            "source": "futu",
            "ticker": ticker,
            "added_bars": added,
            "total_bars": len(existing_bars),
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
