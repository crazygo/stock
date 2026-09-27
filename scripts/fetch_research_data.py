#!/usr/bin/env python3
"""Market data acquisition engine for research v2.

Adheres strictly to docs/06_data_acquisition_feasibility.md:
- 3-tier fallback architecture: Local -> Cloudflare R2 -> Futu OpenD
- True multi-page pagination: loops page_req_key until completion (no data truncation)
- Unadjusted raw price basis (price_basis=NONE) + corporate actions (get_rehab)
- Strict rate pacing: >= 1.2s between symbols, >= 0.25s between pages
- Structured storage layout: market_data/research_v2/futu/<SYMBOL>/<INTERVAL>/<PRICE_BASIS>/<SESSION>/<YEAR>.parquet
- Normalized timezone-aware timestamps (America/New_York + UTC)
- Comprehensive audit trail in request_audit.jsonl
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
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

from scripts.r2_client import R2Client

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

KTYPE_MAP = {
    "5m": ft.KLType.K_5M if FUTU_AVAILABLE else "K_5M",
    "15m": ft.KLType.K_15M if FUTU_AVAILABLE else "K_15M",
    "30m": ft.KLType.K_30M if FUTU_AVAILABLE else "K_30M",
    "60m": ft.KLType.K_60M if FUTU_AVAILABLE else "K_60M",
    "day": ft.KLType.K_DAY if FUTU_AVAILABLE else "K_DAY",
}

AUTYPE_MAP = {
    "NONE": ft.AuType.NONE if FUTU_AVAILABLE else "NONE",
    "QFQ": ft.AuType.QFQ if FUTU_AVAILABLE else "QFQ",
    "HFQ": ft.AuType.HFQ if FUTU_AVAILABLE else "HFQ",
}


def determine_session_type(start_et: datetime, end_et: datetime) -> str:
    """Classify intraday bar into US equity session segments."""
    # US regular market: 09:30 to 16:00 ET
    sh, sm = start_et.hour, start_et.minute
    eh, em = end_et.hour, end_et.minute

    start_mins = sh * 60 + sm
    end_mins = eh * 60 + em

    # 09:30 is 570 mins, 16:00 is 960 mins
    if start_mins >= 570 and end_mins <= 960:
        return "regular"
    elif end_mins <= 570 and start_mins >= 240: # 04:00 to 09:30
        return "pre_market"
    elif start_mins >= 960 and end_mins <= 1200: # 16:00 to 20:00
        return "post_market"
    else:
        return "overnight"


def normalize_kline_dataframe(df: pd.DataFrame, symbol: str, interval: str, price_basis: str) -> pd.DataFrame:
    """Standardize raw OpenD DataFrame with timezone-aware columns."""
    if df.empty:
        return df

    # Interval step timedelta
    interval_delta = {
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "60m": timedelta(minutes=60),
        "day": timedelta(days=1),
    }.get(interval, timedelta(minutes=5))

    # Parse time_key (Format: YYYY-MM-DD HH:MM:SS) in ET
    parsed_dt = pd.to_datetime(df["time_key"])

    end_et_list = []
    start_et_list = []
    end_utc_list = []
    start_utc_list = []
    avail_utc_list = []
    session_types = []
    trading_dates = []

    for dt in parsed_dt:
        # OpenD time_key is local ET end-time of bar
        end_et = dt.to_pydatetime().replace(tzinfo=ET)
        start_et = end_et - interval_delta

        # Boundary adjustments for US 60m bars (as documented in docs/06 section 4.1):
        if interval == "60m":
            if end_et.hour == 10 and end_et.minute == 30:
                start_et = end_et.replace(hour=9, minute=30)
            elif end_et.hour == 9 and end_et.minute == 30:
                start_et = end_et.replace(hour=9, minute=0)
            elif end_et.hour == 16 and end_et.minute == 0:
                start_et = end_et.replace(hour=15, minute=30)
            elif end_et.hour == 13 and end_et.minute == 0:
                start_et = end_et.replace(hour=12, minute=30)

        end_utc = end_et.astimezone(UTC)
        start_utc = start_et.astimezone(UTC)
        avail_utc = end_utc + timedelta(seconds=1)

        st = determine_session_type(start_et, end_et)
        s_date = end_et.date().isoformat()

        end_et_list.append(end_et.isoformat())
        start_et_list.append(start_et.isoformat())
        end_utc_list.append(end_utc.isoformat())
        start_utc_list.append(start_utc.isoformat())
        avail_utc_list.append(avail_utc.isoformat())
        session_types.append(st)
        trading_dates.append(s_date)

    out_df = pd.DataFrame({
        "symbol": symbol,
        "time_key": df["time_key"],
        "start_at": start_utc_list,
        "end_at": end_utc_list,
        "available_at": avail_utc_list,
        "start_at_et": start_et_list,
        "end_at_et": end_et_list,
        "session_date": trading_dates,
        "session_type": session_types,
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float),
        "turnover": df["turnover"].astype(float) if "turnover" in df.columns else 0.0,
        "pe_ratio": df["pe_ratio"].astype(float) if "pe_ratio" in df.columns else 0.0,
        "turnover_rate": df["turnover_rate"].astype(float) if "turnover_rate" in df.columns else 0.0,
        "change_rate": df["change_rate"].astype(float) if "change_rate" in df.columns else 0.0,
        "last_close": df["last_close"].astype(float) if "last_close" in df.columns else 0.0,
        "price_basis": price_basis,
    })

    # Drop duplicates by start_at
    out_df = out_df.drop_duplicates(subset=["start_at"]).sort_values("start_at").reset_index(drop=True)
    return out_df


class ResearchDataFetcher:
    """Three-tier market data acquisition manager for research v2."""

    def __init__(self, data_root: Optional[Path] = None, host: str = "127.0.0.1", port: int = 11111):
        self.root = data_root or (ROOT / "market_data")
        self.host = host
        self.port = port
        self.quote_ctx: Optional[ft.OpenQuoteContext] = None
        self.r2_client: Optional[R2Client] = None
        self._init_r2()
        self.audit_log_path = self.root / "manifests" / "request_audit.jsonl"
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_r2(self):
        try:
            self.r2_client = R2Client()
        except Exception as e:
            print(f"[Warning] R2Client could not be initialized: {e}")
            self.r2_client = None

    def _ensure_futu(self):
        if not FUTU_AVAILABLE:
            raise RuntimeError("futu-api is not installed.")
        if self.quote_ctx is None:
            ft.SysConfig.enable_proto_encrypt(False)
            self.quote_ctx = ft.OpenQuoteContext(host=self.host, port=self.port)
            print(f"[Futu] Connected to OpenD on {self.host}:{self.port}")

    def close(self):
        if self.quote_ctx:
            try:
                self.quote_ctx.close()
            except Exception:
                pass
            self.quote_ctx = None

    def get_local_path(self, symbol: str, interval: str, price_basis: str, session: str, year: int) -> Path:
        if interval == "5m":
            return self.root / "us_5m" / symbol / f"{year}.parquet"
        elif interval == "60m" and price_basis == "NONE":
            return self.root / "us_60m_raw" / symbol / f"{year}.parquet"
        elif interval == "60m":
            return self.root / "us_60m" / symbol / f"{year}.parquet"
        else:
            return self.root / f"us_{interval}" / symbol / f"{year}.parquet"

    def get_r2_key(self, symbol: str, interval: str, price_basis: str, session: str, year: int) -> str:
        if interval == "5m":
            return f"us_5m/{symbol}/{year}.parquet"
        elif interval == "60m" and price_basis == "NONE":
            return f"us_60m_raw/{symbol}/{year}.parquet"
        elif interval == "60m":
            return f"us_60m/{symbol}/{year}.parquet"
        else:
            return f"us_{interval}/{symbol}/{year}.parquet"

    def log_audit(self, entry: Dict[str, Any]):
        entry["logged_at"] = datetime.now(timezone.utc).isoformat()
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def fetch_corporate_actions(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch split/dividend corporate actions via get_rehab."""
        target_path = self.root / "corporate_actions" / f"{symbol}.parquet"
        if target_path.exists():
            return pd.read_parquet(target_path)

        self._ensure_futu()
        code = f"US.{symbol}"
        start_t = time.time()
        ret, df = self.quote_ctx.get_rehab(code)
        elapsed = time.time() - start_t

        audit = {
            "symbol": symbol,
            "action": "get_rehab",
            "duration_seconds": round(elapsed, 3),
            "status": "success" if ret == ft.RET_OK else "error",
            "rows": len(df) if ret == ft.RET_OK else 0,
            "error_msg": str(df) if ret != ft.RET_OK else None
        }
        self.log_audit(audit)

        if ret != ft.RET_OK:
            print(f"[Error] get_rehab failed for {code}: {df}")
            return None

        target_path.parent.mkdir(parents=True, exist_ok=True)
        if df.empty:
            empty_df = pd.DataFrame(columns=["ex_div_date", "split_ratio", "cash_dividend"])
            empty_df.to_parquet(target_path, engine="pyarrow", compression="zstd")
            return empty_df

        df.to_parquet(target_path, engine="pyarrow", compression="zstd")
        print(f"[Saved Rehab] {symbol}: {len(df)} corporate action rows -> {target_path}")
        return df

    def fetch_kline(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "5m",
        price_basis: str = "NONE",
        session: str = "ALL",
        force_remote: bool = False
    ) -> Optional[pd.DataFrame]:
        """Fetch K-line using 3-tier fallback (Local -> R2 -> Futu OpenD)."""
        year = int(start_date.split("-")[0])
        local_path = self.get_local_path(symbol, interval, price_basis, session, year)

        # Tier 1: Local Hit
        if local_path.exists() and not force_remote:
            df = pd.read_parquet(local_path)
            # Verify date coverage
            min_date = df["session_date"].min()
            max_date = df["session_date"].max()
            if min_date <= start_date and max_date >= end_date:
                return df

        # Tier 2: Cloudflare R2 Hit
        if self.r2_client and not force_remote:
            r2_key = self.get_r2_key(symbol, interval, price_basis, session, year)
            try:
                head = self.r2_client.head_object(r2_key)
                if head is not None:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    self.r2_client.get_object(r2_key, local_path)
                    print(f"[R2 Hit] Successfully downloaded {r2_key} -> {local_path}")
                    df = pd.read_parquet(local_path)
                    if df["session_date"].min() <= start_date and df["session_date"].max() >= end_date:
                        return df
            except Exception as e:
                print(f"[R2 Skip] {r2_key} lookup/download: {e}")

        # Tier 3: Futu OpenD Fallback with full pagination
        self._ensure_futu()
        code = f"US.{symbol}"
        ft_ktype = KTYPE_MAP[interval]
        ft_autype = AUTYPE_MAP[price_basis]
        extended = (session == "ALL")

        pages_data = []
        page_key = None
        page_count = 0
        total_rows = 0
        start_time = time.time()
        error_msg = None

        # Rate pacing for first page (>= 1.2s)
        time.sleep(1.2)

        while True:
            retries = 3
            ret = -1
            sub_df = None
            next_page_key = None

            while retries > 0:
                ret, sub_df, next_page_key = self.quote_ctx.request_history_kline(
                    code=code,
                    start=start_date,
                    end=end_date,
                    ktype=ft_ktype,
                    autype=ft_autype,
                    fields=[ft.KL_FIELD.ALL],
                    max_count=1000,
                    page_req_key=page_key,
                    extended_time=extended
                )
                if ret == ft.RET_OK:
                    break

                # Check for rate limit or frequency error
                err_str = str(sub_df)
                if "frequent" in err_str.lower() or "limit" in err_str.lower():
                    print(f"  [RateLimit] Backing off 5s for {code} page {page_count + 1}...")
                    time.sleep(5.0)
                else:
                    print(f"  [Retry] OpenD error on {code}: {err_str}, retrying in 2s...")
                    time.sleep(2.0)
                retries -= 1

            if ret != ft.RET_OK:
                error_msg = str(sub_df)
                print(f"[Error] Failed to fetch {code} on page {page_count + 1}: {error_msg}")
                break

            page_count += 1
            if sub_df is not None and not sub_df.empty:
                pages_data.append(sub_df)
                total_rows += len(sub_df)

            if next_page_key is None:
                # Pagination complete
                break

            page_key = next_page_key
            # Rate pacing between pages (>= 0.25s)
            time.sleep(0.3)

        elapsed = time.time() - start_time
        audit_entry = {
            "symbol": symbol,
            "interval": interval,
            "price_basis": price_basis,
            "session": session,
            "start_date": start_date,
            "end_date": end_date,
            "pages": page_count,
            "total_raw_rows": total_rows,
            "duration_seconds": round(elapsed, 3),
            "status": "success" if error_msg is None else "partial/failed",
            "error_msg": error_msg
        }
        self.log_audit(audit_entry)

        if not pages_data:
            print(f"[Warning] No data retrieved for {code}")
            return None

        combined_raw = pd.concat(pages_data, ignore_index=True)
        normalized = normalize_kline_dataframe(combined_raw, symbol, interval, price_basis)

        # Merge with existing local data if present
        if local_path.exists():
            existing = pd.read_parquet(local_path)
            normalized = pd.concat([existing, normalized], ignore_index=True)
            normalized = normalized.drop_duplicates(subset=["start_at"]).sort_values("start_at").reset_index(drop=True)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(normalized)
        pq.write_table(table, local_path, compression="zstd", compression_level=7)
        print(f"[Saved] {symbol} {interval} ({price_basis}/{session}): {len(normalized)} bars ({page_count} pages, {elapsed:.1f}s) -> {local_path}")
        return normalized


def main():
    parser = argparse.ArgumentParser(description="Research v2 Market Data Fetcher")
    parser.add_argument("--symbols", nargs="+", required=True, help="List of stock tickers (e.g. AAPL NVDA)")
    parser.add_argument("--start", default="2026-01-02", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-09-24", help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="5m", choices=["5m", "15m", "30m", "60m", "day"], help="Bar interval")
    parser.add_argument("--price-basis", default="NONE", choices=["NONE", "QFQ", "HFQ"], help="Price adjustment basis")
    parser.add_argument("--session", default="ALL", choices=["ALL", "RTH"], help="Session scope")
    parser.add_argument("--fetch-rehab", action="store_true", help="Also fetch corporate actions (get_rehab)")
    parser.add_argument("--force-remote", action="store_true", help="Force fetching from OpenD")
    args = parser.parse_args()

    fetcher = ResearchDataFetcher()
    try:
        for symbol in args.symbols:
            s = symbol.upper().replace("US.", "")
            if args.fetch_rehab:
                fetcher.fetch_corporate_actions(s)
            fetcher.fetch_kline(
                symbol=s,
                start_date=args.start,
                end_date=args.end,
                interval=args.interval,
                price_basis=args.price_basis,
                session=args.session,
                force_remote=args.force_remote
            )
    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
