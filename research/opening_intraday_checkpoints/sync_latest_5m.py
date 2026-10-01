#!/usr/bin/env python3
"""Sync latest 5m K-line data for QQQ constituents + QQQ from Futu OpenD (2026-09-24 to 2026-09-29).

Safely merges new bars into market_data/us_5m/<SYMBOL>/2026.parquet with deduplication.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import futu as ft
from scripts.fetch_research_data import normalize_kline_dataframe

QQQ_CONSTITUENTS_FILE = ROOT / "analysis" / "qqq_constituents.json"
MARKET_DATA_5M_DIR = ROOT / "market_data" / "us_5m"

START_DATE = "2026-09-24"
END_DATE = "2026-09-29"


def main():
    print(f"=== Starting 5m Market Data Sync ({START_DATE} to {END_DATE}) ===")
    
    with open(QQQ_CONSTITUENTS_FILE) as f:
        tickers = json.load(f)["tickers"]
    symbols = tickers + ["QQQ"]
    print(f"Total symbols to sync: {len(symbols)}")

    ft.SysConfig.enable_proto_encrypt(False)
    ctx = ft.OpenQuoteContext(host="127.0.0.1", port=11111)

    success_count = 0
    err_count = 0
    t_start = time.time()

    for idx, symbol in enumerate(symbols, 1):
        code = f"US.{symbol}"
        local_path = MARKET_DATA_5M_DIR / symbol / "2026.parquet"

        try:
            ret, raw_df, _ = ctx.request_history_kline(
                code=code,
                start=START_DATE,
                end=END_DATE,
                ktype=ft.KLType.K_5M,
                autype=ft.AuType.NONE,
                max_count=1000,
                extended_time=True,
            )

            if ret != ft.RET_OK:
                print(f"[{idx:3d}/{len(symbols)}] ❌ {symbol:6s}: OpenD error: {raw_df}")
                err_count += 1
                time.sleep(1.0)
                continue

            if raw_df is None or raw_df.empty:
                print(f"[{idx:3d}/{len(symbols)}] ⚠️ {symbol:6s}: No data returned")
                err_count += 1
                time.sleep(1.0)
                continue

            # Normalize dataframe
            norm_df = normalize_kline_dataframe(raw_df, symbol, "5m", "NONE")

            # Merge with existing file
            if local_path.exists():
                existing_df = pd.read_parquet(local_path)
                merged_df = pd.concat([existing_df, norm_df], ignore_index=True)
                merged_df = merged_df.drop_duplicates(subset=["start_at"]).sort_values("start_at").reset_index(drop=True)
            else:
                merged_df = norm_df

            # Save back with zstd
            local_path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pandas(merged_df)
            pq.write_table(table, local_path, compression="zstd", compression_level=7)

            success_count += 1
            max_t = merged_df["time_key"].max()
            if idx % 10 == 0 or idx == len(symbols):
                print(f"[{idx:3d}/{len(symbols)}] ✅ {symbol:6s}: {len(raw_df)} new bars merged. Total bars: {len(merged_df)}, latest: {max_t}")

        except Exception as e:
            print(f"[{idx:3d}/{len(symbols)}] 💥 {symbol:6s}: Exception: {e}")
            err_count += 1

        # Rate pacing: 0.9s delay between requests
        time.sleep(0.9)

    ctx.close()
    elapsed = time.time() - t_start
    print(f"\n=== Sync Complete in {elapsed:.1f}s ===")
    print(f"Success: {success_count}/{len(symbols)}, Errors: {err_count}")


if __name__ == "__main__":
    main()
