#!/usr/bin/env python3
"""
build_hourly_dataset.py
Assembles unified regular hourly market data for QQQ constituent stocks
covering 2025-01-02 to 2026-09-28.
Sources:
  1. 2025 regular bars from market_data/model_training_history_v1/parts/
  2. 2026 regular bars from market_data/us_60m/
  3. 2026-09-28 regular bars from analysis/window_width_study/bars_20260928.json
Output:
  analysis/window_width_study/qqq_hourly_2025_2026.parquet
"""

import os
import sys
import glob
import json
import time
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONSTITUENTS_PATH = os.path.join(ROOT_DIR, "analysis/qqq_constituents.json")
HISTORY_PARTS_DIR = os.path.join(ROOT_DIR, "market_data/model_training_history_v1/parts")
US_60M_DIR = os.path.join(ROOT_DIR, "market_data/us_60m")
SEP28_JSON = os.path.join(os.path.dirname(__file__), "bars_20260928.json")
OUTPUT_PARQUET = os.path.join(os.path.dirname(__file__), "qqq_hourly_2025_2026.parquet")

REGULAR_HOURS = ['10:30:00', '11:30:00', '12:30:00', '13:30:00', '14:30:00', '15:30:00', '16:00:00']

def aggregate_5m_to_60m(df_5m):
    """Aggregate 5m regular bars to 7 regular hourly bars per day."""
    if df_5m.empty:
        return pd.DataFrame()
    
    time_str = df_5m['start_at_et'].str[11:19]
    conds = [
        (time_str >= '09:30:00') & (time_str < '10:30:00'),
        (time_str >= '10:30:00') & (time_str < '11:30:00'),
        (time_str >= '11:30:00') & (time_str < '12:30:00'),
        (time_str >= '12:30:00') & (time_str < '13:30:00'),
        (time_str >= '13:30:00') & (time_str < '14:30:00'),
        (time_str >= '14:30:00') & (time_str < '15:30:00'),
        (time_str >= '15:30:00') & (time_str <= '16:00:00'),
    ]
    choices = ['10:30:00', '11:30:00', '12:30:00', '13:30:00', '14:30:00', '15:30:00', '16:00:00']
    df_5m = df_5m.copy()
    df_5m['bucket'] = np.select(conds, choices, default=None)
    df_valid = df_5m[df_5m['bucket'].notna()].copy()
    df_valid['time_key'] = df_valid['session_date'] + ' ' + df_valid['bucket']
    
    agg = df_valid.groupby('time_key', as_index=False).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).sort_values('time_key').reset_index(drop=True)
    return agg

def main():
    t0 = time.time()
    print("=" * 70)
    print("BUILDING UNIFIED QQQ REGULAR HOURLY DATASET (2025-01 TO 2026-09-28)")
    print("=" * 70)

    with open(CONSTITUENTS_PATH, "r", encoding="utf-8") as f:
        tickers = json.load(f)["tickers"]
    print(f"Constituent tickers count: {len(tickers)}")

    sep28_data = {}
    if os.path.exists(SEP28_JSON):
        with open(SEP28_JSON, "r", encoding="utf-8") as f:
            sep28_data = json.load(f)
        print(f"Loaded 2026-09-28 data for {len(sep28_data)} tickers.")

    all_stock_dfs = []
    processed_count = 0

    for idx, ticker in enumerate(tickers):
        ticker_frames = []

        # 1. 2025 data from history parts
        history_2025_files = sorted(glob.glob(f"{HISTORY_PARTS_DIR}/{ticker}/2025-*/bars.parquet"))
        if history_2025_files:
            dfs_2025 = []
            for pf in history_2025_files:
                df = pd.read_parquet(pf, columns=['session_date', 'session_type', 'start_at_et', 'open', 'high', 'low', 'close', 'volume'])
                df = df[df['session_type'] == 'regular']
                dfs_2025.append(df)
            if dfs_2025:
                raw_5m = pd.concat(dfs_2025, ignore_index=True).sort_values('start_at_et').reset_index(drop=True)
                df_2025_h = aggregate_5m_to_60m(raw_5m)
                if not df_2025_h.empty:
                    ticker_frames.append(df_2025_h)

        # 2. 2026 data from us_60m
        p_2026 = os.path.join(US_60M_DIR, ticker, "2026.parquet")
        if os.path.exists(p_2026):
            df_2026 = pd.read_parquet(p_2026)
            df_2026['time'] = df_2026['time_key'].str.split(' ').str[1]
            df_2026_reg = df_2026[df_2026['time'].isin(REGULAR_HOURS)][['time_key', 'open', 'high', 'low', 'close', 'volume']]
            if not df_2026_reg.empty:
                ticker_frames.append(df_2026_reg)
        else:
            # Fallback to history parts 2026 if us_60m missing
            history_2026_files = sorted(glob.glob(f"{HISTORY_PARTS_DIR}/{ticker}/2026-*/bars.parquet"))
            if history_2026_files:
                dfs_2026 = []
                for pf in history_2026_files:
                    df = pd.read_parquet(pf, columns=['session_date', 'session_type', 'start_at_et', 'open', 'high', 'low', 'close', 'volume'])
                    df = df[df['session_type'] == 'regular']
                    dfs_2026.append(df)
                if dfs_2026:
                    raw_5m_2026 = pd.concat(dfs_2026, ignore_index=True).sort_values('start_at_et').reset_index(drop=True)
                    df_2026_h = aggregate_5m_to_60m(raw_5m_2026)
                    if not df_2026_h.empty:
                        ticker_frames.append(df_2026_h)

        # 3. Append 2026-09-28 if present
        if ticker in sep28_data and sep28_data[ticker]:
            df_sep28 = pd.DataFrame(sep28_data[ticker])
            if not df_sep28.empty:
                ticker_frames.append(df_sep28[['time_key', 'open', 'high', 'low', 'close', 'volume']])

        if ticker_frames:
            combined = pd.concat(ticker_frames, ignore_index=True)
            combined = combined.drop_duplicates(subset=['time_key']).sort_values('time_key').reset_index(drop=True)
            combined['ticker'] = ticker
            all_stock_dfs.append(combined)
            processed_count += 1

    print(f"Combined data for {processed_count} tickers.")
    final_df = pd.concat(all_stock_dfs, ignore_index=True)
    print(f"Total regular hourly bars: {len(final_df):,}")
    print(f"Min time_key: {final_df['time_key'].min()}, Max time_key: {final_df['time_key'].max()}")

    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    final_df.to_parquet(OUTPUT_PARQUET, index=False, compression="zstd")
    file_size_mb = os.path.getsize(OUTPUT_PARQUET) / (1024 * 1024)
    print(f"Saved to {OUTPUT_PARQUET} ({file_size_mb:.2f} MB in {time.time() - t0:.2f}s)")

if __name__ == "__main__":
    main()
