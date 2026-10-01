#!/usr/bin/env python3
"""Feature extraction across opening checkpoints: 5m, 10m, 15m, 20m, 25m, 30m.

Supports Multi-Day Holding Horizons:
- 1d: Intraday (Day D checkpoint T to Day D 16:00 close)
- 3d: 3 Trading Days (Day D checkpoint T to Day D+2 16:00 close)
- 5d: 5 Trading Days (Day D checkpoint T to Day D+4 16:00 close)
- 10d: 10 Trading Days (Day D checkpoint T to Day D+9 16:00 close)

For QQQ constituents in 2026:
Extracts 6 orthogonal micro-feature dimensions at each decision point T in {5, 10, 15, 20, 25, 30} minutes after 09:30 open.
Calculates forward path-dependent targets (MFE, MAE, CloseRet, Y_hit_5pct, Touch_5pct) for each horizon.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
QQQ_CONSTITUENTS_FILE = ROOT / "analysis" / "qqq_constituents.json"
MARKET_DATA_5M_DIR = ROOT / "market_data" / "us_5m"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = OUTPUT_DIR / "checkpoints_dataset.parquet"

CHECKPOINTS = [
    (5, "09:35:00"),
    (10, "09:40:00"),
    (15, "09:45:00"),
    (20, "09:50:00"),
    (25, "09:55:00"),
    (30, "10:00:00"),
]

HORIZONS = [1, 3, 5, 10]
HORIZON_CONFIG = {
    1: {"target_pct": 0.05, "stop_thresh": -0.03, "label_name": "1d_5pct"},
    3: {"target_pct": 0.05, "stop_thresh": -0.05, "label_name": "3d_5pct"},
    5: {"target_pct": 0.10, "stop_thresh": -0.06, "label_name": "5d_10pct"},
    10: {"target_pct": 0.10, "stop_thresh": -0.08, "label_name": "10d_10pct"},
}


def load_qqq_benchmark() -> pd.DataFrame:
    """Pre-load QQQ 5m data to compute market benchmark features."""
    qqq_path = MARKET_DATA_5M_DIR / "QQQ" / "2026.parquet"
    if not qqq_path.exists():
        raise FileNotFoundError(f"QQQ benchmark parquet missing at {qqq_path}")
    df_qqq = pd.read_parquet(qqq_path)
    df_qqq["session_date"] = df_qqq["session_date"].fillna(df_qqq["time_key"].astype(str).str.slice(0, 10))
    return df_qqq


def build_qqq_lookup(df_qqq: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, float]]:
    """Build fast lookup table for QQQ metrics at each date and checkpoint."""
    lookup = {}
    reg = df_qqq[df_qqq["session_type"] == "regular"].copy()
    
    for date, group in reg.groupby("session_date"):
        group = group.sort_values("time_key")
        if len(group) == 0:
            continue
        
        last_close = group["last_close"].iloc[0]
        open_0930 = group["open"].iloc[0]
        qqq_gap = (open_0930 - last_close) / last_close if last_close > 0 else 0.0
        
        for t_min, t_str in CHECKPOINTS:
            sub = group[group["time_key"].str.endswith(t_str)]
            if len(sub) == 0:
                continue
            bar_t = sub.iloc[0]
            p_t = bar_t["close"]
            qqq_return = (p_t - open_0930) / open_0930 if open_0930 > 0 else 0.0
            
            bars_up_to_t = group[group["time_key"] <= bar_t["time_key"]]
            tot_vol = bars_up_to_t["volume"].sum()
            tot_to = bars_up_to_t["turnover"].sum()
            vwap = (tot_to / tot_vol) if tot_vol > 0 else p_t
            vwap_dev = (p_t - vwap) / vwap if vwap > 0 else 0.0
            
            lookup[(date, t_min)] = {
                "qqq_gap": float(qqq_gap),
                "qqq_intra_return": float(qqq_return),
                "qqq_vwap_dev": float(vwap_dev),
            }
    return lookup


def compute_stock_history_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute 20-day rolling ATR and 20-day rolling premarket volume per date."""
    dates = sorted([d for d in df["session_date"].unique() if pd.notnull(d)])
    
    records = []
    for d in dates:
        day_df = df[df["session_date"] == d]
        reg = day_df[day_df["session_type"] == "regular"]
        pm = day_df[day_df["session_type"] == "pre_market"]
        
        if len(reg) == 0:
            continue
            
        high_reg = reg["high"].max()
        low_reg = reg["low"].min()
        last_c = reg["last_close"].iloc[0]
        close_reg = reg["close"].iloc[-1]
        
        tr = max(high_reg - low_reg, abs(high_reg - last_c), abs(low_reg - last_c)) if last_c > 0 else (high_reg - low_reg)
        pm_vol = pm["volume"].sum() if len(pm) > 0 else 0.0
        
        cp_vols = {}
        for t_min, t_str in CHECKPOINTS:
            bars = reg[reg["time_key"].astype(str).str.slice(11, 19) <= t_str]
            cp_vols[f"vol_{t_min}m"] = bars["volume"].sum() if len(bars) > 0 else 0.0
            
        records.append({
            "session_date": d,
            "tr": tr,
            "close": close_reg,
            "pm_vol": pm_vol,
            **cp_vols
        })
        
    stats_df = pd.DataFrame(records)
    if len(stats_df) == 0:
        return {}
        
    stats_df = stats_df.sort_values("session_date").reset_index(drop=True)
    
    stats_df["atr_20d"] = stats_df["tr"].rolling(20, min_periods=5).mean().shift(1)
    stats_df["atr_20d_pct"] = (stats_df["atr_20d"] / stats_df["close"].shift(1)).fillna(0.02)
    stats_df["pm_vol_20d_med"] = stats_df["pm_vol"].rolling(20, min_periods=5).median().shift(1).fillna(10000.0)
    
    for t_min, _ in CHECKPOINTS:
        col = f"vol_{t_min}m"
        stats_df[f"{col}_20d_med"] = stats_df[col].rolling(20, min_periods=5).median().shift(1).fillna(50000.0)
        
    lookup = {}
    for _, row in stats_df.iterrows():
        d = row["session_date"]
        d_dict = {
            "atr_20d_pct": float(row["atr_20d_pct"]),
            "pm_vol_20d_med": float(row["pm_vol_20d_med"]),
        }
        for t_min, _ in CHECKPOINTS:
            d_dict[f"vol_{t_min}m_med"] = float(row[f"vol_{t_min}m_20d_med"])
        lookup[d] = d_dict
        
    return lookup


def extract_features_for_stock(
    symbol: str,
    df: pd.DataFrame,
    qqq_lookup: Dict[Tuple[str, int], Dict[str, float]],
    min_pm_intra_gain: float = 0.02,
    min_pm_gap: float = 0.015,
) -> List[Dict]:
    """Extract features at 6 checkpoints and forward outcomes for 1d, 3d, 5d, 10d horizons."""
    rows = []
    
    # Robust session_date fillna
    df["session_date"] = df["session_date"].fillna(df["time_key"].astype(str).str.slice(0, 10))
    df = df.dropna(subset=["time_key"])
    
    hist_stats = compute_stock_history_stats(df)
    
    pm_all = df[df["session_type"] == "pre_market"]
    reg_all = df[df["session_type"] == "regular"]
    
    # Index regular bars by date for fast slicing
    date_reg_groups = {d: g.sort_values("time_key") for d, g in reg_all.groupby("session_date")}
    date_pm_groups = {d: g.sort_values("time_key") for d, g in pm_all.groupby("session_date")}
    
    all_dates = sorted([d for d in date_reg_groups.keys() if len(date_reg_groups[d]) >= 8])
    
    for i, date in enumerate(all_dates):
        day_reg = date_reg_groups[date]
        day_pm = date_pm_groups.get(date, pd.DataFrame())
        
        last_close = day_reg["last_close"].iloc[0]
        open_0930 = day_reg["open"].iloc[0]
        if last_close <= 0 or open_0930 <= 0:
            continue
            
        # Premarket metrics
        if len(day_pm) >= 2:
            pm_high = day_pm["high"].max()
            pm_low = day_pm["low"].min()
            pm_close = day_pm["close"].iloc[-1]
            pm_vol = day_pm["volume"].sum()
            pm_turnover = day_pm["turnover"].sum()
            pm_vwap = (pm_turnover / pm_vol) if pm_vol > 0 else pm_close
            
            peak_row = day_pm.loc[day_pm["high"].idxmax()]
            peak_time_str = str(peak_row["time_key"])[11:16]
            peak_h, peak_m = map(int, peak_time_str.split(":"))
            peak_minutes_to_open = (9 * 60 + 30) - (peak_h * 60 + peak_m)
            
            pm_final_sub = day_pm[day_pm["time_key"].astype(str).str.slice(11, 19) >= "09:20:00"]
            if len(pm_final_sub) >= 2:
                pm_final_vel = (pm_final_sub["close"].iloc[-1] - pm_final_sub["open"].iloc[0]) / pm_final_sub["open"].iloc[0]
            else:
                pm_final_vel = 0.0
                
            pm_intra_gain = (pm_high - pm_low) / pm_low if pm_low > 0 else 0.0
            pm_gap = (open_0930 - last_close) / last_close
            pm_gain = (pm_close - last_close) / last_close
            pm_surge_retention = (pm_close - pm_low) / (pm_high - pm_low) if pm_high > pm_low else 1.0
            pm_vwap_dev = (pm_close - pm_vwap) / pm_vwap if pm_vwap > 0 else 0.0
        else:
            pm_high = open_0930
            pm_low = open_0930
            pm_close = open_0930
            pm_vol = 0.0
            pm_turnover = 0.0
            pm_vwap = open_0930
            peak_minutes_to_open = 0
            pm_final_vel = 0.0
            pm_intra_gain = 0.0
            pm_gap = (open_0930 - last_close) / last_close
            pm_gain = pm_gap
            pm_surge_retention = 0.0
            pm_vwap_dev = 0.0
            
        # Screener Trigger: active premarket OR opening 5m momentum
        first_bar_ret = (day_reg["close"].iloc[0] - open_0930) / open_0930
        if not (pm_intra_gain >= min_pm_intra_gain or pm_gap >= min_pm_gap or first_bar_ret >= 0.015):
            continue
            
        h_stat = hist_stats.get(date, {"atr_20d_pct": 0.02, "pm_vol_20d_med": 10000.0})
        atr_pct = h_stat["atr_20d_pct"]
        pm_rvol = pm_vol / max(1.0, h_stat["pm_vol_20d_med"])
        
        # Iterate over 6 checkpoints
        for t_min, t_str in CHECKPOINTS:
            sub_cp = day_reg[day_reg["time_key"].astype(str).str.endswith(t_str)]
            if len(sub_cp) == 0:
                continue
                
            bar_t = sub_cp.iloc[0]
            p_t = bar_t["close"]
            if p_t <= 0:
                continue
                
            bars_up_to_t = day_reg[day_reg["time_key"] <= bar_t["time_key"]].copy()
            first_day_remaining = day_reg[day_reg["time_key"] > bar_t["time_key"]].copy()
            if len(first_day_remaining) == 0:
                continue
                
            # Intraday opening metrics in [09:30, T]
            intra_high = bars_up_to_t["high"].max()
            intra_low = bars_up_to_t["low"].min()
            intra_vol = bars_up_to_t["volume"].sum()
            intra_turnover = bars_up_to_t["turnover"].sum()
            intra_vwap = (intra_turnover / intra_vol) if intra_vol > 0 else p_t
            
            intra_return_from_open = (p_t - open_0930) / open_0930
            intra_high_from_open = (intra_high - open_0930) / open_0930
            intra_low_from_open = (intra_low - open_0930) / open_0930
            intra_range_pct = (intra_high - intra_low) / open_0930
            intra_range_retention = (p_t - intra_low) / (intra_high - intra_low) if intra_high > intra_low else 1.0
            intra_vwap_dev = (p_t - intra_vwap) / intra_vwap if intra_vwap > 0 else 0.0
            
            n_bars = len(bars_up_to_t)
            green_bars = (bars_up_to_t["close"] > bars_up_to_t["open"]).sum()
            intra_upbar_ratio = green_bars / n_bars if n_bars > 0 else 0.5
            
            curr_bar_ret = (bar_t["close"] - bar_t["open"]) / bar_t["open"]
            curr_bar_range = (bar_t["high"] - bar_t["low"]) / bar_t["open"]
            
            vol_accel = (bar_t["volume"] / max(1.0, bars_up_to_t["volume"].iloc[:-1].mean())) if n_bars > 1 else 1.0
            cp_vol_med = h_stat.get(f"vol_{t_min}m_med", 50000.0)
            intra_rvol = intra_vol / max(1.0, cp_vol_med)
            
            intra_breakout_pm_high = int(p_t >= pm_high)
            intra_dist_to_pm_high = (p_t - pm_high) / pm_high if pm_high > 0 else 0.0
            
            qqq_info = qqq_lookup.get((date, t_min), {"qqq_gap": 0.0, "qqq_intra_return": 0.0, "qqq_vwap_dev": 0.0})
            qqq_gap = qqq_info["qqq_gap"]
            qqq_intra_ret = qqq_info["qqq_intra_return"]
            alpha_vs_qqq = intra_return_from_open - qqq_intra_ret
            qqq_vwap_dev = qqq_info["qqq_vwap_dev"]
            
            # --- Multi-Day Forward Outcomes Extraction ---
            forward_dict = {}
            for h in HORIZONS:
                if h == 1:
                    fw_bars = first_day_remaining
                    is_matured = len(fw_bars) > 0
                else:
                    available_days_ahead = len(all_dates) - 1 - i
                    is_matured = available_days_ahead >= (h - 1)
                    days_to_take = min(h - 1, available_days_ahead)
                    subsequent_days = [date_reg_groups[all_dates[i + k]] for k in range(1, days_to_take + 1)] if days_to_take > 0 else []
                    fw_bars = pd.concat([first_day_remaining] + subsequent_days, ignore_index=True)
                
                cfg = HORIZON_CONFIG[h]
                target_pct = cfg["target_pct"]
                stop_thresh = cfg["stop_thresh"]

                forward_dict[f"is_matured_{h}d"] = int(is_matured)
                forward_dict[f"target_pct_{h}d"] = target_pct
                
                if len(fw_bars) == 0:
                    forward_dict[f"forward_mfe_{h}d"] = np.nan
                    forward_dict[f"forward_mae_{h}d"] = np.nan
                    forward_dict[f"forward_close_ret_{h}d"] = np.nan
                    forward_dict[f"touch_{h}d"] = np.nan
                    forward_dict[f"y_hit_{h}d"] = np.nan
                    forward_dict[f"touch_5pct_{h}d"] = np.nan
                    forward_dict[f"touch_10pct_{h}d"] = np.nan
                    forward_dict[f"y_hit_5pct_{h}d"] = np.nan
                else:
                    h_max = fw_bars["high"].max()
                    p_idx = fw_bars["high"].idxmax()
                    l_before = fw_bars.loc[:p_idx, "low"].min()
                    c_final = fw_bars["close"].iloc[-1]
                    
                    mfe = (h_max - p_t) / p_t
                    mae = (l_before - p_t) / p_t
                    close_ret = (c_final - p_t) / p_t
                    
                    touch_target = int(mfe >= target_pct)
                    
                    forward_dict[f"forward_mfe_{h}d"] = float(mfe)
                    forward_dict[f"forward_mae_{h}d"] = float(mae)
                    forward_dict[f"forward_close_ret_{h}d"] = float(close_ret)
                    forward_dict[f"touch_{h}d"] = int(touch_target)
                    forward_dict[f"touch_5pct_{h}d"] = int(mfe >= 0.05)
                    forward_dict[f"touch_10pct_{h}d"] = int(mfe >= 0.10)
                    
                    if is_matured:
                        y_target = int(mfe >= target_pct and mae > stop_thresh)
                        forward_dict[f"y_hit_{h}d"] = int(y_target)
                        forward_dict[f"y_hit_5pct_{h}d"] = int(mfe >= 0.05 and mae > -0.05)
                    else:
                        if mfe >= target_pct and mae > stop_thresh:
                            forward_dict[f"y_hit_{h}d"] = 1
                        else:
                            forward_dict[f"y_hit_{h}d"] = np.nan
                            
                        if mfe >= 0.05 and mae > -0.05:
                            forward_dict[f"y_hit_5pct_{h}d"] = 1
                        else:
                            forward_dict[f"y_hit_5pct_{h}d"] = np.nan
            
            # Base record
            rec = {
                # Metadata
                "symbol": symbol,
                "session_date": date,
                "checkpoint_min": t_min,
                "checkpoint_time": t_str,
                "entry_price": float(p_t),
                "open_0930": float(open_0930),
                "last_close": float(last_close),
                
                # Dim 1: Pre-Market Geometry
                "pm_gap": float(pm_gap),
                "pm_intra_gain": float(pm_intra_gain),
                "pm_gain": float(pm_gain),
                "pm_surge_retention": float(pm_surge_retention),
                "pm_peak_to_open_mins": float(peak_minutes_to_open),
                "pm_final_velocity_10m": float(pm_final_vel),
                
                # Dim 2: Pre-Market Liquidity
                "pm_volume": float(pm_vol),
                "pm_turnover": float(pm_turnover),
                "pm_vwap_dev": float(pm_vwap_dev),
                "pm_rvol": float(pm_rvol),
                
                # Dim 3: Intraday Opening Momentum
                "intra_return_from_open": float(intra_return_from_open),
                "intra_high_from_open": float(intra_high_from_open),
                "intra_low_from_open": float(intra_low_from_open),
                "intra_range_pct": float(intra_range_pct),
                "intra_range_retention": float(intra_range_retention),
                "intra_vwap_dev": float(intra_vwap_dev),
                "intra_upbar_ratio": float(intra_upbar_ratio),
                "intra_curr_bar_ret": float(curr_bar_ret),
                "intra_curr_bar_range": float(curr_bar_range),
                "intra_breakout_pm_high": int(intra_breakout_pm_high),
                "intra_dist_to_pm_high": float(intra_dist_to_pm_high),
                
                # Dim 4: Order Flow & Volume
                "intra_volume": float(intra_vol),
                "intra_turnover": float(intra_turnover),
                "intra_rvol": float(intra_rvol),
                "intra_vol_acceleration": float(vol_accel),
                
                # Dim 5: Market Benchmark & Alpha
                "qqq_gap": float(qqq_gap),
                "qqq_intra_ret": float(qqq_intra_ret),
                "alpha_vs_qqq": float(alpha_vs_qqq),
                "qqq_vwap_dev": float(qqq_vwap_dev),
                
                # Dim 6: Stock Volatility Profile
                "atr_20d_pct": float(atr_pct),
                
                # Forward Multi-Horizon Outcomes
                **forward_dict
            }
            
            # Legacy aliases for backward compatibility with 1d
            rec["forward_mfe"] = rec.get("forward_mfe_1d", np.nan)
            rec["forward_mae"] = rec.get("forward_mae_1d", np.nan)
            rec["forward_close_ret"] = rec.get("forward_close_ret_1d", np.nan)
            rec["y_hit_5pct"] = rec.get("y_hit_1d", np.nan)
            
            rows.append(rec)
            
    return rows


def main():
    print("=== Starting Feature Extraction Across Checkpoints & Multi-Day Horizons (1d, 3d, 5d, 10d) ===")
    t0 = time.time()
    
    with open(QQQ_CONSTITUENTS_FILE) as f:
        data = json.load(f)
        tickers = data["tickers"]
    print(f"Loaded {len(tickers)} QQQ constituent tickers.")
    
    print("Loading QQQ benchmark index...")
    df_qqq = load_qqq_benchmark()
    qqq_lookup = build_qqq_lookup(df_qqq)
    print(f"QQQ lookup built: {len(qqq_lookup)} (date, checkpoint) points.")
    
    all_rows = []
    processed_count = 0
    
    for i, symbol in enumerate(tickers, 1):
        parquet_path = MARKET_DATA_5M_DIR / symbol / "2026.parquet"
        if not parquet_path.exists():
            continue
            
        try:
            df_sym = pd.read_parquet(parquet_path)
            rows = extract_features_for_stock(symbol, df_sym, qqq_lookup)
            all_rows.extend(rows)
            processed_count += 1
            if i % 15 == 0 or i == len(tickers):
                print(f"[{i}/{len(tickers)}] {symbol:6s} -> Extracted {len(rows)} samples. Total: {len(all_rows)}")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            
    print(f"\nCompleted extraction for {processed_count} tickers in {time.time() - t0:.2f}s.")
    print(f"Total extracted checkpoint samples: {len(all_rows)}")
    
    df_result = pd.DataFrame(all_rows)
    df_result.to_parquet(OUTPUT_FILE, index=False, compression="zstd")
    print(f"Saved dataset to {OUTPUT_FILE} (size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB)")
    
    # Print breakdown per checkpoint and horizon
    print("\n=== Multi-Horizon Base Rates at Checkpoint T=10m (09:40 ET) (1d 5%, 3d 5%, 5d 10%, 10d 10%) ===")
    sub_10 = df_result[df_result["checkpoint_min"] == 10]
    for h in HORIZONS:
        valid_rows = sub_10.dropna(subset=[f"touch_{h}d"])
        tot = len(valid_rows)
        tgt = int(HORIZON_CONFIG[h]["target_pct"] * 100)
        t_cnt = valid_rows[f"touch_{h}d"].sum()
        v_cnt = valid_rows[f"y_hit_{h}d"].sum()
        mfe_avg = valid_rows[f"forward_mfe_{h}d"].mean()
        print(f"Horizon {h:2d} Days | Target: >=+{tgt:2d}% | Samples: {tot} | Touch: {t_cnt/tot*100:5.1f}% | Valid: {v_cnt/tot*100:5.1f}% | Avg MFE: {mfe_avg*100:+5.2f}%")


if __name__ == "__main__":
    main()
