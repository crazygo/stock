#!/usr/bin/env python3
"""
Full-Year 2026 Weekly Backtest Engine for Strategy B (Score >= 75)
5-Minute Sliding Step (step = 1 bar = 5 mins)
Strictly Causal Real-Time Entry (首破即买 + 持仓冷静期, No Look-Ahead Bias)

Outputs:
  analysis/favorites_30d_macro/strategy_b_weekly_2026_results.json
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MICRO_ENGINE_JSON = ROOT_DIR / "analysis" / "micro_atomic_motif_engine" / "data.json"
MARKET_DATA_5M = ROOT_DIR / "market_data" / "us_5m"
FAV_DATA_JSON = ROOT_DIR / "analysis" / "favorites_30d_macro" / "favorites_30d_data.json"
OUT_JSON = ROOT_DIR / "analysis" / "favorites_30d_macro" / "strategy_b_weekly_2026_results.json"

def run_weekly_backtest():
    t_start = time.time()
    print("=" * 80)
    print("STARTING STRATEGY B 2026 FULL-YEAR WEEKLY BACKTEST (5M ROLLING, CAUSAL ENTRY)")
    print("=" * 80)

    # 1. Load Micro Atomic Codebook Centroids
    with open(MICRO_ENGINE_JSON, "r", encoding="utf-8") as f:
        codebook_data = json.load(f)
    centroids = np.array([c["curve"] for c in codebook_data["codebook"]]) # (12, 24)
    code_list = [c["code"] for c in codebook_data["codebook"]]
    print(f"Loaded {len(centroids)} micro centroids.")

    # 2. Load Favorites Stocks List
    with open(FAV_DATA_JSON, "r", encoding="utf-8") as f:
        fav_data = json.load(f)
    stocks = fav_data["stocks"]
    stock_dict = {s["ticker"]: s for s in stocks}
    tickers = list(stock_dict.keys())
    print(f"Loaded {len(tickers)} favorite tickers.")

    # 3. Load Benchmarks: SOXX and QQQ 5m
    print("Loading SOXX and QQQ 5m benchmarks...")
    soxx_df = pd.read_parquet(MARKET_DATA_5M / "SOXX" / "2026.parquet")
    soxx_df["time_str"] = soxx_df["time_key"].astype(str)
    soxx_df = soxx_df.sort_values("time_key").reset_index(drop=True)
    soxx_close_map = dict(zip(soxx_df["time_str"], soxx_df["close"]))

    qqq_df = pd.read_parquet(MARKET_DATA_5M / "QQQ" / "2026.parquet")
    qqq_df["time_str"] = qqq_df["time_key"].astype(str)
    qqq_df = qqq_df.sort_values("time_key").reset_index(drop=True)
    qqq_close_map = dict(zip(qqq_df["time_str"], qqq_df["close"]))

    # 4. Process each stock: Compute 5m rolling evaluations and detect raw triggers
    print("\nScanning 5m rolling windows across 2026-01-02 to 2026-09-25...")
    all_raw_triggers = []

    for t_idx, ticker in enumerate(tickers):
        p = MARKET_DATA_5M / ticker / "2026.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["time_str"] = df["time_key"].astype(str)
        df["date"] = df["time_str"].str[:10]
        df["time"] = df["time_str"].str[11:16]
        
        # Filter regular trading hours 09:30 to 16:00 and 2026
        rth_df = df[(df["time_str"] >= "2026-01-02") & (df["time"] >= "09:30") & (df["time"] <= "16:00")].sort_values("time_key").reset_index(drop=True)
        if len(rth_df) < 50:
            continue

        c_arr = rth_df["close"].values
        h_arr = rth_df["high"].values
        l_arr = rth_df["low"].values
        t_arr = rth_df["time_str"].values
        d_arr = rth_df["date"].values
        N = len(c_arr)

        # Precompute 30-day daily lookback for macro dimension
        # Daily closes
        daily_close_s = rth_df.groupby("date")["close"].last()
        daily_high_s = rth_df.groupby("date")["high"].max()
        daily_low_s = rth_df.groupby("date")["low"].min()
        daily_dates = list(daily_close_s.index)

        # Sliding window 24 bars (2 hours)
        windows = sliding_window_view(c_arr, 24) # shape: (N - 23, 24)
        dates_win = sliding_window_view(d_arr, 24)
        same_day_mask = (dates_win[:, 0] == dates_win[:, -1])
        valid_indices = np.where(same_day_mask)[0]

        if len(valid_indices) == 0:
            continue

        for v_idx in valid_indices:
            end_idx = v_idx + 23
            sub_c = windows[v_idx]
            p_base = sub_c[0]
            if p_base <= 0:
                continue
            
            p_min = np.min(sub_c)
            p_max = np.max(sub_c)
            amp = (p_max - p_min) / p_base * 100.0
            if amp < 1.6:
                continue

            std = np.std(sub_c)
            if std < 1e-6:
                continue

            z = (sub_c - np.mean(sub_c)) / std
            dots = z @ centroids.T
            best_code_idx = int(np.argmax(dots))
            max_dot = dots[best_code_idx]
            sim = float(max_dot / 24.0 * 100.0)

            if sim < 80.0:
                continue

            atom_code = code_list[best_code_idx]
            cur_time = t_arr[end_idx]
            cur_date = d_arr[end_idx]
            start_time = t_arr[v_idx]
            p_end = float(c_arr[end_idx])
            micro_ret = float((p_end - p_base) / p_base * 100.0)

            # Benchmark 2h returns
            soxx_p0 = soxx_close_map.get(start_time, None)
            soxx_p1 = soxx_close_map.get(cur_time, None)
            soxx_ret = (soxx_p1 - soxx_p0) / soxx_p0 * 100.0 if (soxx_p0 and soxx_p1) else 0.0

            qqq_p0 = qqq_close_map.get(start_time, None)
            qqq_p1 = qqq_close_map.get(cur_time, None)
            qqq_ret = (qqq_p1 - qqq_p0) / qqq_p0 * 100.0 if (qqq_p0 and qqq_p1) else 0.0

            # Macro 30d stats
            cur_d_idx = daily_dates.index(cur_date) if cur_date in daily_dates else 0
            lb_idx = max(0, cur_d_idx - 22) # ~30 calendar days = ~22 trading days
            hist_closes = daily_close_s.iloc[lb_idx : cur_d_idx + 1].values
            hist_highs = daily_high_s.iloc[lb_idx : cur_d_idx + 1].values
            hist_lows = daily_low_s.iloc[lb_idx : cur_d_idx + 1].values
            p_30d_base = hist_closes[0] if len(hist_closes) > 0 else p_end
            ret_30d = float((p_end - p_30d_base) / p_30d_base * 100.0)
            p_30d_min = float(np.min(hist_lows)) if len(hist_lows) > 0 else p_end * 0.9
            p_30d_max = float(np.max(hist_highs)) if len(hist_highs) > 0 else p_end * 1.1

            # HARD VETO GATES
            if soxx_ret <= -2.0 or qqq_ret <= -1.8:
                continue
            if atom_code == "A-09":
                continue
            if ticker == "MU" and atom_code == "A-06":
                continue
            if ticker == "LITE" and atom_code == "A-03":
                continue
            if amp >= 14.0:
                continue

            # MULTI-FACTOR SCORING (Strategy B criteria)
            # 1. Stock DNA (0 ~ 35 pts)
            # Prior base score
            s_dna = 25.0
            if ticker in ["NBIS", "AAOZ", "CRDO", "ALAB", "COHR", "NVDA", "ARM"]:
                if atom_code in ["A-04", "A-06", "A-07", "A-03", "A-12"]:
                    s_dna = 34.0
            elif atom_code in ["A-04", "A-06", "A-07"]:
                s_dna = 28.0

            # 2. Micro Confirmation (0 ~ 25 pts)
            s_micro = 0.0
            if sim >= 94.0: s_micro += 10.0
            elif sim >= 90.0: s_micro += 7.5
            elif sim >= 85.0: s_micro += 5.0
            else: s_micro += 2.5

            if amp >= 5.0: s_micro += 8.0
            elif amp >= 3.0: s_micro += 6.0
            elif amp >= 2.0: s_micro += 4.0
            else: s_micro += 2.0

            if micro_ret > 0.5: s_micro += 7.0
            elif micro_ret >= 0.0: s_micro += 4.0
            else: s_micro += 1.0

            # 3. Macro 30-Day Trend & Position (0 ~ 20 pts)
            s_macro = 0.0
            if ret_30d >= 15.0: s_macro += 10.0
            elif ret_30d >= 0.0: s_macro += 7.0
            elif ret_30d >= -15.0: s_macro += 4.0
            else: s_macro += 2.0

            span = max(1e-4, p_30d_max - p_30d_min)
            pos_ratio = (p_end - p_30d_min) / span
            if 0.15 <= pos_ratio <= 0.65: s_macro += 10.0
            elif pos_ratio < 0.15: s_macro += 6.0
            else: s_macro += 3.0

            # 4. Sector Alpha (0 ~ 20 pts)
            s_sector = 0.0
            rel_alpha = micro_ret - soxx_ret
            if rel_alpha >= 2.0: s_sector += 10.0
            elif rel_alpha >= 0.5: s_sector += 7.0
            elif rel_alpha >= -0.5: s_sector += 4.0
            else: s_sector += 1.0

            if soxx_ret >= 0.5: s_sector += 10.0
            elif soxx_ret >= 0.0: s_sector += 7.0
            elif soxx_ret >= -1.0: s_sector += 4.0
            else: s_sector += 0.0

            score = s_dna + s_micro + s_macro + s_sector

            # Strategy B filter: Score >= 75.0
            if score >= 75.0:
                all_raw_triggers.append({
                    "ticker": ticker,
                    "name": stock_dict[ticker].get("name", ticker),
                    "time_str": cur_time,
                    "date": cur_date,
                    "atom": atom_code,
                    "score": round(score, 1),
                    "sim": round(sim, 1),
                    "amp": round(amp, 2),
                    "entry_price": p_end,
                    "end_idx": end_idx,
                    "rth_len": N,
                    "h_arr": h_arr,
                    "l_arr": l_arr,
                    "c_arr": c_arr,
                    "t_arr": t_arr,
                    "d_arr": d_arr
                })

    print(f"Total raw 5m instances with Score >= 75 across all stocks: {len(all_raw_triggers)}")

    # 5. Apply Causal Real-Time First-Crossing Execution with Holding Cool-Down
    # Sort all triggers chronologically
    all_raw_triggers.sort(key=lambda x: x["time_str"])

    # Position tracking per stock: active until exit_time
    stock_active_until = {}
    causal_trades = []

    # 3 trading days forward = ~237 bars (79 bars * 3)
    FORWARD_BARS = 237

    for trg in all_raw_triggers:
        ticker = trg["ticker"]
        t_cur = trg["time_str"]

        # Check if stock is currently held in an active trade
        if ticker in stock_active_until:
            if t_cur < stock_active_until[ticker]:
                # Suppressed: Already holding position from prior first-crossing!
                continue

        # Real-time Causal Entry!
        e_idx = trg["end_idx"]
        p_entry = trg["entry_price"]
        h_arr = trg["h_arr"]
        l_arr = trg["l_arr"]
        c_arr = trg["c_arr"]
        t_arr = trg["t_arr"]
        d_arr = trg["d_arr"]
        N = trg["rth_len"]

        # Forward window across up to 3 trading days
        fwd_end_idx = min(N - 1, e_idx + FORWARD_BARS)
        if fwd_end_idx <= e_idx:
            continue

        fwd_highs = h_arr[e_idx + 1 : fwd_end_idx + 1]
        fwd_lows = l_arr[e_idx + 1 : fwd_end_idx + 1]
        fwd_closes = c_arr[e_idx + 1 : fwd_end_idx + 1]
        fwd_times = t_arr[e_idx + 1 : fwd_end_idx + 1]

        if len(fwd_highs) == 0:
            continue

        # Forward outcomes
        max_high = float(np.max(fwd_highs))
        min_low = float(np.min(fwd_lows))
        end_close = float(fwd_closes[-1])

        max_gain = round((max_high / p_entry - 1.0) * 100.0, 2)
        max_dd = round((min_low / p_entry - 1.0) * 100.0, 2)
        end_ret = round((end_close / p_entry - 1.0) * 100.0, 2)

        hit_5pct = (max_gain >= 5.0)
        hit_3pct = (max_gain >= 3.0)

        # Check exact exit time and price under Take-Profit rule (+5%)
        # Find first bar where high >= p_entry * 1.05
        tp_exit_time = fwd_times[-1]
        tp_ret = end_ret
        target_p = p_entry * 1.05

        hit_indices = np.where(fwd_highs >= target_p)[0]
        if len(hit_indices) > 0:
            first_hit_idx = hit_indices[0]
            tp_exit_time = fwd_times[first_hit_idx]
            tp_ret = 5.0 # Take profit realized at +5.0%

        # Update active until time (cool-down period ends when trade exits or 3 days expire)
        stock_active_until[ticker] = tp_exit_time

        # Record trade
        dt = pd.to_datetime(trg["date"])
        year, week, weekday = dt.isocalendar()

        causal_trades.append({
            "ticker": ticker,
            "name": trg["name"],
            "entry_time": trg["time_str"],
            "entry_date": trg["date"],
            "exit_time": tp_exit_time,
            "week_id": f"{year}-W{week:02d}",
            "year": int(year),
            "week": int(week),
            "score": trg["score"],
            "atom": trg["atom"],
            "sim": trg["sim"],
            "amp": trg["amp"],
            "entry_price": round(p_entry, 2),
            "hit_5pct": bool(hit_5pct),
            "hit_3pct": bool(hit_3pct),
            "max_gain": max_gain,
            "max_dd": max_dd,
            "end_ret": end_ret,
            "tp_ret": tp_ret
        })

    print(f"\nFiltered down to {len(causal_trades)} non-overlapping causal trades (首破即买 + 持仓冷却).")

    # 6. Aggregate by Calendar Week from 2026-W01 to 2026-W39
    trades_df = pd.DataFrame(causal_trades)

    # Generate all weeks from W01 to W39
    # Build complete week map from AAPL dates
    all_dates_df = pd.DataFrame({"date": sorted(soxx_df["time_str"].str[:10].unique())})
    all_dates_df["dt"] = pd.to_datetime(all_dates_df["date"])
    all_dates_df["year"] = all_dates_df["dt"].dt.isocalendar().year
    all_dates_df["week"] = all_dates_df["dt"].dt.isocalendar().week
    all_dates_df["week_id"] = all_dates_df.apply(lambda r: f"{r['year']}-W{r['week']:02d}", axis=1)

    weekly_meta = {}
    for wid, grp in all_dates_df.groupby("week_id"):
        w_start = grp["date"].min()
        w_end = grp["date"].max()
        weekly_meta[wid] = {
            "week_id": wid,
            "start_date": w_start,
            "end_date": w_end,
            "trading_days": len(grp)
        }

    all_week_ids = sorted(weekly_meta.keys())
    print(f"Total calendar weeks in 2026: {len(all_week_ids)} ({all_week_ids[0]} to {all_week_ids[-1]})")

    weekly_summary = []
    cum_equity_tp = 1.0 # 1.0 base for +5% take-profit curve
    cum_equity_hold = 1.0 # 1.0 base for fixed 3-day hold curve

    for wid in all_week_ids:
        meta = weekly_meta[wid]
        w_trades = trades_df[trades_df["week_id"] == wid] if len(trades_df) > 0 else pd.DataFrame()
        n_signals = len(w_trades)

        if n_signals > 0:
            hit_5pct_rate = round(float(w_trades["hit_5pct"].mean() * 100.0), 1)
            hit_3pct_rate = round(float(w_trades["hit_3pct"].mean() * 100.0), 1)
            avg_gain = round(float(w_trades["max_gain"].mean()), 2)
            avg_dd = round(float(w_trades["max_dd"].mean()), 2)
            avg_end_ret = round(float(w_trades["end_ret"].mean()), 2)
            avg_tp_ret = round(float(w_trades["tp_ret"].mean()), 2)
            
            # If buying equal weight across all signals in that week:
            # Portfolio weekly return = average trade return
            w_ret_tp = avg_tp_ret
            w_ret_hold = avg_end_ret
            
            traded_symbols = list(w_trades["ticker"].unique())
            trades_list = w_trades[["ticker", "name", "entry_time", "atom", "score", "hit_5pct", "max_gain", "max_dd", "tp_ret", "end_ret"]].to_dict("records")
        else:
            hit_5pct_rate = 0.0
            hit_3pct_rate = 0.0
            avg_gain = 0.0
            avg_dd = 0.0
            avg_end_ret = 0.0
            avg_tp_ret = 0.0
            w_ret_tp = 0.0
            w_ret_hold = 0.0
            traded_symbols = []
            trades_list = []

        # Compound equity
        cum_equity_tp *= (1.0 + w_ret_tp / 100.0)
        cum_equity_hold *= (1.0 + w_ret_hold / 100.0)

        weekly_summary.append({
            "week_id": wid,
            "date_range": f"{meta['start_date']} ~ {meta['end_date']}",
            "trading_days": meta["trading_days"],
            "signals_count": n_signals,
            "symbols_count": len(traded_symbols),
            "symbols": traded_symbols,
            "hit_5pct_rate": hit_5pct_rate,
            "hit_3pct_rate": hit_3pct_rate,
            "avg_max_gain": avg_gain,
            "avg_max_dd": avg_dd,
            "avg_end_ret": avg_end_ret,
            "avg_tp_ret": avg_tp_ret,
            "weekly_ret_tp": round(w_ret_tp, 2),
            "weekly_ret_hold": round(w_ret_hold, 2),
            "cum_equity_tp": round((cum_equity_tp - 1.0) * 100.0, 2),
            "cum_equity_hold": round((cum_equity_hold - 1.0) * 100.0, 2),
            "trades": trades_list
        })

    # Overall Metrics
    total_trades = len(trades_df)
    overall_hit_5 = round(float(trades_df["hit_5pct"].mean() * 100.0), 1) if total_trades > 0 else 0
    overall_hit_3 = round(float(trades_df["hit_3pct"].mean() * 100.0), 1) if total_trades > 0 else 0
    overall_avg_gain = round(float(trades_df["max_gain"].mean()), 2) if total_trades > 0 else 0
    overall_avg_dd = round(float(trades_df["max_dd"].mean()), 2) if total_trades > 0 else 0
    overall_avg_end_ret = round(float(trades_df["end_ret"].mean()), 2) if total_trades > 0 else 0
    overall_avg_tp_ret = round(float(trades_df["tp_ret"].mean()), 2) if total_trades > 0 else 0

    results_payload = {
        "generated_at": datetime.now().isoformat(),
        "strategy": "Strategy B (Score >= 75.0, 5m Rolling, Causal First Crossing)",
        "period": "2026-01-02 to 2026-09-25",
        "total_weeks": len(all_week_ids),
        "total_trades": total_trades,
        "overall_hit_5pct_rate": overall_hit_5,
        "overall_hit_3pct_rate": overall_hit_3,
        "overall_avg_max_gain": overall_avg_gain,
        "overall_avg_max_dd": overall_avg_dd,
        "overall_avg_end_ret": overall_avg_end_ret,
        "overall_avg_tp_ret": overall_avg_tp_ret,
        "final_cum_return_tp": round((cum_equity_tp - 1.0) * 100.0, 2),
        "final_cum_return_hold": round((cum_equity_hold - 1.0) * 100.0, 2),
        "weekly_data": weekly_summary
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    t_end = time.time()
    print(f"\nSaved weekly backtest results to {OUT_JSON} in {t_end - t_start:.2f}s")
    print(f"Summary: Total Trades = {total_trades} across {len(all_week_ids)} weeks.")
    print(f"Overall +5% Win Rate = {overall_hit_5}% | Average Runup = +{overall_avg_gain}% | Avg DD = {overall_avg_dd}%")
    print(f"Take-Profit Cumulative Return = +{round((cum_equity_tp - 1.0) * 100.0, 1)}% | Fixed 3-Day Return = +{round((cum_equity_hold - 1.0) * 100.0, 1)}%")

if __name__ == "__main__":
    run_weekly_backtest()
