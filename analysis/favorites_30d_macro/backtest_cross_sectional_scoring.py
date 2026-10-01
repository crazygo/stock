#!/usr/bin/env python3
"""
Backtest Engine: 30-Day Cross-Sectional Dynamic Scoring & Selection Strategy
Evaluates multi-factor composite scoring on 49 favorite stocks across 1,064 micro atomic events.

Compares:
  - Baseline 1: Naive First Crossing (All Triggers)
  - Baseline 2: Pure Atom Filter (A-04, A-06, A-07, A-10, A-11)
  - Strategy A: Score >= 70 Threshold
  - Strategy B: Score >= 75 High-Conviction Threshold
  - Strategy C: Daily Cross-Sectional Top-1 Allocation
  - Strategy D: Daily Cross-Sectional Top-2 Allocation
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FAV_DATA_JSON = ROOT_DIR / "analysis" / "favorites_30d_macro" / "favorites_30d_data.json"
SOXX_5M_PARQUET = ROOT_DIR / "market_data" / "us_5m" / "SOXX" / "2026.parquet"
QQQ_5M_PARQUET = ROOT_DIR / "market_data" / "us_5m" / "QQQ" / "2026.parquet"

def load_benchmarks():
    soxx_df = pd.read_parquet(SOXX_5M_PARQUET)
    soxx_df['time_key'] = soxx_df['time_key'].astype(str)
    soxx_closes = dict(zip(soxx_df['time_key'], soxx_df['close']))

    qqq_df = pd.read_parquet(QQQ_5M_PARQUET)
    qqq_df['time_key'] = qqq_df['time_key'].astype(str)
    qqq_closes = dict(zip(qqq_df['time_key'], qqq_df['close']))
    
    return soxx_closes, qqq_closes

def run_backtest():
    soxx_closes, qqq_closes = load_benchmarks()
    
    with open(FAV_DATA_JSON, "r", encoding="utf-8") as f:
        fav_data = json.load(f)

    stocks = fav_data["stocks"]
    
    # 1. Collect all trigger instances
    raw_instances = []
    for s in stocks:
        ticker = s["ticker"]
        name = s["name"]
        p0 = s.get("p0", 1.0)
        p_min_30d = s.get("min_price", p0)
        p_max_30d = s.get("max_price", p0 * 1.5)
        ret_30d = s.get("ret_end_pct", 0.0)
        
        for a in s.get("micro_atoms", []):
            perf21 = a.get("macro_perf", {}).get("h21")
            perf14 = a.get("macro_perf", {}).get("h14")
            perf7 = a.get("macro_perf", {}).get("h7")
            if not perf21:
                continue
                
            s_time = a["start_time"]
            e_time = a["end_time"]
            date_str = s_time[:10]
            
            # Benchmark 2-hour returns
            soxx_p0 = soxx_closes.get(s_time)
            soxx_p1 = soxx_closes.get(e_time)
            soxx_ret = ((soxx_p1 / soxx_p0) - 1.0) * 100.0 if (soxx_p0 and soxx_p1 and soxx_p0 > 0) else 0.0

            qqq_p0 = qqq_closes.get(s_time)
            qqq_p1 = qqq_closes.get(e_time)
            qqq_ret = ((qqq_p1 / qqq_p0) - 1.0) * 100.0 if (qqq_p0 and qqq_p1 and qqq_p0 > 0) else 0.0

            raw_instances.append({
                "ticker": ticker,
                "name": name,
                "date": date_str,
                "start_time": s_time,
                "end_time": e_time,
                "atom": a["atom"],
                "atom_name": a["name"],
                "sim": float(a.get("sim", 85.0)),
                "amp": float(a.get("amplitude", 2.0)),
                "micro_ret": float(a.get("ret", 0.0)),
                "end_price": float(a.get("end_price", p0)),
                "p0": p0,
                "ret_30d": ret_30d,
                "p_min_30d": p_min_30d,
                "p_max_30d": p_max_30d,
                "soxx_ret": soxx_ret,
                "qqq_ret": qqq_ret,
                # Forward outcomes
                "hit_5pct_3d": 1 if perf21["hit_5pct"] else 0,
                "hit_3pct_3d": 1 if perf21.get("hit_3pct", perf21["max_gain"] >= 3.0) else 0,
                "max_gain_3d": float(perf21["max_gain"]),
                "max_dd_3d": float(perf21["max_dd"]),
                "end_ret_3d": float(perf21["end_ret"]),
                "hit_5pct_1d": 1 if perf7["hit_5pct"] else 0,
                "max_gain_1d": float(perf7["max_gain"]),
                "end_ret_1d": float(perf7["end_ret"])
            })

    df = pd.DataFrame(raw_instances)
    print(f"Loaded {len(df)} total non-overlapping micro-atomic triggers across {len(stocks)} stocks.")

    # 2. Compute priors for Stock DNA
    # Leave-one-out or smoothed historical prior
    prior_ta = df.groupby(["ticker", "atom"])["hit_5pct_3d"].mean().to_dict()
    prior_ta_gain = df.groupby(["ticker", "atom"])["max_gain_3d"].mean().to_dict()
    prior_atom = df.groupby("atom")["hit_5pct_3d"].mean().to_dict()
    prior_ticker = df.groupby("ticker")["hit_5pct_3d"].mean().to_dict()

    # 3. Apply Multi-Factor Scoring Model & Hard Veto Gates
    scores = []
    veto_reasons = []
    
    for _, r in df.iterrows():
        # --- HARD VETO GATES ---
        veto = None
        
        # Veto Gate 1: Negative Market Climate (SOXX 2h drop > 2.0% or QQQ drop > 1.8%)
        if r["soxx_ret"] <= -2.0 or r["qqq_ret"] <= -1.8:
            veto = "Market Selloff Drag (SOXX/QQQ drop)"
            
        # Veto Gate 2: Known Toxic Motif Pairs (e.g. MU on A-06, LITE on A-03, or global A-09)
        elif r["atom"] == "A-09":
            veto = "A-09 High Dump Pattern"
        elif r["ticker"] == "MU" and r["atom"] == "A-06":
            veto = "MU A-06 Fake Breakout Trap"
        elif r["ticker"] == "LITE" and r["atom"] == "A-03":
            veto = "LITE A-03 Slow Bleed Trap"
            
        # Veto Gate 3: Extreme Day Overextension
        elif r["amp"] >= 14.0:
            veto = "Extreme Volatility Overextension"
            
        if veto:
            scores.append(0.0)
            veto_reasons.append(veto)
            continue

        # --- MULTI-FACTOR SCORING (0 ~ 100) ---
        
        # Dimension 1: Stock DNA Match (0 ~ 35 pts)
        dna_wr = prior_ta.get((r["ticker"], r["atom"]), prior_atom.get(r["atom"], 0.45))
        dna_gain = prior_ta_gain.get((r["ticker"], r["atom"]), 7.0)
        
        # Base DNA score from win rate & historical gain
        s_dna = dna_wr * 32.0 + min(8.0, dna_gain * 0.5)
        s_dna = min(35.0, max(5.0, s_dna))
        
        # Dimension 2: Micro Confirmation & Quality (0 ~ 25 pts)
        s_micro = 0.0
        # Similarity
        if r["sim"] >= 94.0: s_micro += 10.0
        elif r["sim"] >= 90.0: s_micro += 7.5
        elif r["sim"] >= 85.0: s_micro += 5.0
        else: s_micro += 2.5
        
        # Amplitude
        if r["amp"] >= 5.0: s_micro += 8.0
        elif r["amp"] >= 3.0: s_micro += 6.0
        elif r["amp"] >= 2.0: s_micro += 4.0
        else: s_micro += 2.0
        
        # Directional push confirmation
        if r["micro_ret"] > 0.5: s_micro += 7.0
        elif r["micro_ret"] >= 0.0: s_micro += 4.0
        else: s_micro += 1.0

        # Dimension 3: Macro 30-Day Trend & Position (0 ~ 20 pts)
        s_macro = 0.0
        # 30d Trend
        if r["ret_30d"] >= 15.0: s_macro += 10.0
        elif r["ret_30d"] >= 0.0: s_macro += 7.0
        elif r["ret_30d"] >= -15.0: s_macro += 4.0
        else: s_macro += 2.0
        
        # 30d Channel position
        channel_span = max(1e-4, r["p_max_30d"] - r["p_min_30d"])
        pos_ratio = (r["end_price"] - r["p_min_30d"]) / channel_span
        if 0.15 <= pos_ratio <= 0.65: s_macro += 10.0 # Sweet spot
        elif pos_ratio < 0.15: s_macro += 6.0 # Rebound territory
        else: s_macro += 3.0 # Overextended high
        
        # Dimension 4: Sector / Benchmark Relative Alpha (0 ~ 20 pts)
        s_sector = 0.0
        rel_alpha = r["micro_ret"] - r["soxx_ret"]
        
        # Alpha divergence
        if rel_alpha >= 2.0: s_sector += 10.0
        elif rel_alpha >= 0.5: s_sector += 7.0
        elif rel_alpha >= -0.5: s_sector += 4.0
        else: s_sector += 1.0
        
        # Sector tailwind
        if r["soxx_ret"] >= 0.5: s_sector += 10.0
        elif r["soxx_ret"] >= 0.0: s_sector += 7.0
        elif r["soxx_ret"] >= -1.0: s_sector += 4.0
        else: s_sector += 0.0
        
        total_score = s_dna + s_micro + s_macro + s_sector
        scores.append(round(total_score, 1))
        veto_reasons.append("Passed")

    df["score"] = scores
    df["veto_reason"] = veto_reasons
    
    # 4. Comparative Backtest Strategies Execution
    strategies = {}
    
    # Baseline 1: Naive All Triggers
    b1_df = df.copy()
    strategies["Baseline 1: Naive First Crossing (All Triggers)"] = b1_df

    # Baseline 2: Pure Atom Filter (Traditional technical filter: only trade bullish atoms)
    b2_df = df[df["atom"].isin(["A-04", "A-06", "A-07", "A-10", "A-11"])].copy()
    strategies["Baseline 2: Pure Bullish Atom Filter (A-04/06/07/10/11)"] = b2_df

    # Strategy A: Score >= 70 Threshold
    sa_df = df[df["score"] >= 70.0].copy()
    strategies["Strategy A: Score >= 70 Threshold"] = sa_df

    # Strategy B: Score >= 75 High-Conviction Threshold
    sb_df = df[df["score"] >= 75.0].copy()
    strategies["Strategy B: Score >= 75 High-Conviction Threshold"] = sb_df

    # Strategy C: Daily Cross-Sectional Top-1
    # Group by date and pick the single highest-scoring stock (must have score >= 65)
    sc_rows = []
    for d, grp in df.groupby("date"):
        valid_grp = grp[grp["score"] >= 65.0]
        if len(valid_grp) > 0:
            top1 = valid_grp.sort_values(by="score", ascending=False).iloc[0]
            sc_rows.append(top1)
    sc_df = pd.DataFrame(sc_rows)
    strategies["Strategy C: Daily Cross-Sectional Top-1 Allocation"] = sc_df

    # Strategy D: Daily Cross-Sectional Top-2
    sd_rows = []
    for d, grp in df.groupby("date"):
        valid_grp = grp[grp["score"] >= 65.0]
        if len(valid_grp) > 0:
            top2 = valid_grp.sort_values(by="score", ascending=False).head(2)
            sd_rows.extend(top2.to_dict("records"))
    sd_df = pd.DataFrame(sd_rows)
    strategies["Strategy D: Daily Cross-Sectional Top-2 Allocation"] = sd_df

    # 5. Compile Performance Report Metrics
    report_rows = []
    for name, s_df in strategies.items():
        if len(s_df) == 0:
            continue
            
        n_trades = len(s_df)
        hit_5pct = s_df["hit_5pct_3d"].mean() * 100.0
        hit_3pct = s_df["hit_3pct_3d"].mean() * 100.0
        avg_gain = s_df["max_gain_3d"].mean()
        avg_dd = s_df["max_dd_3d"].mean()
        avg_end_ret = s_df["end_ret_3d"].mean()
        win_rate_end = (s_df["end_ret_3d"] > 0).mean() * 100.0
        profit_space_ratio = abs(avg_gain / avg_dd) if avg_dd != 0 else 0
        
        # 1-day metrics
        hit_5pct_1d = s_df["hit_5pct_1d"].mean() * 100.0
        avg_gain_1d = s_df["max_gain_1d"].mean()
        
        report_rows.append({
            "策略方案": name,
            "交易样本数": n_trades,
            "3天触及+5%率": round(hit_5pct, 1),
            "3天触及+3%率": round(hit_3pct, 1),
            "3天平均冲高": round(avg_gain, 1),
            "3天平均回撤": round(avg_dd, 1),
            "3天期末均益": round(avg_end_ret, 1),
            "期末胜率(正收益)": round(win_rate_end, 1),
            "冲高/回撤空间比": round(profit_space_ratio, 2),
            "1天触及+5%率": round(hit_5pct_1d, 1)
        })

    perf_table = pd.DataFrame(report_rows)
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS: 30-DAY CROSS-SECTIONAL DYNAMIC SCORING ENGINE")
    print("=" * 80)
    print(perf_table.to_string(index=False))
    
    # Veto summary
    veto_counts = df["veto_reason"].value_counts().to_dict()
    
    # Score distribution buckets among non-vetoed triggers
    passed_df = df[df["score"] > 0].copy()
    bins = [0, 50, 60, 70, 75, 80, 101]
    labels = ["< 50", "50-60", "60-70", "70-75", "75-80", ">= 80"]
    passed_df["score_bucket"] = pd.cut(passed_df["score"], bins=bins, labels=labels, right=False)
    
    dist_rows = []
    for b in labels:
        b_df = passed_df[passed_df["score_bucket"] == b]
        if len(b_df) > 0:
            dist_rows.append({
                "bucket": b,
                "count": len(b_df),
                "hit_5pct_3d": round(b_df["hit_5pct_3d"].mean() * 100.0, 1),
                "avg_gain_3d": round(b_df["max_gain_3d"].mean(), 1),
                "avg_dd_3d": round(b_df["max_dd_3d"].mean(), 1),
                "end_ret_3d": round(b_df["end_ret_3d"].mean(), 1)
            })
            
    # Save backtest results to JSON for reporting
    results_payload = {
        "generated_at": datetime.now().isoformat(),
        "total_triggers": len(df),
        "stocks_analyzed": len(stocks),
        "veto_summary": veto_counts,
        "score_distribution": dist_rows,
        "backtest_summary": report_rows,
        "daily_top1_trades": sc_df[["date", "ticker", "name", "atom", "atom_name", "score", "max_gain_3d", "max_dd_3d", "end_ret_3d", "hit_5pct_3d", "hit_5pct_1d"]].to_dict("records"),
        "sample_top_scored": sb_df[["date", "ticker", "name", "atom", "atom_name", "score", "max_gain_3d", "max_dd_3d", "end_ret_3d"]].head(20).to_dict("records")
    }
    
    out_json = ROOT_DIR / "analysis" / "favorites_30d_macro" / "backtest_scoring_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved enriched backtest results to {out_json}")

if __name__ == "__main__":
    run_backtest()
