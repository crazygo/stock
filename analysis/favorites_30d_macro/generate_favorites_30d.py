#!/usr/bin/env python3
"""
Fetch 30-day 60m Macro K-lines & Snapshots from Futu OpenD for user's Favorites (自选股),
compute Log-scale return series, high/low drawdown, today's performance, and holding tags.

Outputs:
  analysis/favorites_30d_macro/favorites_30d_data.json
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import futu as ft
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = ROOT_DIR / "analysis" / "favorites_30d_macro"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "favorites_30d_data.json"

TARGET_CASH_ACC_ID = 283445330641239202 # Moomoo US 现金账户 (尾号 0086)

def classify_trend(ret_end, ret_min, ret_max, p_series):
    """Classify 30-day macro trend shape."""
    if len(p_series) < 10:
        return "数据不足 ⏳", "#888888"
    
    # Check V-shape reversal
    if ret_min <= -12.0 and (ret_end - ret_min) >= 12.0:
        return "深V反弹 ⚡", "#10b981"
    # Check inverted V (pump and dump)
    if ret_max >= 15.0 and (ret_max - ret_end) >= 15.0:
        return "冲高回落 ⚠️", "#ef4444"
    if ret_end >= 25.0:
        return "强势主升 🚀", "#059669"
    if ret_end >= 8.0:
        return "温和走多 ↗️", "#2563eb"
    if ret_end <= -25.0:
        return "破位阴跌 📉", "#dc2626"
    if ret_end <= -8.0:
        return "弱势回调 ↘️", "#b91c1c"
    return "箱体震荡 ⏸️", "#6b7280"

def main():
    t0 = time.time()
    print("=" * 70)
    print("FETCHING 30-DAY MACRO TRENDS FOR USER FAVORITES FROM FUTU OPEND")
    print("=" * 70)

    ft.SysConfig.enable_proto_encrypt(False)
    
    quote_ctx = None
    trd_ctx = None
    
    try:
        print("Connecting to Futu OpenD (127.0.0.1:11111)...")
        quote_ctx = ft.OpenQuoteContext(host="127.0.0.1", port=11111)
    except Exception as e:
        print(f"ERROR: Cannot connect to Futu OpenD: {e}")
        return

    # 1. Fetch User Favorites
    print("\n1. Fetching Favorites (自选股)...")
    ret_fav, fav_df = quote_ctx.get_user_security(group_name="Favorites")
    if ret_fav != ft.RET_OK or len(fav_df) == 0:
        print(f"Failed to fetch Favorites: {fav_df}")
        quote_ctx.close()
        return

    fav_codes = list(fav_df["code"])
    code_to_name = dict(zip(fav_df["code"], fav_df["name"]))
    print(f"Loaded {len(fav_codes)} items from Favorites: {fav_codes}")

    # Also check other user groups (📉惨了)
    extra_groups = {}
    try:
        ret_bad, bad_df = quote_ctx.get_user_security(group_name="📉惨了")
        if ret_bad == ft.RET_OK and len(bad_df) > 0:
            extra_groups["📉惨了"] = list(bad_df["code"])
            for _, r in bad_df.iterrows():
                if r["code"] not in code_to_name:
                    code_to_name[r["code"]] = r["name"]
    except Exception as e:
        print(f"Note: Could not check 📉惨了 group: {e}")

    # 2. Query 0086 Cash Account holdings to tag positions
    print("\n2. Checking 0086 Cash Account positions...")
    positions = {}
    try:
        trd_ctx = ft.OpenSecTradeContext(
            filter_trdmarket=ft.TrdMarket.US,
            host="127.0.0.1",
            port=11111,
            security_firm=ft.SecurityFirm.FUTUINC
        )
        r_pos, pos_df = trd_ctx.position_list_query(acc_id=TARGET_CASH_ACC_ID)
        if r_pos == ft.RET_OK and len(pos_df) > 0:
            for _, row in pos_df.iterrows():
                c = row["code"]
                qty = float(row.get("qty", 0))
                if qty > 0:
                    positions[c] = {
                        "qty": qty,
                        "cost_price": float(row.get("cost_price", 0)),
                        "nominal_price": float(row.get("nominal_price", 0)),
                        "pl_ratio": float(row.get("pl_ratio", 0)) if pd.notna(row.get("pl_ratio")) else 0.0,
                        "pl_val": float(row.get("pl_val", 0)) if pd.notna(row.get("pl_val")) else 0.0
                    }
        print(f"Loaded {len(positions)} active holdings in 0086 cash account.")
    except Exception as e:
        print(f"Warning: Failed to query trade positions: {e}")
    finally:
        if trd_ctx:
            trd_ctx.close()

    # Total target codes to query: union of Favorites and 0086 holdings
    all_codes = list(fav_codes)
    for c in positions.keys():
        if c not in all_codes:
            all_codes.append(c)
    
    print(f"\nTotal unique stocks to process: {len(all_codes)} ({len(fav_codes)} in Favorites, {len(positions)} held)")

    # 3. Batch Market Snapshots
    print("\n3. Fetching Market Snapshots in batch...")
    ret_snap, snap_df = quote_ctx.get_market_snapshot(all_codes)
    snapshots = {}
    if ret_snap == ft.RET_OK and len(snap_df) > 0:
        for _, row in snap_df.iterrows():
            snapshots[row["code"]] = row
        print(f"Successfully retrieved snapshots for {len(snapshots)} stocks.")
    else:
        print(f"Warning: Snapshot query returned {ret_snap}")

    # 4. Fetch 60m K-lines for each stock (30 trading days: 2026-08-15 to 2026-09-29)
    print("\n4. Fetching 30-day 60m K-lines...")
    stocks_data = []

    for i, code in enumerate(all_codes, 1):
        market = "US" if code.startswith("US.") else ("HK" if code.startswith("HK.") else "OTHER")
        ticker = code.split(".")[1]
        name = code_to_name.get(code, ticker)
        snap = snapshots.get(code)
        
        cur_price = None
        last_close = None
        today_chg = 0.0
        turnover = 0.0
        pe_ratio = None
        turnover_rate = 0.0

        if snap is not None:
            cur_price = float(snap.get("last_price", 0))
            last_close = float(snap.get("prev_close_price", 0))
            if last_close > 0 and cur_price > 0:
                today_chg = round((cur_price - last_close) / last_close * 100.0, 2)
            turnover = float(snap.get("turnover", 0)) if pd.notna(snap.get("turnover")) else 0.0
            pe_ratio = float(snap.get("pe_ratio", 0)) if pd.notna(snap.get("pe_ratio")) else None
            turnover_rate = float(snap.get("turnover_rate", 0)) if pd.notna(snap.get("turnover_rate")) else 0.0
            if not name or name == ticker:
                name = str(snap.get("name", ticker))

        ret_kl, df, _ = quote_ctx.request_history_kline(
            code,
            start="2026-08-15",
            end="2026-09-29",
            ktype=ft.KLType.K_60M,
            autype=ft.AuType.QFQ,
            max_count=500
        )

        if ret_kl != ft.RET_OK or len(df) == 0:
            print(f"[{i}/{len(all_codes)}] {code:10s} - ⚠️ Failed to fetch K-lines: {df}")
            time.sleep(0.1)
            continue

        df = df.sort_values("time_key").reset_index(drop=True)
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        p0 = float(df.iloc[0]["open"]) if "open" in df else float(closes[0])
        if p0 <= 0:
            p0 = float(closes[0])

        # If live cur_price is available and different, we can treat it as current
        if cur_price is None or cur_price <= 0:
            cur_price = float(closes[-1])

        p_min = float(np.min(lows))
        p_max = float(np.max(highs))

        ret_min = round(((p_min / p0) - 1.0) * 100.0, 2)
        ret_max = round(((p_max / p0) - 1.0) * 100.0, 2)
        ret_end = round(((cur_price / p0) - 1.0) * 100.0, 2)
        amplitude = round(((p_max - p_min) / p0) * 100.0, 2)
        max_drawdown = round(((p_max - p_min) / p_max) * 100.0, 2)

        # Simplify bar list for JSON
        bars = []
        for _, r in df.iterrows():
            c = float(r["close"])
            ret_pct = round(((c / p0) - 1.0) * 100.0, 2)
            bars.append({
                "time": str(r["time_key"]),
                "open": round(float(r["open"]), 2),
                "high": round(float(r["high"]), 2),
                "low": round(float(r["low"]), 2),
                "close": round(c, 2),
                "volume": int(r.get("volume", 0)),
                "ret_pct": ret_pct
            })

        trend_tag, trend_color = classify_trend(ret_end, ret_min, ret_max, closes)

        stock_item = {
            "code": code,
            "ticker": ticker,
            "name": name,
            "market": market,
            "is_favorite": code in fav_codes,
            "is_holding": code in positions,
            "holding_info": positions.get(code, None),
            "group": "自选" if code in fav_codes else "持仓",
            "p0": round(p0, 2),
            "cur_price": round(cur_price, 2),
            "last_close": round(last_close, 2) if last_close else round(cur_price, 2),
            "today_change_pct": today_chg,
            "ret_end_pct": ret_end,
            "ret_min_pct": ret_min,
            "ret_max_pct": ret_max,
            "min_price": round(p_min, 2),
            "max_price": round(p_max, 2),
            "amplitude_pct": amplitude,
            "max_drawdown_pct": max_drawdown,
            "turnover": turnover,
            "turnover_rate": turnover_rate,
            "pe_ratio": pe_ratio,
            "trend_tag": trend_tag,
            "trend_color": trend_color,
            "bars_count": len(bars),
            "start_time": bars[0]["time"],
            "end_time": bars[-1]["time"],
            "bars": bars
        }

        stocks_data.append(stock_item)
        print(f"[{i}/{len(all_codes)}] {code:10s} ({name}) | 30d Ret: {ret_end:+.2f}% | Today: {today_chg:+.2f}% | Range: [{ret_min:+.1f}%, {ret_max:+.1f}%] | Bars: {len(bars)}")
        time.sleep(0.12) # safety throttle

    quote_ctx.close()

    # 5. Output JSON
    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_stocks": len(stocks_data),
        "favorites_count": len([s for s in stocks_data if s["is_favorite"]]),
        "holdings_count": len([s for s in stocks_data if s["is_holding"]]),
        "stocks": stocks_data
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"SUCCESS: Saved {len(stocks_data)} stocks data to {OUTPUT_JSON} ({time.time() - t0:.1f}s)")
    print("=" * 70)

if __name__ == "__main__":
    main()
