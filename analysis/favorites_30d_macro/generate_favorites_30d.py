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

# Hourly motif codebook & centroids
CODEBOOK_PATH = ROOT_DIR / "analysis" / "qqq_hourly_motif_codebook" / "data.json"
CENTROIDS = None
CODEBOOK = None

W_INFO = {
    "W-01": {"name": "弧形底企稳", "color": "#06b6d4", "type": "深U反转"},
    "W-02": {"name": "超跌极值见底", "color": "#0284c7", "type": "恐慌探底"},
    "W-03": {"name": "深凹回抽破高", "color": "#10b981", "type": "强力反转"},
    "W-04": {"name": "凸性加速主升", "color": "#059669", "type": "加速主升"},
    "W-05": {"name": "平缓震荡走高", "color": "#3b82f6", "type": "稳步爬坡"},
    "W-06": {"name": "倒U见顶回落", "color": "#f97316", "type": "冲高受阻"},
    "W-07": {"name": "阶梯下行破位", "color": "#ea580c", "type": "台阶阴跌"},
    "W-08": {"name": "箱体收敛盘整", "color": "#8b5cf6", "type": "收敛整理"},
    "W-09": {"name": "弱势探底抽升", "color": "#6366f1", "type": "超跌反抽"},
    "W-10": {"name": "早期冲高滞涨", "color": "#f59e0b", "type": "高位横盘"},
    "W-11": {"name": "抛物线见顶下坠", "color": "#e11d48", "type": "加速见顶"},
    "W-12": {"name": "破位阴跌破底", "color": "#dc2626", "type": "单边破位"}
}

def load_codebook():
    global CENTROIDS, CODEBOOK
    if CODEBOOK_PATH.exists():
        with open(CODEBOOK_PATH, "r", encoding="utf-8") as f:
            cb_data = json.load(f)
        CODEBOOK = cb_data.get("codebook", [])
        CENTROIDS = np.array([c["curve"] for c in CODEBOOK])
        print(f"Loaded {len(CENTROIDS)} centroids from {CODEBOOK_PATH.name}")

def extract_matched_features(bars):
    if CENTROIDS is None or len(CENTROIDS) == 0:
        return []
    closes = np.array([b["close"] for b in bars])
    if len(closes) < 35:
        return []
    raw_matches = []
    for i in range(0, len(closes) - 35 + 1):
        sub = closes[i:i+35]
        std = float(np.std(sub))
        if std < 1e-6:
            continue
        z = (sub - np.mean(sub)) / std
        dists = np.linalg.norm(CENTROIDS - z, axis=1)
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        sim = max(0.0, 1.0 - (best_dist**2) / 70.0) * 100.0
        m_code = CODEBOOK[best_idx]["code"]
        info = W_INFO.get(m_code, {"name": m_code, "color": "#64748b", "type": "波形"})
        raw_matches.append({
            "start_idx": i,
            "end_idx": i + 34,
            "start_time": bars[i]["time"],
            "end_time": bars[i+34]["time"],
            "code": m_code,
            "name": info["name"],
            "type": info["type"],
            "color": info["color"],
            "sim": round(sim, 1),
            "hit3d_rate": round(CODEBOOK[best_idx]["stats"]["hit3d_rate"] * 100.0, 1)
        })
    raw_matches.sort(key=lambda x: x["sim"], reverse=True)
    selected = []
    for m in raw_matches:
        if m["sim"] < 78.0:
            break
        overlap = False
        for s in selected:
            inter_s = max(m["start_idx"], s["start_idx"])
            inter_e = min(m["end_idx"], s["end_idx"])
            if inter_e > inter_s:
                overlap_len = inter_e - inter_s
                if overlap_len > 12:
                    overlap = True
                    break
        if not overlap:
            selected.append(m)
            if len(selected) >= 3:
                break
    selected.sort(key=lambda x: x["start_idx"])
    return selected

def main():
    t0 = time.time()
    print("=" * 70)
    print("FETCHING 30-DAY MACRO TRENDS FOR USER FAVORITES FROM FUTU OPEND")
    print("=" * 70)

    load_codebook()
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
        matched_features = extract_matched_features(bars)

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
            "matched_features": matched_features,
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
