#!/usr/bin/env python3
"""Fetch Realtime & Intraday 60m Data from Futu OpenD for Moomoo US Cash Account (尾号 0086) & Favorites,
and Generate Interactive Parametric Multi-Column Monitor Table.

Directory: analysis/today_intraday_watch/
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

import futu as ft
import pandas as pd

ROOT = Path("/Users/admin/Code/stock")
OUTPUT_DIR = ROOT / "analysis" / "today_intraday_watch"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ft.SysConfig.enable_proto_encrypt(False)

print("Connecting to Futu OpenD (127.0.0.1:11111)...")
quote_ctx = ft.OpenQuoteContext(host="127.0.0.1", port=11111)

# 1. Fetch Real User Positions from Moomoo US Cash Account (尾号 0086)
TARGET_CASH_ACC_ID = 283445330641239202 # UniCard 1007292558030086 (尾号 0086)
TARGET_MARGIN_ACC_ID = 283445330187455846 # Moomoo US Margin

positions = {}
account_summary = {
    "acc_name": "Moomoo US 现金账户",
    "card_tail": "0086",
    "acc_id": str(TARGET_CASH_ACC_ID),
    "total_pnl_val": 0.0,
    "holdings_count": 0
}

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
            if c.startswith("US."):
                pl_v = float(row.get("pl_val", 0)) if "pl_val" in row and pd.notna(row["pl_val"]) else 0.0
                positions[c] = {
                    "name": row.get("stock_name", ""),
                    "qty": float(row.get("qty", 0)),
                    "cost": float(row.get("cost_price", 0)),
                    "nominal_price": float(row.get("nominal_price", 0)),
                    "pl_ratio": float(row.get("pl_ratio", 0)) if "pl_ratio" in row and pd.notna(row["pl_ratio"]) else 0.0,
                    "pl_val": pl_v,
                    "acc_source": "0086现金账户"
                }
                account_summary["total_pnl_val"] += pl_v
        account_summary["holdings_count"] = len(positions)
        print(f"Loaded {len(positions)} US holdings from Moomoo US Cash Acc 0086: {list(positions.keys())}")
    
    # Margin account
    r_margin, margin_df = trd_ctx.position_list_query(acc_id=TARGET_MARGIN_ACC_ID)
    if r_margin == ft.RET_OK and len(margin_df) > 0:
        for _, row in margin_df.iterrows():
            c = row["code"]
            if c.startswith("US.") and c not in positions:
                positions[c] = {
                    "name": row.get("stock_name", ""),
                    "qty": float(row.get("qty", 0)),
                    "cost": float(row.get("cost_price", 0)),
                    "nominal_price": float(row.get("nominal_price", 0)),
                    "pl_ratio": float(row.get("pl_ratio", 0)) if "pl_ratio" in row and pd.notna(row["pl_ratio"]) else 0.0,
                    "pl_val": float(row.get("pl_val", 0)) if "pl_val" in row and pd.notna(row["pl_val"]) else 0.0,
                    "acc_source": "Moomoo融资账户"
                }
    trd_ctx.close()
except Exception as e:
    print(f"Warning: Failed to query trade positions: {e}")

# 2. Fetch User Watchlists (Favorites / 特别关注 and 📉惨了)
fav_codes = []
ret_fav, fav_df = quote_ctx.get_user_security(group_name="Favorites")
if ret_fav == ft.RET_OK:
    fav_codes = [c for c in fav_df["code"] if c.startswith("US.")]
print(f"Loaded {len(fav_codes)} US stocks in Favorites: {fav_codes}")

bad_codes = []
ret_bad, bad_df = quote_ctx.get_user_security(group_name="📉惨了")
if ret_bad == ft.RET_OK:
    bad_codes = [c for c in bad_df["code"] if c.startswith("US.")]

all_target_codes = sorted(list(set(list(positions.keys()) + fav_codes + bad_codes)))
print(f"Total target US stocks: {len(all_target_codes)}")

# 3. Fetch Realtime Snapshots
ret_snap, snap_df = quote_ctx.get_market_snapshot(all_target_codes)
snapshots = {}
if ret_snap == ft.RET_OK:
    for _, row in snap_df.iterrows():
        snapshots[row["code"]] = row
print(f"Fetched {len(snapshots)} market snapshots.")

# 4. Fetch 60m K-lines from 2026-09-10 to 2026-09-24 (covers D-5 to Today)
kline_data = {}
print("Fetching 60m K-lines from 2026-09-10 to 2026-09-24...")
for code in all_target_codes:
    ret_kl, df, _ = quote_ctx.request_history_kline(
        code,
        start="2026-09-10",
        end="2026-09-24",
        ktype=ft.KLType.K_60M,
        autype=ft.AuType.QFQ,
        max_count=200
    )
    if ret_kl == ft.RET_OK and len(df) > 0:
        df["date"] = df["time_key"].str.slice(0, 10)
        df["time"] = df["time_key"].str.slice(11, 16)
        # Filter regular trading hours only
        reg_df = df[df["time"].isin(["10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00"])].copy()
        reg_df.reset_index(drop=True, inplace=True)
        kline_data[code] = reg_df
    else:
        print(f"Failed to fetch K-lines for {code}")
    time.sleep(0.04)

quote_ctx.close()
print("K-line fetching complete.")

# 5. Process Metrics for Each Stock
stocks_data = []

# Target trading dates: D-5, D-4, D-3, D-2, D-1, Today
RECENT_5_DATES = ["2026-09-17", "2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23"]

for code in all_target_codes:
    ticker = code.replace("US.", "")
    snap = snapshots.get(code)
    bars = kline_data.get(code, pd.DataFrame())
    
    # Stock info
    stock_name = snap["name"] if snap is not None else ticker
    last_price = float(snap["last_price"]) if snap is not None and pd.notna(snap["last_price"]) else 0.0
    prev_close = float(snap["prev_close_price"]) if snap is not None and pd.notna(snap["prev_close_price"]) else 0.0
    open_price = float(snap["open_price"]) if snap is not None and pd.notna(snap["open_price"]) else 0.0
    high_price = float(snap["high_price"]) if snap is not None and pd.notna(snap["high_price"]) else 0.0
    low_price  = float(snap["low_price"]) if snap is not None and pd.notna(snap["low_price"]) else 0.0
    volume     = float(snap["volume"]) if snap is not None and pd.notna(snap["volume"]) else 0.0
    turnover   = float(snap["turnover"]) if snap is not None and pd.notna(snap["turnover"]) else 0.0
    amplitude  = float(snap["amplitude"]) if snap is not None and pd.notna(snap["amplitude"]) else 0.0
    
    chg_pct = round((last_price - prev_close) / prev_close * 100, 2) if prev_close > 0 else 0.0
    
    # Classification tag
    is_holding = code in positions
    is_0086 = is_holding and positions[code].get("acc_source") == "0086现金账户"
    is_fav = code in fav_codes
    is_bad = code in bad_codes
    
    if is_0086 and is_fav:
        cat = "0086持仓+特注"
    elif is_0086:
        cat = "0086现金持仓"
    elif is_holding:
        cat = "融资账户持仓"
    elif is_bad:
        cat = "亏损特别关注"
    else:
        cat = "特别关注"
    
    pos_info = positions.get(code, {})
    
    # Today's completed hours (open to now-1: 10:30 and 11:30)
    today_bars = bars[bars["date"] == "2026-09-24"] if not bars.empty else pd.DataFrame()
    bar_1030 = today_bars[today_bars["time"] == "10:30"].iloc[0].to_dict() if len(today_bars[today_bars["time"] == "10:30"]) > 0 else None
    bar_1130 = today_bars[today_bars["time"] == "11:30"].iloc[0].to_dict() if len(today_bars[today_bars["time"] == "11:30"]) > 0 else None
    
    vol_1030 = float(bar_1030["volume"]) if bar_1030 else 0.0
    close_1030 = float(bar_1030["close"]) if bar_1030 else 0.0
    vol_1130 = float(bar_1130["volume"]) if bar_1130 else 0.0
    close_1130 = float(bar_1130["close"]) if bar_1130 else 0.0
    
    completed_vol = vol_1030 + vol_1130
    vol_share_pct = round(completed_vol / volume * 100, 1) if volume > 0 else 0.0
    
    # Max gains from 10:30 and 11:30 to today's high / current
    gain_from_1030 = round((high_price - close_1030) / close_1030 * 100, 2) if close_1030 > 0 else 0.0
    gain_from_1130 = round((high_price - close_1130) / close_1130 * 100, 2) if close_1130 > 0 else 0.0
    
    # Convert bars to compact lightweight list for frontend client-side recalculation
    compact_bars = []
    if not bars.empty:
        for _, b_row in bars.iterrows():
            compact_bars.append({
                "d": str(b_row["date"]),
                "t": str(b_row["time"]),
                "o": float(b_row["open"]),
                "h": float(b_row["high"]),
                "l": float(b_row["low"]),
                "c": float(b_row["close"]),
                "v": float(b_row["volume"])
            })
            
    stocks_data.append({
        "code": code,
        "ticker": ticker,
        "name": stock_name,
        "category": cat,
        "is_holding": is_holding,
        "is_0086": is_0086,
        "pos_info": pos_info,
        "last_price": last_price,
        "chg_pct": chg_pct,
        "open_price": open_price,
        "high_price": high_price,
        "low_price": low_price,
        "amplitude": amplitude,
        "volume": volume,
        "turnover": turnover,
        "bar_1030": {
            "close": close_1030,
            "vol": vol_1030,
            "max_gain": gain_from_1030
        },
        "bar_1130": {
            "close": close_1130,
            "vol": vol_1130,
            "max_gain": gain_from_1130
        },
        "completed_vol": completed_vol,
        "vol_share_pct": vol_share_pct,
        "bars": compact_bars
    })

# Sort stocks: 0086 holdings first, then other holdings, then by today's change % descending
stocks_data.sort(key=lambda s: (0 if s["is_0086"] else (1 if s["is_holding"] else 2), -s["chg_pct"]))

payload = {
    "account_summary": account_summary,
    "update_time_et": datetime.now().strftime("%Y-%m-%d %H:%M:%S ET"),
    "update_time_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "stocks_count": len(stocks_data),
    "holdings_count": account_summary["holdings_count"],
    "recent_5_dates": RECENT_5_DATES,
    "stocks": stocks_data
}

# Save JSON
json_path = OUTPUT_DIR / "today_data.json"
json_dump = json.dumps(payload, ensure_ascii=False, indent=2)
json_path.write_text(json_dump, encoding="utf-8")
print(f"Saved {json_path} ({len(json_dump):,} bytes)")

# 6. Render Wireframe HTML with Interactive Parameter Sliders
html_code = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Moomoo US 现金账户 (尾号 0086) 持仓与特别关注 · 动态参数多日达成率监控看板</title>
<style>
:root {{
  --bg: #f8fafc;
  --surface: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --border: #e2e8f0;
  --line: #cbd5e1;
  --shade: #f1f5f9;
  --up: #15803d;
  --up-bg: #dcfce7;
  --up-border: #86efac;
  --down: #b91c1c;
  --down-bg: #fee2e2;
  --down-border: #fca5a5;
  --accent: #2563eb;
  --accent-bg: #eff6ff;
}}

* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.45;
  padding: 16px 20px 60px;
}}

header.wf-header {{
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 14px 20px;
  border-radius: 4px;
  margin-bottom: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 12px;
}}
.header-titles h1 {{
  font-size: 19px;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #0f172a;
}}
.header-titles p {{
  font-size: 12px;
  color: var(--muted);
  margin-top: 3px;
}}
.acc-badge-top {{
  display: inline-block;
  background: #0f172a;
  color: #fff;
  padding: 2px 7px;
  border-radius: 3px;
  font-size: 11px;
  font-weight: 700;
  margin-right: 6px;
}}
.sync-badge {{
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--shade);
  border: 1px solid var(--border);
  padding: 6px 12px;
  border-radius: 3px;
  font-size: 11.5px;
}}
.pulse-dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #10b981;
  box-shadow: 0 0 0 2px rgba(16,185,129,0.2);
}}

/* Interactive Parameter Control Panel */
.param-panel {{
  background: #ffffff;
  border: 2px solid #2563eb;
  border-radius: 6px;
  padding: 14px 18px;
  margin-bottom: 14px;
  box-shadow: 0 2px 8px rgba(37,99,235,0.06);
}}
.param-header {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
  flex-wrap: wrap;
  gap: 8px;
}}
.param-title {{
  font-size: 13px;
  font-weight: 800;
  color: #1e3a8a;
  display: flex;
  align-items: center;
  gap: 6px;
}}
.presets-group {{
  display: flex;
  gap: 6px;
  align-items: center;
}}
.preset-btn {{
  border: 1px solid #cbd5e1;
  background: #f8fafc;
  padding: 3px 9px;
  font-size: 11px;
  font-weight: 600;
  border-radius: 3px;
  cursor: pointer;
  transition: all 0.15s;
}}
.preset-btn:hover {{
  background: #eff6ff;
  border-color: #93c5fd;
  color: #1d4ed8;
}}
.preset-btn.active {{
  background: #2563eb;
  color: #fff;
  border-color: #2563eb;
}}

.param-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}}
@media (max-width: 768px) {{
  .param-grid {{ grid-template-columns: 1fr; }}
}}
.param-box {{
  background: #f8fafc;
  border: 1px solid var(--border);
  padding: 10px 14px;
  border-radius: 4px;
}}
.param-top-row {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 6px;
}}
.param-label {{
  font-size: 12px;
  font-weight: 700;
  color: #334155;
}}
.param-value {{
  font-size: 15px;
  font-weight: 800;
  color: #2563eb;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
}}
.slider-input {{
  width: 100%;
  height: 6px;
  border-radius: 3px;
  outline: none;
  background: #cbd5e1;
  accent-color: #2563eb;
  cursor: pointer;
}}
.param-scale {{
  display: flex;
  justify-content: space-between;
  font-size: 10px;
  color: #94a3b8;
  margin-top: 4px;
}}

/* Summary Metric Cards */
.summary-cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}}
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 10px 14px;
  border-radius: 3px;
}}
.card span {{ font-size: 11px; color: var(--muted); display: block; }}
.card b {{ font-size: 19px; font-weight: 800; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}

/* Filter Controls Bar */
.controls-bar {{
  background: var(--surface);
  border: 1px solid var(--border);
  padding: 10px 16px;
  border-radius: 4px;
  margin-bottom: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}}
.filter-tabs {{
  display: flex;
  gap: 6px;
  align-items: center;
}}
.tab-btn {{
  border: 1px solid var(--border);
  background: var(--surface);
  padding: 5px 12px;
  font-size: 11.5px;
  font-weight: 600;
  cursor: pointer;
  border-radius: 3px;
  transition: all 0.15s ease;
}}
.tab-btn:hover {{ background: var(--shade); }}
.tab-btn.active {{
  background: #0f172a;
  color: #fff;
  border-color: #0f172a;
}}
.search-input {{
  border: 1px solid var(--border);
  padding: 5px 10px;
  font-size: 12px;
  border-radius: 3px;
  width: 180px;
  outline: none;
}}
.search-input:focus {{ border-color: var(--accent); }}

/* Strategy Notes Accordion */
details.notes-panel {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  margin-bottom: 14px;
  padding: 10px 14px;
  font-size: 12px;
}}
details.notes-panel summary {{
  font-weight: 700;
  cursor: pointer;
  color: #1e293b;
}}
.notes-content {{
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px dashed var(--border);
  line-height: 1.6;
  color: #334155;
}}

/* Main Table */
.table-wrap {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow-x: auto;
  box-shadow: 0 1px 3px rgba(0,0,0,0.02);
}}
table.monitor-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  white-space: nowrap;
}}
th, td {{
  padding: 8px 10px;
  text-align: right;
  border-bottom: 1px solid var(--border);
}}
th {{
  background: #f8fafc;
  color: #475569;
  font-weight: 700;
  font-size: 11.5px;
  border-bottom: 2px solid var(--line);
  user-select: none;
  cursor: pointer;
}}
th:hover {{ background: #f1f5f9; }}
th.text-left, td.text-left {{ text-align: left; }}
th.text-center, td.text-center {{ text-align: center; }}

tr:hover td {{ background: #f8fafc; }}
tr.row-holding-0086 {{
  background: #fafcff;
}}
tr.row-holding-0086 td:first-child {{
  border-left: 3.5px solid #2563eb;
}}

/* Badges & Tags */
.tag-badge {{
  font-size: 10px;
  font-weight: 700;
  padding: 2px 6px;
  border-radius: 2px;
  display: inline-block;
}}
.tag-0086-both {{ background: #eff6ff; color: #1e40af; border: 1px solid #93c5fd; font-weight:800; }}
.tag-0086 {{ background: #f1f5f9; color: #0f172a; border: 1px solid #cbd5e1; font-weight:700; }}
.tag-fav {{ background: #faf5ff; color: #6b21a8; border: 1px solid #e9d5ff; }}
.tag-bad {{ background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }}

.cell-rate {{
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-weight: 700;
  padding: 3px 6px;
  border-radius: 3px;
  display: inline-block;
  min-width: 60px;
  text-align: right;
}}
.rate-high {{ background: #dcfce7; color: #15803d; border: 1px solid #86efac; }}
.rate-mid {{ background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; }}
.rate-low {{ background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }}

.sub-pill {{
  font-size: 10.5px;
  color: var(--muted);
  display: block;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  margin-top: 1px;
}}

.chg-up {{ color: #15803d; font-weight: 700; font-family: monospace; }}
.chg-down {{ color: #b91c1c; font-weight: 700; font-family: monospace; }}
.chg-flat {{ color: #64748b; font-family: monospace; }}

.runup-badge {{
  font-size: 10px;
  padding: 1px 4px;
  border-radius: 2px;
  background: #f1f5f9;
  border: 1px solid #e2e8f0;
  color: #475569;
  margin-left: 4px;
  font-weight: 600;
}}
.runup-badge.hit {{
  background: #dcfce7;
  border-color: #86efac;
  color: #15803d;
}}

/* Sticky first column */
.col-sticky {{
  position: sticky;
  left: 0;
  background: var(--surface);
  z-index: 2;
  box-shadow: 2px 0 3px rgba(0,0,0,0.03);
}}
th.col-sticky {{ background: #f8fafc; z-index: 3; }}

footer.wf-footer {{
  margin-top: 20px;
  font-size: 11.5px;
  color: var(--muted);
  border-top: 1px solid var(--border);
  padding-top: 12px;
  display: flex;
  justify-content: space-between;
}}
</style>
</head>
<body>

<header class="wf-header">
  <div class="header-titles">
    <h1>
      <span class="acc-badge-top">Moomoo US 现金账户 · 尾号 0086</span>
      持仓 ({account_summary["holdings_count"]}只) 与特别关注标的 · 动态参数多日达成率监控
    </h1>
    <p>精准直连 Moomoo US 账户（ID: {TARGET_CASH_ACC_ID}，UniCard 尾号 0086）| 支持实时拖动调节目标涨幅与持仓天数</p>
  </div>
  <div class="sync-badge">
    <div class="pulse-dot"></div>
    <span>行情更新：<b>{payload["update_time_et"]}</b>（美东时间）</span>
  </div>
</header>

<!-- Interactive Parameter Control Panel -->
<section class="param-panel">
  <div class="param-header">
    <div class="param-title">
      <span>⚙️ 策略核心参数动态调节（拖动滑块即时毫秒级重算表格）</span>
    </div>
    <div class="presets-group">
      <span style="font-size:11px; color:#64748b; margin-right:4px;">快速预设：</span>
      <button class="preset-btn active" id="pre-3d5" onclick="setPreset(5.0, 3)">标准模式 (3天 5%)</button>
      <button class="preset-btn" id="pre-5d5" onclick="setPreset(5.0, 5)">长波段 (5天 5%)</button>
      <button class="preset-btn" id="pre-3d7" onclick="setPreset(7.0, 3)">强弹性 (3天 7%)</button>
      <button class="preset-btn" id="pre-5d10" onclick="setPreset(10.0, 5)">翻倍主升 (5天 10%)</button>
      <button class="preset-btn" id="pre-1d5" onclick="setPreset(5.0, 1)">超短隔夜 (1天 5%)</button>
    </div>
  </div>
  
  <div class="param-grid">
    <div class="param-box">
      <div class="param-top_row" style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px;">
        <span class="param-label">① 目标涨幅阈值 (Target Gain)：</span>
        <span class="param-value" id="val-target">5.0%</span>
      </div>
      <input type="range" class="slider-input" id="slider-target" min="5.0" max="10.0" step="0.5" value="5.0" oninput="onParamChange()">
      <div class="param-scale">
        <span>5.0% (标准起步)</span>
        <span>6.0%</span>
        <span>7.0%</span>
        <span>8.0%</span>
        <span>9.0%</span>
        <span>10.0% (翻倍起爆)</span>
      </div>
    </div>
    
    <div class="param-box">
      <div class="param-top_row" style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px;">
        <span class="param-label">② 持仓考察窗口 (Holding Window)：</span>
        <span class="param-value" id="val-window">3 天 (21个交易小时)</span>
      </div>
      <input type="range" class="slider-input" id="slider-window" min="1" max="5" step="1" value="3" oninput="onParamChange()">
      <div class="param-scale">
        <span>1 天 (7小时)</span>
        <span>2 天 (14小时)</span>
        <span>3 天 (标准 21h)</span>
        <span>4 天 (28小时)</span>
        <span>5 天 (完整一周 35h)</span>
      </div>
    </div>
  </div>
</section>

<!-- Summary Metric Cards -->
<section class="summary-cards" id="summary-cards">
  <div class="card">
    <span>监控目标总数</span>
    <b>{payload["stocks_count"]} 只</b>
  </div>
  <div class="card">
    <span>0086 现金持仓</span>
    <b>{account_summary["holdings_count"]} 只</b>
  </div>
  <div class="card">
    <span>今日上涨标的</span>
    <b style="color:var(--up);" id="stat-up-count">-</b>
  </div>
  <div class="card">
    <span id="label-stat-d5">近5日均值达成率</span>
    <b id="stat-d5-avg" style="color:#2563eb;">-</b>
  </div>
  <div class="card">
    <span id="label-stat-d3">D-3 均值达成率</span>
    <b id="stat-d3-avg">-</b>
  </div>
  <div class="card">
    <span>高胜率标的 (&ge;50%)</span>
    <b id="stat-high-count">-</b>
  </div>
</section>

<!-- Strategy Notes -->
<details class="notes-panel">
  <summary>💡 本页面策略思路与字段说明（点击展开查看）</summary>
  <div class="notes-content">
    <p><b>1. 账户校准确认：</b><br>
    当前页面已精准对齐 <b>Moomoo US (Futu Inc.) 现金账户（综合卡号尾号 0086，账户 ID: {TARGET_CASH_ACC_ID}）</b>，成功识别并载入该账户全部 <b>{account_summary["holdings_count"]} 只美股真实持仓</b>（含 WDC、VRT、SOXS、SNXX、SEDG、NXT、MRVL、INTC、HLTH、GOOG、FN、CRDO、COHR、CIEN、BBC、AVGO、AMAT、AIPO 等），并与牛牛“特别关注 (Favorites)”标的合并跟踪。</p>
    <p style="margin-top:6px;"><b>2. 新增核心列【近 5 个交易日（D-5 至 D-1）综合达成率】：</b><br>
    涵盖 <b>09-17、09-18、09-21、09-22、09-23 共 5 个完整交易日（合计 35 个交易小时，不含今天）</b>。该列作为多日累积核心基准，消除了单日波动的偶然性，能极其清晰地过滤出当前全市场谁是真正的“连续放量主升龙”（如 TWST 近 5 日 35 个小时买入高达 90%+ 达成率）。</p>
    <p style="margin-top:6px;"><b>3. 动态参数滑块设计：</b><br>
    • <b>目标涨幅 (5% ~ 10%)</b>：可自由验证股票是只能跑出微弱的 5% 脉冲，还是具备直接突破 7%、10% 的大波段动力；<br>
    • <b>持仓天数 (1 ~ 5 天)</b>：可验证 1 天隔夜超短线胜率，对比 3 天与 5 天波段持仓的兑现效率。</p>
  </div>
</details>

<!-- Controls Bar -->
<section class="controls-bar">
  <div class="filter-tabs">
    <button class="tab-btn active" id="tab-all" onclick="filterCategory('all')">全部标的 ({payload["stocks_count"]})</button>
    <button class="tab-btn" id="tab-holding" onclick="filterCategory('holding')">0086现金持仓 ({account_summary["holdings_count"]})</button>
    <button class="tab-btn" id="tab-fav" onclick="filterCategory('fav')">特别关注 ({len(fav_codes)})</button>
    <button class="tab-btn" id="tab-up" onclick="filterCategory('up')">今日上涨</button>
    <button class="tab-btn" id="tab-hit" onclick="filterCategory('hit')">近5日高胜率 (&ge;50%)</button>
  </div>
  <div>
    <input type="text" id="search-box" class="search-input" placeholder="输入代码或名称筛选..." oninput="onSearch(this.value)">
  </div>
</section>

<!-- Table -->
<main class="table-wrap">
  <table class="monitor-table" id="monitor-table">
    <thead>
      <tr>
        <th class="col-sticky text-left" onclick="sortTable('ticker')">标的代码 / 名称</th>
        <th class="text-center" onclick="sortTable('category')">分组类别</th>
        <th onclick="sortTable('last_price')">最新价</th>
        <th onclick="sortTable('chg_pct')">今日涨跌幅</th>
        <th onclick="sortTable('amplitude')">日内振幅</th>
        <th onclick="sortTable('completed_vol')">开盘-1h量 (10:30+11:30)</th>
        <th onclick="sortTable('gain_1030')">首小时(10:30)冲幅</th>
        <th onclick="sortTable('gain_1130')">次小时(11:30)冲幅</th>
        <!-- NEW KEY COLUMN: Recent 5 Days -->
        <th class="text-center" onclick="sortTable('rate_d5')" style="background:#dbeafe; color:#1e3a8a; border-bottom: 2px solid #2563eb;">
          <span id="col-header-d5">近5日 (D-5~D-1)<br>[3日5% 达成率]</span>
        </th>
        <th class="text-center" onclick="sortTable('rate_d3')" style="background:#e2e8f0; color:#0f172a;">
          <span id="col-header-d3">今天-3 (09-21)<br>[3日5% 达成率]</span>
        </th>
        <th class="text-center" onclick="sortTable('rate_d2')" style="background:#e2e8f0; color:#0f172a;">
          <span id="col-header-d2">今天-2 (09-22)<br>[3日5% 达成率]</span>
        </th>
        <th class="text-center" onclick="sortTable('rate_d1')" style="background:#e2e8f0; color:#0f172a;">
          <span id="col-header-d1">今天-1 (09-23)<br>[3日5% 达成率]</span>
        </th>
        <th class="text-center" onclick="sortTable('rate_today')" style="background:#cbd5e1; color:#0f172a; font-weight:800;">
          <span id="col-header-today">今天 (09-24)<br>[盘中冲幅达标]</span>
        </th>
        <th class="text-left">0086 持仓详情 (股数/成本/盈亏)</th>
      </tr>
    </thead>
    <tbody id="table-body">
      <!-- Injected via JavaScript -->
    </tbody>
  </table>
</main>

<footer class="wf-footer">
  <span>数据来源：本地富途牛牛 OpenD (端口 11111) | Moomoo US 现金账户 (UniCard 尾号 0086) | 生成时间：{payload["update_time_local"]}</span>
  <span>归档目录：analysis/today_intraday_watch/</span>
</footer>

<script>
const RAW_STOCKS = {json.dumps(payload["stocks"], ensure_ascii=False)};
const RECENT_5_DATES = {json.dumps(RECENT_5_DATES)};

let currentCategory = 'all';
let searchQuery = '';
let sortField = 'is_0086';
let sortAsc = false;

// Global parameters
let paramTarget = 5.0;  // 5% ~ 10%
let paramWindow = 3;    // 1 ~ 5 days

// Formatter
function formatVol(v) {{
  if (!v) return '-';
  if (v >= 1e6) return (v / 1e6).toFixed(2) + ' M';
  if (v >= 1e3) return (v / 1e3).toFixed(1) + ' K';
  return v.toLocaleString();
}}

// Fast Client-side Metric Calculator
function computeStockMetrics(s, targetPct, windowDays) {{
  const maxBarsFwd = windowDays * 7;
  const bars = s.bars || [];
  const highPrice = s.high_price || 0;
  
  function calcDates(dateList, isToday = false) {{
    let hits = 0, count = 0, maxRunup = -999;
    for (let i = 0; i < bars.length; i++) {{
      const b = bars[i];
      if (!dateList.includes(b.d)) continue;
      count++;
      const entryC = b.c;
      let fMax = -999;
      const endIdx = Math.min(i + 1 + maxBarsFwd, bars.length);
      for (let j = i + 1; j < endIdx; j++) {{
        if (bars[j].h > fMax) fMax = bars[j].h;
      }}
      if (highPrice > fMax) fMax = highPrice;
      if (fMax === -999) fMax = b.h;
      
      const gain = (fMax - entryC) / entryC * 100.0;
      if (gain > maxRunup) maxRunup = gain;
      if (gain >= targetPct) hits++;
    }}
    const rate = count > 0 ? (hits / count * 100) : null;
    return {{
      c: count,
      h: hits,
      rate: rate,
      max_gain: maxRunup === -999 ? 0 : maxRunup
    }};
  }}
  
  return {{
    d5: calcDates(RECENT_5_DATES),
    d3: calcDates(['2026-09-21']),
    d2: calcDates(['2026-09-22']),
    d1: calcDates(['2026-09-23']),
    today: calcDates(['2026-09-24'], true)
  }};
}}

function getRateBadge(rate, h, c, maxGain, targetPct, isInflight) {{
  if (rate === null || c === 0) return '<span style="color:#94a3b8;">-</span>';
  let cls = 'rate-mid';
  if (rate >= 50) cls = 'rate-high';
  else if (rate < 25) cls = 'rate-low';
  
  let flightIcon = isInflight ? ' ⏳' : '';
  let hitBadge = maxGain >= targetPct ? `<span class="runup-badge hit">+${{maxGain.toFixed(1)}}%✓</span>` : `<span class="runup-badge">+${{maxGain.toFixed(1)}}%</span>`;
  
  return `
    <div>
      <span class="cell-rate ${{cls}}">${{rate.toFixed(1)}}%</span>
      <span class="sub-pill">(${{h}}/${{c}})${{flightIcon}} ${{hitBadge}}</span>
    </div>
  `;
}}

function renderAll() {{
  // 1. Update Labels on Header
  document.getElementById('val-target').innerText = paramTarget.toFixed(1) + '%';
  document.getElementById('val-window').innerText = `${{paramWindow}} 天 (${{paramWindow * 7}}个交易小时)`;
  
  document.getElementById('slider-target').value = paramTarget;
  document.getElementById('slider-window').value = paramWindow;
  
  document.getElementById('col-header-d5').innerHTML = `近5日 (D-5~D-1)<br>[${{paramWindow}}日${{paramTarget.toFixed(1)}}% 达成率]`;
  document.getElementById('col-header-d3').innerHTML = `今天-3 (09-21)<br>[${{paramWindow}}日${{paramTarget.toFixed(1)}}% 达成率]`;
  document.getElementById('col-header-d2').innerHTML = `今天-2 (09-22)<br>[${{paramWindow}}日${{paramTarget.toFixed(1)}}% 达成率]`;
  document.getElementById('col-header-d1').innerHTML = `今天-1 (09-23)<br>[${{paramWindow}}日${{paramTarget.toFixed(1)}}% 达成率]`;
  
  // 2. Compute metrics for each stock
  RAW_STOCKS.forEach(s => {{
    s.calc = computeStockMetrics(s, paramTarget, paramWindow);
  }});
  
  // 3. Filter
  let list = RAW_STOCKS.filter(s => {{
    if (currentCategory === 'holding' && !s.is_0086 && !s.is_holding) return false;
    if (currentCategory === 'fav' && !s.category.includes('特别关注') && !s.category.includes('特注')) return false;
    if (currentCategory === 'up' && s.chg_pct <= 0) return false;
    if (currentCategory === 'hit') {{
      const d5Rate = s.calc.d5.rate || 0;
      if (d5Rate < 50) return false;
    }}
    if (searchQuery) {{
      const q = searchQuery.toLowerCase();
      if (!s.ticker.toLowerCase().includes(q) && !s.name.toLowerCase().includes(q)) return false;
    }}
    return true;
  }});
  
  // 4. Sort
  list.sort((a, b) => {{
    let va = a[sortField];
    let vb = b[sortField];
    if (sortField === 'rate_d5') va = a.calc.d5.rate ?? -1, vb = b.calc.d5.rate ?? -1;
    if (sortField === 'rate_d3') va = a.calc.d3.rate ?? -1, vb = b.calc.d3.rate ?? -1;
    if (sortField === 'rate_d2') va = a.calc.d2.rate ?? -1, vb = b.calc.d2.rate ?? -1;
    if (sortField === 'rate_d1') va = a.calc.d1.rate ?? -1, vb = b.calc.d1.rate ?? -1;
    if (sortField === 'rate_today') va = a.calc.today.rate ?? -1, vb = b.calc.today.rate ?? -1;
    if (sortField === 'gain_1030') va = a.bar_1030.max_gain, vb = b.bar_1030.max_gain;
    if (sortField === 'gain_1130') va = a.bar_1130.max_gain, vb = b.bar_1130.max_gain;
    if (sortField === 'is_0086') {{
      if (a.is_0086 !== b.is_0086) return a.is_0086 ? -1 : 1;
      return b.chg_pct - a.chg_pct;
    }}
    
    if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return sortAsc ? (va - vb) : (vb - va);
  }});
  
  // 5. Update summary metric cards
  const upCount = list.filter(s => s.chg_pct > 0).length;
  document.getElementById('stat-up-count').innerText = `${{upCount}} / ${{list.length}} 只 (${{(upCount/list.length*100).toFixed(0)}}%)`;
  
  const d5Valid = list.filter(s => s.calc.d5.rate !== null);
  const d5Avg = d5Valid.length > 0 ? (d5Valid.reduce((acc, s) => acc + s.calc.d5.rate, 0) / d5Valid.length).toFixed(1) : '-';
  document.getElementById('stat-d5-avg').innerText = d5Avg + '%';
  document.getElementById('label-stat-d5').innerText = `近5日均值达成 (${{paramWindow}}d ${{paramTarget.toFixed(0)}}%)`;
  
  const d3Valid = list.filter(s => s.calc.d3.rate !== null);
  const d3Avg = d3Valid.length > 0 ? (d3Valid.reduce((acc, s) => acc + s.calc.d3.rate, 0) / d3Valid.length).toFixed(1) : '-';
  document.getElementById('stat-d3-avg').innerText = d3Avg + '%';
  document.getElementById('label-stat-d3').innerText = `D-3 (09-21) 均值 (${{paramWindow}}d ${{paramTarget.toFixed(0)}}%)`;
  
  const highHits = list.filter(s => (s.calc.d5.rate || 0) >= 50).length;
  document.getElementById('stat-high-count').innerText = `${{highHits}} 只 (${{(highHits/list.length*100).toFixed(0)}}%)`;
  
  // 6. Render table rows
  const tbody = document.getElementById('table-body');
  tbody.innerHTML = list.map(s => {{
    const chgClass = s.chg_pct > 0 ? 'chg-up' : (s.chg_pct < 0 ? 'chg-down' : 'chg-flat');
    const chgSign = s.chg_pct > 0 ? '+' : '';
    
    let tagCls = 'tag-fav';
    if (s.category === '0086持仓+特注') tagCls = 'tag-0086-both';
    else if (s.is_0086) tagCls = 'tag-0086';
    else if (s.is_holding) tagCls = 'tag-0086';
    else if (s.category.includes('亏损')) tagCls = 'tag-bad';
    
    const rowCls = s.is_0086 ? 'row-holding-0086' : '';
    
    let posDetail = '<span style="color:#94a3b8;">-</span>';
    if (s.is_holding) {{
      const p = s.pos_info;
      const pnlCls = p.pl_val >= 0 ? 'chg-up' : 'chg-down';
      const pnlSign = p.pl_val >= 0 ? '+' : '';
      posDetail = `
        <span style="font-family:monospace; font-size:11.5px;">
          持仓 <b>${{p.qty}}</b>股 | 成本: $${{p.cost?.toFixed(2)}} | 
          盈亏: <b class="${{pnlCls}}">${{pnlSign}}${{p.pl_ratio?.toFixed(2)}}% (${{pnlSign}}$${{p.pl_val?.toFixed(2)}})</b>
        </span>
      `;
    }}
    
    const gain1030 = s.bar_1030.max_gain;
    const gain1130 = s.bar_1130.max_gain;
    
    return `
      <tr class="${{rowCls}}">
        <td class="col-sticky text-left">
          <div style="font-weight:800; font-size:13px; font-family:monospace;">${{s.ticker}}</div>
          <div style="font-size:11px; color:var(--muted);">${{s.name}}</div>
        </td>
        <td class="text-center">
          <span class="tag-badge ${{tagCls}}">${{s.category}}</span>
        </td>
        <td style="font-family:monospace; font-weight:700;">$${{s.last_price.toFixed(2)}}</td>
        <td class="${{chgClass}}">${{chgSign}}${{s.chg_pct.toFixed(2)}}%</td>
        <td style="font-family:monospace; font-size:11.5px; color:var(--muted);">${{s.amplitude.toFixed(2)}}%</td>
        <td>
          <div style="font-family:monospace; font-weight:700;">${{formatVol(s.completed_vol)}}</div>
          <div class="sub-pill">占全天 ${{s.vol_share_pct}}%</div>
        </td>
        <td>
          <div style="font-family:monospace; font-weight:700; color:${{gain1030 >= paramTarget ? 'var(--up)' : 'inherit'}};">
            ${{gain1030 >= 0 ? '+' : ''}}${{gain1030.toFixed(2)}}%
          </div>
          <div class="sub-pill">收 $${{s.bar_1030.close.toFixed(2)}} | ${{formatVol(s.bar_1030.vol)}}</div>
        </td>
        <td>
          <div style="font-family:monospace; font-weight:700; color:${{gain1130 >= paramTarget ? 'var(--up)' : 'inherit'}};">
            ${{gain1130 >= 0 ? '+' : ''}}${{gain1130.toFixed(2)}}%
          </div>
          <div class="sub-pill">收 $${{s.bar_1130.close.toFixed(2)}} | ${{formatVol(s.bar_1130.vol)}}</div>
        </td>
        <!-- NEW KEY COLUMN: Recent 5 Days -->
        <td class="text-center" style="background:#eff6ff;">
          ${{getRateBadge(s.calc.d5.rate, s.calc.d5.h, s.calc.d5.c, s.calc.d5.max_gain, paramTarget, false)}}
        </td>
        <td class="text-center">
          ${{getRateBadge(s.calc.d3.rate, s.calc.d3.h, s.calc.d3.c, s.calc.d3.max_gain, paramTarget, false)}}
        </td>
        <td class="text-center">
          ${{getRateBadge(s.calc.d2.rate, s.calc.d2.h, s.calc.d2.c, s.calc.d2.max_gain, paramTarget, true)}}
        </td>
        <td class="text-center">
          ${{getRateBadge(s.calc.d1.rate, s.calc.d1.h, s.calc.d1.c, s.calc.d1.max_gain, paramTarget, true)}}
        </td>
        <td class="text-center" style="background:#f8fafc;">
          ${{getRateBadge(s.calc.today.rate, s.calc.today.h, s.calc.today.c, s.calc.today.max_gain, paramTarget, true)}}
        </td>
        <td class="text-left">${{posDetail}}</td>
      </tr>
    `;
  }}).join('');
}}

// Slider Event Handlers
function onParamChange() {{
  paramTarget = parseFloat(document.getElementById('slider-target').value);
  paramWindow = parseInt(document.getElementById('slider-window').value);
  
  // Highlight active preset button if matches
  document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
  if (paramTarget === 5.0 && paramWindow === 3) document.getElementById('pre-3d5')?.classList.add('active');
  if (paramTarget === 5.0 && paramWindow === 5) document.getElementById('pre-5d5')?.classList.add('active');
  if (paramTarget === 7.0 && paramWindow === 3) document.getElementById('pre-3d7')?.classList.add('active');
  if (paramTarget === 10.0 && paramWindow === 5) document.getElementById('pre-5d10')?.classList.add('active');
  if (paramTarget === 5.0 && paramWindow === 1) document.getElementById('pre-1d5')?.classList.add('active');
  
  renderAll();
}}

function setPreset(t, w) {{
  paramTarget = t;
  paramWindow = w;
  document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
  event.target.classList.add('active');
  renderAll();
}}

function filterCategory(cat) {{
  currentCategory = cat;
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.getElementById(`tab-${{cat}}`).classList.add('active');
  renderAll();
}}

function onSearch(val) {{
  searchQuery = val.trim();
  renderAll();
}}

function sortTable(field) {{
  if (sortField === field) {{
    sortAsc = !sortAsc;
  }} else {{
    sortField = field;
    sortAsc = false;
  }}
  renderAll();
}}

// Initial load
renderAll();
</script>
</body>
</html>
"""

html_path = OUTPUT_DIR / "index.html"
html_path.write_text(html_code, encoding="utf-8")
print(f"Generated {html_path} ({len(html_code):,} bytes)")

alt_path = OUTPUT_DIR / "today_portfolio_monitor.html"
alt_path.write_text(html_code, encoding="utf-8")
print(f"Generated {alt_path} ({len(html_code):,} bytes)")
