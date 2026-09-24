#!/usr/bin/env python3
"""Fetch Realtime & Intraday 60m Data from Futu OpenD for Holdings & Favorites,
and Generate Interactive Multi-Column Monitor Table.

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

# 1. Fetch Real User Positions
positions = {}
try:
    trd_ctx = ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US, host="127.0.0.1", port=11111)
    ret, acc_df = trd_ctx.get_acc_list()
    if ret == ft.RET_OK:
        for acc_id in acc_df["acc_id"]:
            r_pos, pos_df = trd_ctx.position_list_query(acc_id=acc_id)
            if r_pos == ft.RET_OK and len(pos_df) > 0:
                for _, row in pos_df.iterrows():
                    c = row["code"]
                    if c.startswith("US."):
                        positions[c] = {
                            "name": row.get("stock_name", ""),
                            "qty": float(row.get("qty", 0)),
                            "cost": float(row.get("cost_price", 0)),
                            "nominal_price": float(row.get("nominal_price", 0)),
                            "pnl_ratio": float(row.get("pl_ratio", 0)) if "pl_ratio" in row else 0.0,
                            "pnl_val": float(row.get("pl_val", 0)) if "pl_val" in row else 0.0
                        }
    trd_ctx.close()
    print(f"Loaded {len(positions)} US trade holdings: {list(positions.keys())}")
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

# 4. Fetch 60m K-lines for past week (2026-09-18 to 2026-09-24)
kline_data = {}
print("Fetching 60m K-lines for all target stocks...")
for code in all_target_codes:
    ret_kl, df, _ = quote_ctx.request_history_kline(
        code,
        start="2026-09-18",
        end="2026-09-24",
        ktype=ft.KLType.K_60M,
        autype=ft.AuType.QFQ,
        max_count=100
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
    time.sleep(0.05)

quote_ctx.close()
print("K-line fetching complete.")

# 5. Process Metrics for Each Stock
stocks_data = []

# Target trading dates
DAYS = [
    ("d_minus_3", "2026-09-21", "D-3 (09-21 周一)"),
    ("d_minus_2", "2026-09-22", "D-2 (09-22 周二)"),
    ("d_minus_1", "2026-09-23", "D-1 (09-23 周三)"),
    ("today",     "2026-09-24", "今天 (09-24 周四)")
]

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
    is_fav = code in fav_codes
    is_bad = code in bad_codes
    
    if is_holding and is_fav:
        cat = "持仓+特注"
    elif is_holding:
        cat = "我的持仓"
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
    
    # Calculate 3-day 5% achievement rates for D-3, D-2, D-1, Today
    day_metrics = {}
    
    for key, date_str, label in DAYS:
        if bars.empty:
            day_metrics[key] = {"c": 0, "h": 0, "rate": None, "max_gain": 0.0, "is_inflight": False, "hours": []}
            continue
            
        d_bars = bars[bars["date"] == date_str]
        c = len(d_bars)
        if c == 0:
            day_metrics[key] = {"c": 0, "h": 0, "rate": None, "max_gain": 0.0, "is_inflight": False, "hours": []}
            continue
            
        h_hits = 0
        max_runup = -999.0
        hour_records = []
        
        for idx_row, row in d_bars.iterrows():
            entry_c = float(row["close"])
            bar_idx = row.name
            future = bars.iloc[bar_idx + 1 : bar_idx + 22]
            
            if len(future) == 0:
                # If no future 60m bar yet, use current high price from snapshot
                future_max_h = max(float(row["high"]), high_price)
            else:
                future_max_h = max(future["high"].max(), high_price)
                
            runup = (future_max_h - entry_c) / entry_c * 100.0
            if runup > max_runup:
                max_runup = runup
                
            is_hit = bool(runup >= 5.0)
            if is_hit:
                h_hits += 1
                
            hour_records.append({
                "time": row["time"],
                "entry": entry_c,
                "max_h": future_max_h,
                "gain": round(runup, 2),
                "is_hit": is_hit
            })
            
        rate = round(h_hits / c * 100, 1) if c > 0 else 0.0
        # In-flight if date is 09-22, 09-23, or 09-24 (less than 21 bars closed)
        is_inflight = bool(date_str >= "2026-09-22")
        
        day_metrics[key] = {
            "c": int(c),
            "h": int(h_hits),
            "rate": rate,
            "max_gain": round(max_runup, 2) if max_runup != -999.0 else 0.0,
            "is_inflight": is_inflight,
            "hours": hour_records
        }
        
    stocks_data.append({
        "code": code,
        "ticker": ticker,
        "name": stock_name,
        "category": cat,
        "is_holding": is_holding,
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
            "max_gain": gain_from_1030,
            "is_hit": bool(gain_from_1030 >= 5.0)
        },
        "bar_1130": {
            "close": close_1130,
            "vol": vol_1130,
            "max_gain": gain_from_1130,
            "is_hit": bool(gain_from_1130 >= 5.0)
        },
        "completed_vol": completed_vol,
        "vol_share_pct": vol_share_pct,
        "metrics": day_metrics
    })

# Sort stocks: Holdings first, then by today's change % descending
stocks_data.sort(key=lambda s: (0 if s["is_holding"] else 1, -s["chg_pct"]))

payload = {
    "update_time_et": datetime.now().strftime("%Y-%m-%d %H:%M:%S ET"),
    "update_time_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "stocks_count": len(stocks_data),
    "holdings_count": len(positions),
    "days_meta": DAYS,
    "stocks": stocks_data
}

# Save JSON
json_path = OUTPUT_DIR / "today_data.json"
json_dump = json.dumps(payload, ensure_ascii=False, indent=2)
json_path.write_text(json_dump, encoding="utf-8")
print(f"Saved {json_path} ({len(json_dump):,} bytes)")

# 6. Render Wireframe HTML
html_code = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>美股盘中实时异动与 3日5% 达成率监控看板（持仓 + 特别关注）</title>
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
  --hold-tag: #0f172a;
  --hold-bg: #e2e8f0;
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
  margin-bottom: 14px;
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

/* Summary Metric Cards */
.summary-cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
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
tr.row-holding {{
  background: #fcfcfc;
}}
tr.row-holding td:first-child {{
  border-left: 3px solid #0f172a;
}}

/* Badges & Tags */
.tag-badge {{
  font-size: 10px;
  font-weight: 700;
  padding: 2px 6px;
  border-radius: 2px;
  display: inline-block;
}}
.tag-holding {{ background: #e2e8f0; color: #0f172a; border: 1px solid #cbd5e1; }}
.tag-fav {{ background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }}
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
    <h1>美股持仓与特别关注标的 · 盘中实时异动 & 3日5% 达成率监控</h1>
    <p>实时读取富途 OpenD 真实账户持仓与自选分组 | 覆盖开盘至 -1 小时（10:30、11:30）走势与过去 3 日达成率闭环</p>
  </div>
  <div class="sync-badge">
    <div class="pulse-dot"></div>
    <span>行情更新：<b>{payload["update_time_et"]}</b>（美东时间）</span>
  </div>
</header>

<!-- Summary Metric Cards -->
<section class="summary-cards" id="summary-cards">
  <div class="card">
    <span>监控目标总数</span>
    <b>{payload["stocks_count"]} 只</b>
  </div>
  <div class="card">
    <span>当前账户持仓</span>
    <b>{payload["holdings_count"]} 只</b>
  </div>
  <div class="card">
    <span>今日上涨标的数</span>
    <b style="color:var(--up);" id="stat-up-count">-</b>
  </div>
  <div class="card">
    <span>D-3 (09-21) 均值达成率</span>
    <b id="stat-d3-avg">-</b>
  </div>
  <div class="card">
    <span>开盘首小时 (10:30) 均值达成率</span>
    <b id="stat-h1030-avg">-</b>
  </div>
</section>

<!-- Strategy Notes -->
<details class="notes-panel">
  <summary>💡 本页面策略思路与字段说明（点击展开查看）</summary>
  <div class="notes-content">
    <p><b>1. 为什么重点监控“开盘到现在的 -1 小时”？</b><br>
    根据我们在聚类方法 1 与方法 2 中的重大实证发现：美股单日开盘首小时（10:30）与次小时（11:30）的群体达成率，对全天后续时段的胜率具有高达 <b>r = 0.788 ~ 0.886</b> 的决定性预测力。开盘两小时放量抗跌且达成率高（&ge;50%）的标的，往往全天动能强劲；反之首小时砸盘的标的，全天绝不轻易追入。</p>
    <p style="margin-top:6px;"><b>2. 四大核心达成率列定义（以 3 日内最高价涨 &ge; 5% 为目标）：</b><br>
    • <b>今天-3（09-21 周一）</b>：分母为当天 7 个交易小时（10:30~16:00），分子为后续 3 天内触达 +5% 的小时数。历经 3 天交易后，该列已近乎 100% 闭环，体现该股初期的突破兑现能力；<br>
    • <b>今天-2（09-22 周二）</b>与 <b>今天-1（09-23 周三）</b>：尚在 3 日窗口内（打 ⏳ 标识）。若当前已有小时触及 +5% 则提前确认为达成，并展示最大浮盈；<br>
    • <b>今天（09-24 周四）</b>：分母为今天已完成的时段数（10:30 与 11:30 共 2 个时段），分子为截至目前已触达 +5% 的时段数，并展示入场至今的盘中最高冲幅（Max Run-up %）。</p>
  </div>
</details>

<!-- Controls Bar -->
<section class="controls-bar">
  <div class="filter-tabs">
    <button class="tab-btn active" id="tab-all" onclick="filterCategory('all')">全部标的 ({payload["stocks_count"]})</button>
    <button class="tab-btn" id="tab-holding" onclick="filterCategory('holding')">我的持仓 ({payload["holdings_count"]})</button>
    <button class="tab-btn" id="tab-fav" onclick="filterCategory('fav')">特别关注 ({len(fav_codes)})</button>
    <button class="tab-btn" id="tab-up" onclick="filterCategory('up')">今日上涨</button>
    <button class="tab-btn" id="tab-hit" onclick="filterCategory('hit')">近期高胜率 (&ge;50%)</button>
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
        <th onclick="sortTable('completed_vol')">开盘-1h成交量 (10:30+11:30)</th>
        <th onclick="sortTable('gain_1030')">首小时 (10:30) 冲幅</th>
        <th onclick="sortTable('gain_1130')">次小时 (11:30) 冲幅</th>
        <th class="text-center" onclick="sortTable('rate_d3')" style="background:#e2e8f0; color:#0f172a;">今天-3 (09-21)<br>3日5% 达成率</th>
        <th class="text-center" onclick="sortTable('rate_d2')" style="background:#e2e8f0; color:#0f172a;">今天-2 (09-22)<br>3日5% 达成率</th>
        <th class="text-center" onclick="sortTable('rate_d1')" style="background:#e2e8f0; color:#0f172a;">今天-1 (09-23)<br>3日5% 达成率</th>
        <th class="text-center" onclick="sortTable('rate_today')" style="background:#cbd5e1; color:#0f172a; font-weight:800;">今天 (09-24)<br>3日5% 达成率</th>
        <th class="text-left">持仓详情 (若有)</th>
      </tr>
    </thead>
    <tbody id="table-body">
      <!-- Injected via JavaScript -->
    </tbody>
  </table>
</main>

<footer class="wf-footer">
  <span>数据来源：本地富途牛牛 OpenD (端口 11111) | 生成时间：{payload["update_time_local"]}</span>
  <span>归档目录：analysis/today_intraday_watch/</span>
</footer>

<script>
const RAW_DATA = {json.dumps(payload["stocks"], ensure_ascii=False)};
let currentCategory = 'all';
let searchQuery = '';
let sortField = 'chg_pct';
let sortAsc = false;

function formatVol(v) {{
  if (!v) return '-';
  if (v >= 1e6) return (v / 1e6).toFixed(2) + ' M';
  if (v >= 1e3) return (v / 1e3).toFixed(1) + ' K';
  return v.toLocaleString();
}}

function getRateBadge(rate, h, c, maxGain, isToday, isInflight) {{
  if (rate === null || c === 0) return '<span style="color:#94a3b8;">-</span>';
  let cls = 'rate-mid';
  if (rate >= 50) cls = 'rate-high';
  else if (rate < 25) cls = 'rate-low';
  
  let flightIcon = isInflight ? ' ⏳' : '';
  let hitBadge = maxGain >= 5.0 ? `<span class="runup-badge hit">+${{maxGain.toFixed(1)}}%✓</span>` : `<span class="runup-badge">+${{maxGain.toFixed(1)}}%</span>`;
  
  return `
    <div>
      <span class="cell-rate ${{cls}}">${{rate.toFixed(1)}}%</span>
      <span class="sub-pill">(${{h}}/${{c}})${{flightIcon}} ${{hitBadge}}</span>
    </div>
  `;
}}

function renderTable() {{
  const tbody = document.getElementById('table-body');
  
  // Filter
  let list = RAW_DATA.filter(s => {{
    if (currentCategory === 'holding' && !s.is_holding) return false;
    if (currentCategory === 'fav' && !s.category.includes('特别关注') && !s.category.includes('特注')) return false;
    if (currentCategory === 'up' && s.chg_pct <= 0) return false;
    if (currentCategory === 'hit') {{
      const d3Rate = s.metrics.d_minus_3.rate || 0;
      const d2Rate = s.metrics.d_minus_2.rate || 0;
      if (d3Rate < 50 && d2Rate < 50) return false;
    }}
    if (searchQuery) {{
      const q = searchQuery.toLowerCase();
      if (!s.ticker.toLowerCase().includes(q) && !s.name.toLowerCase().includes(q)) return false;
    }}
    return true;
  }});
  
  // Sort
  list.sort((a, b) => {{
    let va = a[sortField];
    let vb = b[sortField];
    if (sortField === 'rate_d3') va = a.metrics.d_minus_3.rate ?? -1, vb = b.metrics.d_minus_3.rate ?? -1;
    if (sortField === 'rate_d2') va = a.metrics.d_minus_2.rate ?? -1, vb = b.metrics.d_minus_2.rate ?? -1;
    if (sortField === 'rate_d1') va = a.metrics.d_minus_1.rate ?? -1, vb = b.metrics.d_minus_1.rate ?? -1;
    if (sortField === 'rate_today') va = a.metrics.today.rate ?? -1, vb = b.metrics.today.rate ?? -1;
    if (sortField === 'gain_1030') va = a.bar_1030.max_gain, vb = b.bar_1030.max_gain;
    if (sortField === 'gain_1130') va = a.bar_1130.max_gain, vb = b.bar_1130.max_gain;
    
    if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return sortAsc ? (va - vb) : (vb - va);
  }});
  
  // Update stats
  const upCount = list.filter(s => s.chg_pct > 0).length;
  document.getElementById('stat-up-count').innerText = `${{upCount}} / ${{list.length}} 只 (${{(upCount/list.length*100).toFixed(0)}}%)`;
  
  const d3Valid = list.filter(s => s.metrics.d_minus_3.rate !== null);
  const d3Avg = d3Valid.length > 0 ? (d3Valid.reduce((acc, s) => acc + s.metrics.d_minus_3.rate, 0) / d3Valid.length).toFixed(1) : '-';
  document.getElementById('stat-d3-avg').innerText = d3Avg + '%';
  
  const h1030Hits = list.filter(s => s.bar_1030.is_hit).length;
  document.getElementById('stat-h1030-avg').innerText = `${{(h1030Hits / list.length * 100).toFixed(1)}}% (${{h1030Hits}}只达标)`;
  
  tbody.innerHTML = list.map(s => {{
    const chgClass = s.chg_pct > 0 ? 'chg-up' : (s.chg_pct < 0 ? 'chg-down' : 'chg-flat');
    const chgSign = s.chg_pct > 0 ? '+' : '';
    
    let tagCls = 'tag-fav';
    if (s.is_holding) tagCls = 'tag-holding';
    else if (s.category.includes('亏损')) tagCls = 'tag-bad';
    
    const rowCls = s.is_holding ? 'row-holding' : '';
    
    let posDetail = '<span style="color:#94a3b8;">-</span>';
    if (s.is_holding) {{
      const p = s.pos_info;
      posDetail = `<span style="font-family:monospace; font-size:11px;">持仓: <b>${{p.qty}}</b>股 | 成本: $${{p.cost?.toFixed(2)}}</span>`;
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
          <div class="sub-pill">占今日全天 ${{s.vol_share_pct}}%</div>
        </td>
        <td>
          <div style="font-family:monospace; font-weight:700; color:${{gain1030 >= 5 ? 'var(--up)' : 'inherit'}};">
            ${{gain1030 >= 0 ? '+' : ''}}${{gain1030.toFixed(2)}}%
          </div>
          <div class="sub-pill">收 $${{s.bar_1030.close.toFixed(2)}} | ${{formatVol(s.bar_1030.vol)}}</div>
        </td>
        <td>
          <div style="font-family:monospace; font-weight:700; color:${{gain1130 >= 5 ? 'var(--up)' : 'inherit'}};">
            ${{gain1130 >= 0 ? '+' : ''}}${{gain1130.toFixed(2)}}%
          </div>
          <div class="sub-pill">收 $${{s.bar_1130.close.toFixed(2)}} | ${{formatVol(s.bar_1130.vol)}}</div>
        </td>
        <td class="text-center">
          ${{getRateBadge(s.metrics.d_minus_3.rate, s.metrics.d_minus_3.h, s.metrics.d_minus_3.c, s.metrics.d_minus_3.max_gain, false, s.metrics.d_minus_3.is_inflight)}}
        </td>
        <td class="text-center">
          ${{getRateBadge(s.metrics.d_minus_2.rate, s.metrics.d_minus_2.h, s.metrics.d_minus_2.c, s.metrics.d_minus_2.max_gain, false, s.metrics.d_minus_2.is_inflight)}}
        </td>
        <td class="text-center">
          ${{getRateBadge(s.metrics.d_minus_1.rate, s.metrics.d_minus_1.h, s.metrics.d_minus_1.c, s.metrics.d_minus_1.max_gain, false, s.metrics.d_minus_1.is_inflight)}}
        </td>
        <td class="text-center" style="background:#f8fafc;">
          ${{getRateBadge(s.metrics.today.rate, s.metrics.today.h, s.metrics.today.c, s.metrics.today.max_gain, true, s.metrics.today.is_inflight)}}
        </td>
        <td class="text-left">${{posDetail}}</td>
      </tr>
    `;
  }}).join('');
}}

function filterCategory(cat) {{
  currentCategory = cat;
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.getElementById(`tab-${{cat}}`).classList.add('active');
  renderTable();
}}

function onSearch(val) {{
  searchQuery = val.trim();
  renderTable();
}}

function sortTable(field) {{
  if (sortField === field) {{
    sortAsc = !sortAsc;
  }} else {{
    sortField = field;
    sortAsc = false;
  }}
  renderTable();
}}

// Initialize
renderTable();
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
