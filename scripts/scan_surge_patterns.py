#!/usr/bin/env python3
"""Scan Universe for 4 Upward Momentum Patterns and Run 3-Week Per-Stock Historical Backtest.

Target Universe:
- 自选 (User Favorites from Futu OpenD)
- 持仓 (Moomoo US 0086 Cash & Margin Holdings)
- 自选 ETF (SOXX, SMH, SOXL, SOXS, IGV, SPCX, QQQ, SPY, DIA, XLU)
- ETF 成分股 (Semiconductor, Optical, Cloud Infrastructure, etc.)
- QQQ 104 Constituents (Total: 138 Tickers)

Detection Logic:
- Window: 2-hour rolling window (N = 24 bars of 5m regular session)
- Surge threshold: M >= +5.0%
- 4 Upward Micro-Patterns:
    1. ▲ 单边拉升: minRet >= -0.5% and maxPos >= 75%
    2. V型深弹: minPos <= 45% and minRet <= -1.5%
    3. 冲高回落: maxPos <= 55% and endRet < maxRet - 1.5%
    4. 阶梯中继: Other continuous step continuations

Per-Stock 3-Week Historical Backtest:
- For each detected (symbol, pattern):
  Scans all occurrences of this exact pattern on this stock in the past 3 weeks (15 trading days = 1,170 bars).
  Computes:
    - 3-Week Total Occurrence Count (N)
    - 3-Day +5% Reach Rate (hit_3d_5 / total)
    - 5-Day +5% Reach Rate (hit_5d_5 / total)
    - Average Maximum Gain in 3 Days (%)
    - Average Maximum Gain in 5 Days (%)

Outputs:
- Terminal formatted table with ANSI colors
- analysis/latest_surge_scan/scan_results.json
- analysis/latest_surge_scan/index.html (Port 8768 ready)

Usage:
    # 1. Quick scan on existing local 5m data
    python3 scripts/scan_surge_patterns.py

    # 2. Auto-sync latest bars from Futu OpenD first, then scan
    python3 scripts/scan_surge_patterns.py --sync

    # 3. Scan specific symbols
    python3 scripts/scan_surge_patterns.py --symbols CRDO NVDA AAPL SOXL
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

M5_DIR = ROOT / "market_data" / "us_5m"
QQQ_FILE = ROOT / "analysis" / "qqq_constituents.json"
OUT_DIR = ROOT / "analysis" / "latest_surge_scan"
JSON_OUT = OUT_DIR / "scan_results.json"
HTML_OUT = OUT_DIR / "index.html"

# User favorites from Futu OpenD
FAVORITES_TICKERS = {
    'WDC', 'STX', 'TER', 'PLTR', 'NBIS', 'AVGO', 'AAOZ', 'CRWV', 'AMAT', 'LITX',
    'CBRS', 'RMBS', 'IONL', 'LRNZ', 'AAOI', 'VRT', 'AIPO', 'CIEN', 'SOXS', 'MU',
    'CRDO', 'AMD', 'SMTC', 'ARM', 'TXG', 'TWST', 'SDGR', 'QCOM', 'COHR', 'ALAB',
    'LITE', 'MRVL', 'LIFE', 'NOK', 'KOD'
}

# User 0086 holdings from Futu OpenD
HOLDINGS_0086 = {
    'SNXX', 'RKLB', 'NXT', 'NVTS', 'INTC', 'GOOG', 'FN', 'COHR', 'AVGO', 'AMZN',
    'AIPO', 'CRDO', 'MRVL', 'AMAT', 'HLTH', 'CIEN', 'VRT', 'LITX', 'SEDG', 'BBC', 'AXTI'
}

# Core ETFs
CORE_ETFS = {
    'QQQ', 'SPY', 'DIA', 'SOXX', 'SMH', 'SOXL', 'SOXS', 'IGV', 'SPCX', 'XLU'
}

# Semiconductor / AI Hardware ecosystem
SEMI_HARDWARE = {
    'NVDA', 'AVGO', 'AMD', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'MU', 'INTC', 'ASML',
    'KLAC', 'MRVL', 'ADI', 'NXPI', 'MCHP', 'MPWR', 'TER', 'RMBS', 'SMTC', 'ALAB',
    'SNDK', 'COHR', 'LITE', 'CRDO', 'VRT', 'NVTS', 'AXTI', 'AAOI', 'AAOZ', 'CBRS',
    'WDC', 'STX', 'SOXX', 'SMH', 'SOXL', 'SOXS', 'FN', 'NXT'
}

# 4 Upward Patterns
UPWARD_PATTERNS = ['▲ 单边拉升', 'V型深弹', '冲高回落', '阶梯中继']

PATTERN_CONFIG = {
    '▲ 单边拉升': {'color': '#10b981', 'role': '多头强攻', 'badge': '拉升'},
    'V型深弹':   {'color': '#06b6d4', 'role': '极致反抽', 'badge': '深弹'},
    '冲高回落': {'color': '#f59e0b', 'role': '高位试盘', 'badge': '试盘'},
    '阶梯中继': {'color': '#84cc16', 'role': '蓄势中继', 'badge': '中继'}
}


def get_default_universe() -> List[str]:
    """Retrieve all 138 tickers."""
    tickers: Set[str] = set()
    if QQQ_FILE.exists():
        try:
            tickers.update(json.loads(QQQ_FILE.read_text())['tickers'])
        except Exception:
            pass
    tickers.update(FAVORITES_TICKERS)
    tickers.update(HOLDINGS_0086)
    tickers.update(CORE_ETFS)
    tickers.update(SEMI_HARDWARE)
    return sorted(tickers)


def classify_pattern(sub_closes: np.ndarray, p0: float, end_ret: float) -> str:
    """Classify 2h window into one of 4 upward patterns."""
    cnt = len(sub_closes)
    sub_r = (sub_closes - p0) / p0
    min_idx = int(np.argmin(sub_r))
    max_idx = int(np.argmax(sub_r))
    min_ret = sub_r[min_idx]
    max_ret = sub_r[max_idx]
    min_pos = min_idx / max(1, cnt - 1)
    max_pos = max_idx / max(1, cnt - 1)

    if min_ret >= -0.005 and max_pos >= 0.75:
        return '▲ 单边拉升'
    elif min_pos <= 0.45 and min_ret <= -0.015:
        return 'V型深弹'
    elif max_pos <= 0.55 and end_ret < max_ret - 0.015:
        return '冲高回落'
    else:
        return '阶梯中继'


def backtest_stock_pattern_3w(
    closes: np.ndarray,
    highs: np.ndarray,
    time_keys: List[str],
    target_pattern: str,
    N: int = 24,
    M: float = 0.05,
    bt_bars: int = 1170
) -> Dict[str, Any]:
    """Run per-stock 3-week backtest for the specified upward pattern."""
    tot = len(closes)
    bt_start = max(N, tot - bt_bars)

    rets = np.full(tot, np.nan)
    rets[N:] = (closes[N:] - closes[:-N]) / closes[:-N]

    # Detect all historical episodes in past 3 weeks
    in_s, s_st = False, 0
    episodes = []
    for i in range(bt_start, tot):
        r = rets[i]
        if r >= M:
            if not in_s:
                in_s, s_st = True, i
        else:
            if in_s:
                episodes.append({'st': s_st - N, 'end': i - 1})
                in_s = False
    if in_s:
        episodes.append({'st': s_st - N, 'end': tot - 1})

    hit_3d_5 = 0
    hit_5d_5 = 0
    total_matched = 0
    max_gains_3d = []
    max_gains_5d = []
    historical_cases = []

    for ep in episodes:
        st_i = ep['st']
        end_i = ep['end']
        c_sub = closes[st_i: end_i + 1]
        p0 = closes[st_i]
        p_entry = closes[end_i]
        end_ret = (p_entry - p0) / p0

        pat = classify_pattern(c_sub, p0, end_ret)
        if pat == target_pattern:
            total_matched += 1

            # 3d +5% (234 bars)
            max_fwd_3 = min(tot - 1, end_i + 234)
            fwd_high_3 = np.max(highs[end_i + 1: max_fwd_3 + 1]) if end_i + 1 <= tot - 1 else p_entry
            hit_3 = bool(fwd_high_3 >= p_entry * 1.05)
            gain_3 = (fwd_high_3 - p_entry) / p_entry * 100
            if hit_3: hit_3d_5 += 1
            max_gains_3d.append(gain_3)

            # 5d +5% (390 bars)
            max_fwd_5 = min(tot - 1, end_i + 390)
            fwd_high_5 = np.max(highs[end_i + 1: max_fwd_5 + 1]) if end_i + 1 <= tot - 1 else p_entry
            hit_5 = bool(fwd_high_5 >= p_entry * 1.05)
            gain_5 = (fwd_high_5 - p_entry) / p_entry * 100
            if hit_5: hit_5d_5 += 1
            max_gains_5d.append(gain_5)

            historical_cases.append({
                'trigger_time': time_keys[end_i],
                'entry_price': round(float(p_entry), 2),
                'window_ret': round(float(end_ret * 100), 2),
                'hit_3d_5': hit_3,
                'hit_5d_5': hit_5,
                'max_gain_3d': round(float(gain_3), 2),
                'max_gain_5d': round(float(gain_5), 2)
            })

    rate_3d_5 = round(hit_3d_5 / total_matched * 100, 1) if total_matched > 0 else 0.0
    rate_5d_5 = round(hit_5d_5 / total_matched * 100, 1) if total_matched > 0 else 0.0
    avg_gain_3d = round(float(np.mean(max_gains_3d)), 2) if max_gains_3d else 0.0
    avg_gain_5d = round(float(np.mean(max_gains_5d)), 2) if max_gains_5d else 0.0

    return {
        'total_3w_events': total_matched,
        'hit_3d_5': hit_3d_5,
        'rate_3d_5': rate_3d_5,
        'hit_5d_5': hit_5d_5,
        'rate_5d_5': rate_5d_5,
        'avg_gain_3d': avg_gain_3d,
        'avg_gain_5d': avg_gain_5d,
        'cases': historical_cases
    }


def scan_universe(
    symbols: List[str],
    lookback_session_bars: int = 78,
    N: int = 24,
    M: float = 0.05
) -> List[Dict[str, Any]]:
    """Scan all target symbols for active/recent upward patterns in the latest trading session."""
    results = []

    for s in symbols:
        p = M5_DIR / s / "2026.parquet"
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            time_str = df['time_key'].astype(str)
            dates = time_str.str[:10]
            times = time_str.str[11:16]

            # Regular trading session bars (09:35 to 16:00 ET)
            reg = df[(dates >= '2026-07-01') & (times >= '09:35') & (times <= '16:00')].sort_values('time_key').reset_index(drop=True)
            if len(reg) < 70:
                continue

            closes = reg['close'].to_numpy(dtype=np.float64)
            highs = reg['high'].to_numpy(dtype=np.float64)
            time_keys = reg['time_key'].tolist()
            tot = len(closes)

            # Rolling return
            rets = np.full(tot, np.nan)
            rets[N:] = (closes[N:] - closes[:-N]) / closes[:-N]

            # Check the recent session bars
            check_len = min(tot, lookback_session_bars)
            recent_surge_indices = [i for i in range(tot - check_len, tot) if rets[i] >= M]

            if not recent_surge_indices:
                continue

            # Pick the most recent surge point
            last_i = recent_surge_indices[-1]
            st_i = last_i - N
            end_i = last_i

            c_sub = closes[st_i: end_i + 1]
            p0 = closes[st_i]
            p_cur = closes[end_i]
            cur_ret = float(rets[last_i])

            pattern = classify_pattern(c_sub, p0, cur_ret)

            # Run 3-week per-stock historical backtest
            bt_stats = backtest_stock_pattern_3w(
                closes=closes,
                highs=highs,
                time_keys=time_keys,
                target_pattern=pattern,
                N=N,
                M=M,
                bt_bars=1170 # 15 trading days
            )

            # Tags
            is_held = (s in HOLDINGS_0086)
            is_fav = (s in FAVORITES_TICKERS)
            is_etf = (s in CORE_ETFS)
            is_semi = (s in SEMI_HARDWARE)

            results.append({
                'symbol': s,
                'pattern': pattern,
                'trigger_time': time_keys[end_i],
                'window_start_time': time_keys[st_i],
                'p_start': round(float(p0), 2),
                'p_entry': round(float(p_cur), 2),
                'surge_2h_pct': round(cur_ret * 100, 2),
                'is_held': is_held,
                'is_fav': is_fav,
                'is_etf': is_etf,
                'is_semi': is_semi,
                'backtest_3w': bt_stats
            })

        except Exception as e:
            print(f"Error processing {s}: {e}", file=sys.stderr)

    # Sort results: highest 3d+5% rate, then sample count, then surge pct
    results.sort(
        key=lambda x: (
            x['backtest_3w']['rate_3d_5'],
            x['backtest_3w']['total_3w_events'],
            x['surge_2h_pct']
        ),
        reverse=True
    )
    return results


def print_cli_table(results: List[Dict[str, Any]], scan_time: str):
    """Print beautifully formatted ANSI terminal table."""
    # Terminal ANSI codes
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    print("\n" + "=" * 115)
    print(f"{BOLD}🎯 全域自选/持仓/ETF 4 大上涨形态实时扫描与 3 周历史胜率回测看板{RESET}")
    print(f"{DIM}执行时间: {scan_time} · 标的池: 138 只 · 窗口: 2 小时 (24根 5m) · 上涨门槛: ≥ +5.0%{RESET}")
    print("=" * 115)

    if not results:
        print(f"\n{YELLOW}⚠️ 当前最新交易时段内暂无标的触发 2小时 ≥ +5.0% 上涨形态。{RESET}\n")
        return

    header = f"{'#':<3} {'标的代码':<8} {'命中上涨形态':<12} {'当前2h涨幅':<11} {'触发时点(ET)':<18} {'3周样本':<8} {'3天+5%胜率 ★':<14} {'5天+5%胜率':<13} {'3日均涨':<9} {'标的属性'}"
    print(f"{BOLD}{header}{RESET}")
    print("-" * 115)

    for idx, r in enumerate(results, 1):
        sym = r['symbol']
        pat = r['pattern']
        p_cfg = PATTERN_CONFIG.get(pat, {'color': '', 'role': ''})
        p_ret = f"+{r['surge_2h_pct']:.2f}%"

        bt = r['backtest_3w']
        n_sample = f"{bt['total_3w_events']} 笔"

        # Rate colors
        r3 = bt['rate_3d_5']
        r3_str = f"{r3:5.1f}% ({bt['hit_3d_5']}/{bt['total_3w_events']})"
        if r3 >= 75.0 and bt['total_3w_events'] >= 3:
            r3_colored = f"{GREEN}{BOLD}{r3_str}{RESET}"
        elif r3 >= 50.0:
            r3_colored = f"{CYAN}{r3_str}{RESET}"
        else:
            r3_colored = f"{DIM}{r3_str}{RESET}"

        r5 = bt['rate_5d_5']
        r5_str = f"{r5:5.1f}% ({bt['hit_5d_5']}/{bt['total_3w_events']})"
        r5_colored = f"{GREEN}{r5_str}{RESET}" if r5 >= 75.0 else f"{r5_str}"

        gain3 = f"+{bt['avg_gain_3d']:.2f}%"

        # Tags
        tags = []
        if r['is_held']: tags.append(f"{RED}0086持仓{RESET}")
        if r['is_fav']: tags.append(f"{YELLOW}自选关注{RESET}")
        if r['is_etf']: tags.append(f"{CYAN}核心ETF{RESET}")
        if r['is_semi']: tags.append(f"{MAGENTA}半导体{RESET}")
        tags_str = " ".join(tags) if tags else "成分股"

        pat_colored = f"{BOLD}{pat}{RESET}"

        line = f"{idx:<3} {BOLD}{sym:<8}{RESET} {pat_colored:<18} {p_ret:<11} {r['trigger_time']:<18} {n_sample:<8} {r3_colored:<23} {r5_colored:<13} {gain3:<9} {tags_str}"
        print(line)

    print("-" * 115)
    print(f"📊 汇总: 共 {len(results)} 只标的在最新交易日内检测到 4 大上涨形态。")
    print(f"💡 说明: ★ [3天+5%胜率] 为历史 3 周内该股票在该形态触发后的达标概率，标绿加粗为胜率 ≥ 75% 且样本 ≥ 3 笔的高信度黄金标的。")
    print("=" * 115 + "\n")


def generate_html_report(results: List[Dict[str, Any]], scan_time: str):
    """Generate interactive self-contained HTML report."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows_html = []
    for i, r in enumerate(results, 1):
        pat_color = PATTERN_CONFIG.get(r['pattern'], {}).get('color', '#fff')
        bt = r['backtest_3w']
        rate_3d_color = '#10b981' if bt['rate_3d_5'] >= 75 and bt['total_3w_events'] >= 3 else ('#38bdf8' if bt['rate_3d_5'] >= 50 else '#94a3b8')
        rate_5d_color = '#10b981' if bt['rate_5d_5'] >= 75 else '#f1f5f9'
        tags_html = ""
        if r['is_held']: tags_html += '<span class="tag tag-held">0086持仓</span>'
        if r['is_fav']: tags_html += '<span class="tag tag-fav">自选关注</span>'
        if r['is_etf']: tags_html += '<span class="tag tag-etf">核心ETF</span>'
        if not tags_html: tags_html = '<span style="color:#64748b;">成分股</span>'

        rows_html.append(f"""
        <tr>
          <td style="color:var(--text-dim); font-family:var(--font-mono);">{i}</td>
          <td style="font-weight:700; font-family:var(--font-mono); font-size:13px; color:#38bdf8;">{r['symbol']}</td>
          <td style="font-weight:700; color:{pat_color};">{r['pattern']}</td>
          <td style="font-family:var(--font-mono); font-weight:700; color:#10b981;">+{r['surge_2h_pct']:.2f}%</td>
          <td style="font-family:var(--font-mono); color:var(--text-muted);">{r['trigger_time']}</td>
          <td style="font-family:var(--font-mono);">${r['p_entry']:.2f}</td>
          <td style="font-family:var(--font-mono);">{bt['total_3w_events']} 笔</td>
          <td style="font-family:var(--font-mono); font-weight:700; color:{rate_3d_color};">
            {bt['rate_3d_5']:.1f}% ({bt['hit_3d_5']}/{bt['total_3w_events']})
          </td>
          <td style="font-family:var(--font-mono); color:{rate_5d_color};">
            {bt['rate_5d_5']:.1f}% ({bt['hit_5d_5']}/{bt['total_3w_events']})
          </td>
          <td style="font-family:var(--font-mono); color:#10b981;">+{bt['avg_gain_3d']:.2f}%</td>
          <td>{tags_html}</td>
        </tr>
        """)

    tbody_content = "".join(rows_html)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>4大上涨形态实时扫描与3周胜率回测 · 最新看板</title>
  <style>
    :root {{
      --bg-dark: #080d1a;
      --bg-card: #0f172a;
      --bg-toolbar: #131c31;
      --border-color: #1e293b;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --color-emerald: #10b981;
      --color-cyan: #06b6d4;
      --color-amber: #f59e0b;
      --color-rose: #e11d48;
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg-dark);
      color: var(--text-main);
      font-family: var(--font-sans);
      padding: 24px 32px;
      line-height: 1.5;
      font-size: 13px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border-color);
      margin-bottom: 20px;
    }}
    .title-area h1 {{
      font-size: 20px;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .badge {{
      background: rgba(16, 185, 129, 0.16);
      color: #10b981;
      border: 1px solid rgba(16, 185, 129, 0.35);
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-family: var(--font-mono);
      font-weight: 600;
    }}
    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 16px 20px;
      overflow-x: auto;
      margin-bottom: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th {{
      background: #111a2e;
      color: var(--text-muted);
      padding: 10px 12px;
      text-align: left;
      font-weight: 700;
      border-bottom: 2px solid var(--border-color);
      white-space: nowrap;
    }}
    td {{
      padding: 10px 12px;
      border-bottom: 1px solid #162033;
      white-space: nowrap;
    }}
    tr:hover td {{
      background: rgba(56, 189, 248, 0.05);
    }}
    .tag {{
      display: inline-block;
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 10px;
      font-family: var(--font-mono);
      margin-right: 4px;
    }}
    .tag-held {{ background: rgba(225, 29, 72, 0.2); color: #f43f5e; border: 1px solid rgba(225, 29, 72, 0.4); }}
    .tag-fav {{ background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.4); }}
    .tag-etf {{ background: rgba(6, 182, 212, 0.2); color: #06b6d4; border: 1px solid rgba(6, 182, 212, 0.4); }}
  </style>
</head>
<body>
  <div class="header">
    <div class="title-area">
      <h1>
        🎯 4 大上涨形态实时扫描与 3 周胜率回测看板
        <span class="badge">实时即查</span>
      </h1>
      <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">
        自选 + 0086 持仓 + 核心 ETF + 产业链龙头 (共 138 只) · 2小时滑动窗口 ≥ +5.0%
      </div>
    </div>
    <div style="text-align:right; font-family:var(--font-mono); color:var(--text-dim); font-size:12px;">
      生成时间: {scan_time}
    </div>
  </div>

  <div class="card">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>标的代码</th>
          <th>命中上涨形态</th>
          <th>当前2h涨幅</th>
          <th>触发时点 (ET)</th>
          <th>入场价格</th>
          <th>3周样本</th>
          <th>★ 3天+5% 历史胜率</th>
          <th>5天+5% 历史胜率</th>
          <th>3日最大均涨</th>
          <th>标的属性</th>
        </tr>
      </thead>
      <tbody>
        {tbody_content}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    with open(HTML_OUT, 'w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    parser = argparse.ArgumentParser(description="Scan 4 Upward Patterns and Run 3-Week Historical Backtest")
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to scan. If omitted, scans all 138 universe tickers.")
    parser.add_argument("--sync", action="store_true", help="Auto-sync latest 5m bars from Futu OpenD before scanning")
    parser.add_argument("--window", type=int, default=24, help="Window bars (default: 24 bars = 2h)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Surge threshold (default: 0.05 = +5.0%%)")
    parser.add_argument("--session-bars", type=int, default=78, help="Lookback session bars to check for recent triggers (default: 78 bars = 1 trading day)")

    args = parser.parse_args()

    # Step 1: Optional sync
    if args.sync:
        sync_cmd = [sys.executable, str(ROOT / "scripts" / "sync_us_5m_bars.py")]
        if args.symbols:
            sync_cmd.extend(["--symbols"] + args.symbols)
        else:
            sync_cmd.append("--all-universe")
        print("🔄 Executing 5m Market Data Sync before scanning...")
        subprocess.run(sync_cmd, check=True)

    # Step 2: Determine symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols]
    else:
        symbols = get_default_universe()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n🔍 Scanning {len(symbols)} tickers for 4 Upward Patterns (Window={args.window} bars, Threshold=+{args.threshold*100:.0f}%)...")

    # Step 3: Scan and backtest
    results = scan_universe(
        symbols=symbols,
        lookback_session_bars=args.session_bars,
        N=args.window,
        M=args.threshold
    )

    # Step 4: CLI output
    print_cli_table(results, now_str)

    # Step 5: Save JSON and HTML
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, 'w', encoding='utf-8') as f:
        json.dump({
            'scan_time': now_str,
            'symbols_count': len(symbols),
            'detected_count': len(results),
            'results': results
        }, f, ensure_ascii=False, indent=2)

    generate_html_report(results, now_str)
    print(f"📁 Results saved to:")
    print(f"   JSON: {JSON_OUT}")
    print(f"   HTML: {HTML_OUT} (http://127.0.0.1:8768/analysis/latest_surge_scan/index.html)\n")


if __name__ == "__main__":
    main()
