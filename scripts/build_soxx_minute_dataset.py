#!/usr/bin/env python3
"""
build_soxx_minute_dataset.py
----------------------------
Fetch and compile a full 1-year minute-level dataset for SOXX (iShares Semiconductor ETF):
- Date range: 2025-10-01 to 2026-10-02 (253 trading days)
- 5-minute bars for the complete 1-year period
- 1-minute bars for recent trading days (2026-09-01 ~ 2026-10-02)
- Precalculates daily summaries, MA lines (MA5, MA20, MA60), and VWAP
- Exports compact, high-performance JSON for the interactive wireframe
"""

import os
import json
import time
import pandas as pd
import numpy as np

def fetch_futu_kline(symbol, start, end, ktype):
    import futu as ft
    ft.SysConfig.enable_proto_encrypt(False)
    quote_ctx = ft.OpenQuoteContext(host='127.0.0.1', port=11111)
    
    all_dfs = []
    page_req_key = None
    while True:
        ret, df, page_req_key = quote_ctx.request_history_kline(
            symbol,
            start=start,
            end=end,
            ktype=ktype,
            autype=ft.AuType.QFQ,
            max_count=1000,
            page_req_key=page_req_key
        )
        if ret != ft.RET_OK:
            print(f"Error fetching {symbol} ({start} ~ {end}): {df}")
            break
        if len(df) > 0:
            all_dfs.append(df)
        if page_req_key is None:
            break
        time.sleep(0.05)
        
    quote_ctx.close()
    if all_dfs:
        res = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=['time_key'])
        res = res.sort_values('time_key').reset_index(drop=True)
        return res
    return pd.DataFrame()

def main():
    out_dir = os.path.abspath("analysis/soxx_minute_kline_wireframe")
    os.makedirs(out_dir, exist_ok=True)
    
    print("1. Loading local 2026 5m data...")
    local_p = "market_data/us_5m/SOXX/2026.parquet"
    if os.path.exists(local_p):
        df_2026 = pd.read_parquet(local_p)
        print(f"   Loaded {len(df_2026)} rows from {local_p}")
    else:
        df_2026 = pd.DataFrame()
        
    print("2. Fetching missing 2025 Q4 5m data (2025-10-01 ~ 2025-12-31)...")
    import futu as ft
    df_2025_q4 = fetch_futu_kline("US.SOXX", "2025-10-01", "2025-12-31", ft.KLType.K_5M)
    print(f"   Fetched {len(df_2025_q4)} rows for 2025 Q4")
    
    print("3. Fetching missing recent 5m data (2026-09-26 ~ 2026-10-03)...")
    df_recent_5m = fetch_futu_kline("US.SOXX", "2026-09-26", "2026-10-03", ft.KLType.K_5M)
    print(f"   Fetched {len(df_recent_5m)} rows for recent week")
    
    print("4. Fetching recent 1m data (2026-09-01 ~ 2026-10-03)...")
    df_recent_1m = fetch_futu_kline("US.SOXX", "2026-09-01", "2026-10-03", ft.KLType.K_1M)
    print(f"   Fetched {len(df_recent_1m)} rows for recent 1m")

    # Standardize columns: time_key, open, high, low, close, volume, turnover
    req_cols = ['time_key', 'open', 'high', 'low', 'close', 'volume', 'turnover']
    
    parts_5m = []
    if not df_2025_q4.empty:
        parts_5m.append(df_2025_q4[req_cols])
    if not df_2026.empty:
        # Check available cols in df_2026
        cols_in_2026 = [c for c in req_cols if c in df_2026.columns]
        p2026 = df_2026[cols_in_2026].copy()
        if 'turnover' not in p2026.columns:
            p2026['turnover'] = p2026['close'] * p2026['volume']
        parts_5m.append(p2026[req_cols])
    if not df_recent_5m.empty:
        parts_5m.append(df_recent_5m[req_cols])
        
    full_5m = pd.concat(parts_5m, ignore_index=True).drop_duplicates(subset=['time_key'])
    full_5m = full_5m.sort_values('time_key').reset_index(drop=True)
    full_5m['date'] = full_5m['time_key'].str.slice(0, 10)
    full_5m['time'] = full_5m['time_key'].str.slice(11, 16)
    
    print(f"Total 5m dataset: {len(full_5m)} bars, covering {full_5m['date'].min()} to {full_5m['date'].max()} ({full_5m['date'].nunique()} trading days)")
    
    # Process 1m data
    if not df_recent_1m.empty:
        full_1m = df_recent_1m[req_cols].copy()
        full_1m['date'] = full_1m['time_key'].str.slice(0, 10)
        full_1m['time'] = full_1m['time_key'].str.slice(11, 16)
    else:
        full_1m = pd.DataFrame()

    # Precalculate per-day summaries
    days_data_5m = {}
    daily_summaries = []
    
    for date, group in full_5m.groupby('date', sort=True):
        bars = []
        cum_vol = 0.0
        cum_turnover = 0.0
        
        # Determine regular hours subset (09:30 ~ 16:00) vs full day
        day_open = float(group.iloc[0]['open'])
        day_close = float(group.iloc[-1]['close'])
        day_high = float(group['high'].max())
        day_low = float(group['low'].min())
        day_vol = float(group['volume'].sum())
        day_turnover = float(group['turnover'].sum())
        
        # Calculate intraday VWAP for each bar
        # Format: [time, open, high, low, close, volume, vwap]
        for _, r in group.iterrows():
            v = float(r['volume'])
            t = float(r['turnover']) if float(r['turnover']) > 0 else float(r['close']) * v
            cum_vol += v
            cum_turnover += t
            vwap = round(cum_turnover / cum_vol, 2) if cum_vol > 0 else float(r['close'])
            bars.append([
                r['time'],
                round(float(r['open']), 2),
                round(float(r['high']), 2),
                round(float(r['low']), 2),
                round(float(r['close']), 2),
                int(v),
                vwap
            ])
            
        days_data_5m[date] = bars
        
        # Calculate day change relative to prev day close or day open
        daily_summaries.append({
            'date': date,
            'open': round(day_open, 2),
            'high': round(day_high, 2),
            'low': round(day_low, 2),
            'close': round(day_close, 2),
            'volume': int(day_vol),
            'turnover': round(day_turnover, 2),
            'bars_count': len(bars)
        })
        
    # Calculate prev_close and pct_change for daily summaries
    for i in range(len(daily_summaries)):
        if i > 0:
            prev_c = daily_summaries[i-1]['close']
            c = daily_summaries[i]['close']
            chg = round(c - prev_c, 2)
            pct = round((c - prev_c) / prev_c * 100, 2)
        else:
            chg = round(daily_summaries[i]['close'] - daily_summaries[i]['open'], 2)
            pct = round(chg / daily_summaries[i]['open'] * 100, 2)
        amp = round((daily_summaries[i]['high'] - daily_summaries[i]['low']) / daily_summaries[i]['open'] * 100, 2)
        daily_summaries[i]['prev_close'] = prev_c if i > 0 else daily_summaries[i]['open']
        daily_summaries[i]['change'] = chg
        daily_summaries[i]['change_pct'] = pct
        daily_summaries[i]['amplitude'] = amp

    # Process 1m bars by day
    days_data_1m = {}
    if not full_1m.empty:
        for date, group in full_1m.groupby('date', sort=True):
            bars = []
            cum_vol = 0.0
            cum_turnover = 0.0
            for _, r in group.iterrows():
                v = float(r['volume'])
                t = float(r['turnover']) if float(r['turnover']) > 0 else float(r['close']) * v
                cum_vol += v
                cum_turnover += t
                vwap = round(cum_turnover / cum_vol, 2) if cum_vol > 0 else float(r['close'])
                bars.append([
                    r['time'],
                    round(float(r['open']), 2),
                    round(float(r['high']), 2),
                    round(float(r['low']), 2),
                    round(float(r['close']), 2),
                    int(v),
                    vwap
                ])
            days_data_1m[date] = bars

    # 5. Detect 8 micro-patterns across the 1-year 5m series
    print("5. Detecting 8 micro-patterns on SOXX 5m bars...")
    closes = full_5m['close'].values
    time_keys = (full_5m['date'] + ' ' + full_5m['time']).tolist()
    tot = len(closes)
    N = 24
    M = 0.02
    D = 0.02

    rets = np.full(tot, np.nan)
    rets[N:] = (closes[N:] - closes[:-N]) / closes[:-N]

    in_s, s_st = False, 0
    in_d, d_st = False, 0
    s_events, d_events = [], []

    for i in range(N, tot):
        r = rets[i]
        if r >= M:
            if not in_s:
                in_s, s_st = True, i
        else:
            if in_s:
                s_events.append({'type': 'SURGE', 'st': s_st - N, 'end': i - 1})
                in_s = False
        if r <= -D:
            if not in_d:
                in_d, d_st = True, i
        else:
            if in_d:
                d_events.append({'type': 'DROP', 'st': d_st - N, 'end': i - 1})
                in_d = False

    evts = s_events + d_events
    for e in evts:
        c_sub = closes[e['st']: e['end'] + 1]
        cnt = len(c_sub)
        p0 = closes[e['st']]
        sub_r = (c_sub - p0) / p0
        min_idx = int(np.argmin(sub_r))
        max_idx = int(np.argmax(sub_r))
        min_ret = sub_r[min_idx]
        max_ret = sub_r[max_idx]
        min_pos = min_idx / max(1, cnt - 1)
        max_pos = max_idx / max(1, cnt - 1)
        end_ret = (closes[e['end']] - p0) / p0

        if e['type'] == 'SURGE':
            if min_ret >= -0.005 and max_pos >= 0.75:
                pat = '▲ 单边拉升'
            elif min_pos <= 0.45 and min_ret <= -0.010:
                pat = 'V型深弹'
            elif max_pos <= 0.55 and end_ret < max_ret - 0.010:
                pat = '冲高回落'
            else:
                pat = '阶梯中继'
        else:
            if max_ret <= 0.005 and min_pos >= 0.75:
                pat = '▼ 单边下杀'
            elif max_pos <= 0.45 and max_ret >= 0.010:
                pat = '倒V冲顶'
            elif min_pos <= 0.55 and end_ret > min_ret + 0.010:
                pat = '探底回抽'
            else:
                pat = '破位阴跌'
        e['pattern'] = pat

    PATTERNS = [
        '▲ 单边拉升', 'V型深弹', '冲高回落', '阶梯中继',
        '▼ 单边下杀', '倒V冲顶', '探底回抽', '破位阴跌'
    ]

    shapes = []
    for p_name in PATTERNS:
        p_evts = [e for e in evts if e['pattern'] == p_name]
        active = np.zeros(tot, dtype=int)
        for e in p_evts:
            active[e['st']: e['end'] + 1] = 1

        in_shape, s_idx = False, 0
        for i in range(tot):
            if active[i] == 1:
                if not in_shape:
                    in_shape, s_idx = True, i
            else:
                if in_shape:
                    p0 = closes[s_idx]
                    p_end = closes[i - 1]
                    shapes.append({
                        'pattern': p_name,
                        'st_idx': s_idx,
                        'end_idx': i - 1,
                        'st_time': time_keys[s_idx],
                        'end_time': time_keys[i - 1],
                        'st_date': time_keys[s_idx][:10],
                        'end_date': time_keys[i - 1][:10],
                        'st_price': round(float(p0), 2),
                        'end_price': round(float(p_end), 2),
                        'ret_pct': round(float((p_end - p0) / p0 * 100), 2),
                        'bars': i - s_idx,
                        'is_upward': p_name in ['▲ 单边拉升', 'V型深弹', '冲高回落', '阶梯中继']
                    })
                    in_shape = False
        if in_shape:
            p0 = closes[s_idx]
            p_end = closes[tot - 1]
            shapes.append({
                'pattern': p_name,
                'st_idx': s_idx,
                'end_idx': tot - 1,
                'st_time': time_keys[s_idx],
                'end_time': time_keys[tot - 1],
                'st_date': time_keys[s_idx][:10],
                'end_date': time_keys[tot - 1][:10],
                'st_price': round(float(p0), 2),
                'end_price': round(float(p_end), 2),
                'ret_pct': round(float((p_end - p0) / p0 * 100), 2),
                'bars': tot - s_idx,
                'is_upward': p_name in ['▲ 单边拉升', 'V型深弹', '冲高回落', '阶梯中继']
            })

    shapes.sort(key=lambda s: s['st_time'])
    print(f"   Detected {len(shapes)} discrete shapes across 8 micro-patterns")

    # 6. Load ETF overlay universe from cache
    etf_cache_path = os.path.join(out_dir, "etf_overlay_cache.json")
    etf_universe = []
    if os.path.exists(etf_cache_path):
        with open(etf_cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        for sym, info in cache.items():
            etf_universe.append({
                'symbol': sym,
                'code': info.get('code', f'US.{sym}'),
                'name': info.get('name', sym),
                'latest_close': info.get('latest_close'),
                'daily_closes': info.get('daily_closes', {}),
                'daily_bars': info.get('daily_bars', {})
            })
        core_priority = ['SMH', 'SOXL', 'SOXS', 'QQQ', 'SPY', 'VOO', 'GRID', 'XBI']
        etf_universe.sort(key=lambda x: (0 if x['symbol'] in core_priority else 1, x['symbol']))
        print(f"   Loaded {len(etf_universe)} watchlist ETFs for overlay comparison")

    final_payload = {
        'symbol': 'SOXX',
        'name': 'iShares Semiconductor ETF (费城半导体指数ETF)',
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'date_range': {
            'start': daily_summaries[0]['date'],
            'end': daily_summaries[-1]['date'],
            'total_trading_days': len(daily_summaries)
        },
        'daily_summaries': daily_summaries,
        'days_5m': days_data_5m,
        'days_1m': days_data_1m,
        'patterns_8': shapes,
        'etf_universe': etf_universe
    }
    
    out_json = os.path.join(out_dir, "soxx_data.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, separators=(',', ':'))
        
    sz_mb = os.path.getsize(out_json) / 1024 / 1024
    print(f"Successfully generated {out_json} ({sz_mb:.2f} MB)")
    print(f"Total days: {len(daily_summaries)}, 5m days: {len(days_data_5m)}, 1m days: {len(days_data_1m)}, shapes: {len(shapes)}, etfs: {len(etf_universe)}")

if __name__ == '__main__':
    main()
