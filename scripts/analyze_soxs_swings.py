#!/usr/bin/env python3
"""
分析 SOXS 最近 30 天内：
- 涨幅 >= 5%
- 回撤 < 4%
- 不重复窗口 (Non-overlapping)

在 日K、60分钟K、5分钟K 三个粒度下的精确计算与比对。
"""

import pandas as pd
import numpy as np

def find_non_overlapping_windows(df, time_col='time_key', high_col='high', low_col='low', close_col='close', label=''):
    """
    寻找所有不重复的上涨波段窗口：
    - 窗口起点为某一局部低点 (或 bar 的 low/close)
    - 推进过程中，跟踪创新高 peak
    - 期间任意时刻从 running high 的回撤不能达到 4%
    - 当出现从 peak 回撤 >= 4% (或跌破起点) 时，当前波段受阻结束
    - 若此时达到 peak 的涨幅 (peak - start) / start >= 5%，且期间最大回撤 < 4%，记录为一个合格窗口
    - 下一个窗口从该波段结束/打破之后寻找，确保窗口互不重叠
    """
    n = len(df)
    windows = []

    i = 0
    while i < n:
        # 寻找起步低点：如果当前点还在下跌，顺延到低点
        start_idx = i
        start_price = df.loc[start_idx, low_col]
        start_time = df.loc[start_idx, time_col]

        peak_idx = start_idx
        peak_price = df.loc[start_idx, high_col]
        peak_time = df.loc[start_idx, time_col]

        curr = start_idx
        found_window = None

        while curr < n:
            c_high = df.loc[curr, high_col]
            c_low = df.loc[curr, low_col]

            # 若创新低且尚未开启有效上涨，重置起点
            if c_low < start_price and peak_price <= start_price * 1.01:
                start_price = c_low
                start_time = df.loc[curr, time_col]
                start_idx = curr
                peak_price = c_high
                peak_idx = curr
                peak_time = df.loc[curr, time_col]
                curr += 1
                continue

            # 更新高点
            if c_high > peak_price:
                peak_price = c_high
                peak_idx = curr
                peak_time = df.loc[curr, time_col]

            # 检查回撤
            dd = (peak_price - c_low) / peak_price
            if dd >= 0.04:
                # 受到 4% 回撤阻断
                gain = (peak_price - start_price) / start_price
                if gain >= 0.05 and peak_idx > start_idx:
                    # 验证从 start 到 peak 之间的实际最大回撤
                    sub_highs = df.loc[start_idx:peak_idx, high_col].values
                    sub_lows = df.loc[start_idx:peak_idx, low_col].values
                    running_max = np.maximum.accumulate(sub_highs)
                    max_dd = np.max((running_max - sub_lows) / running_max) if len(running_max) > 0 else 0

                    if max_dd < 0.04:
                        found_window = {
                            'start_time': start_time,
                            'peak_time': peak_time,
                            'break_time': df.loc[curr, time_col],
                            'start_price': start_price,
                            'peak_price': peak_price,
                            'gain_pct': gain * 100,
                            'max_dd_pct': max_dd * 100,
                            'bars_count': peak_idx - start_idx + 1,
                            'start_idx': start_idx,
                            'peak_idx': peak_idx,
                            'break_idx': curr
                        }
                break

            curr += 1

        if found_window is None and curr >= n:
            # 检查到达尾盘时是否仍在波段中且达标
            gain = (peak_price - start_price) / start_price
            if gain >= 0.05 and peak_idx > start_idx:
                sub_highs = df.loc[start_idx:peak_idx, high_col].values
                sub_lows = df.loc[start_idx:peak_idx, low_col].values
                running_max = np.maximum.accumulate(sub_highs)
                max_dd = np.max((running_max - sub_lows) / running_max) if len(running_max) > 0 else 0
                if max_dd < 0.04:
                    found_window = {
                        'start_time': start_time,
                        'peak_time': peak_time,
                        'break_time': '最新/收盘未阻断',
                        'start_price': start_price,
                        'peak_price': peak_price,
                        'gain_pct': gain * 100,
                        'max_dd_pct': max_dd * 100,
                        'bars_count': peak_idx - start_idx + 1,
                        'start_idx': start_idx,
                        'peak_idx': peak_idx,
                        'break_idx': curr
                    }

        if found_window:
            windows.append(found_window)
            # 不重叠：从 break_idx 或 peak_idx + 1 开始下一轮扫描
            next_start = max(found_window['break_idx'], found_window['peak_idx'] + 1)
            i = next_start
        else:
            i += 1

    return windows

def find_close_to_close_windows(df, time_col='time_key', close_col='close'):
    closes = df[close_col].values
    times = df[time_col].values
    n = len(closes)
    windows = []
    i = 0
    while i < n:
        start_p = closes[i]
        peak_p = start_p
        peak_idx = i
        curr = i
        found = None
        while curr < n:
            c = closes[curr]
            if c > peak_p:
                peak_p = c
                peak_idx = curr
            dd = (peak_p - c) / peak_p
            if dd >= 0.04:
                gain = (peak_p - start_p) / start_p
                if gain >= 0.05 and peak_idx > i:
                    sub_c = closes[i:peak_idx+1]
                    run_max = np.maximum.accumulate(sub_c)
                    max_dd = np.max((run_max - sub_c) / run_max)
                    if max_dd < 0.04:
                        found = {
                            'start_time': times[i],
                            'peak_time': times[peak_idx],
                            'start_price': start_p,
                            'peak_price': peak_p,
                            'gain_pct': gain * 100,
                            'max_dd_pct': max_dd * 100,
                            'bars_count': peak_idx - i + 1,
                            'break_time': times[curr]
                        }
                break
            curr += 1
        if found:
            windows.append(found)
            i = peak_idx + 1
        else:
            i += 1
    return windows

def main():
    df_d = pd.read_parquet('market_data/SOXS_recent_daily.parquet')
    df_60 = pd.read_parquet('market_data/SOXS_recent_60m.parquet')
    df_5m = pd.read_parquet('market_data/SOXS_recent_5m.parquet')

    # 最近 30 个自然日: 2026-09-08 至 2026-10-07 (22 个交易日)
    df_d_30cal = df_d[df_d['time_key'] >= '2026-09-08'].reset_index(drop=True)
    df_60_30cal = df_60[df_60['time_key'] >= '2026-09-08'].reset_index(drop=True)
    df_5m_30cal = df_5m[df_5m['time_key'] >= '2026-09-08'].reset_index(drop=True)

    # 最近 30 个交易日 (从倒数第 30 根日K开始)
    df_d_30trd = df_d.tail(30).reset_index(drop=True)
    start_trd_date = df_d_30trd['time_key'].min()
    df_60_30trd = df_60[df_60['time_key'] >= start_trd_date].reset_index(drop=True)

    print("=" * 80)
    print("🎯 SOXS 最近 30 天 上涨波段分析 (2026-09-08 ~ 2026-10-07)")
    print("核心约束: 涨幅 >= 5% | 期间最大回撤 < 4% | 窗口互不重叠 (Disjoint)")
    print("=" * 80)

    print("\n" + "#" * 60)
    print("一、基准口径：最近 30 个自然日 (2026-09-08 ~ 2026-10-07, 共 22 交易日)")
    print("#" * 60)

    # 1. 日K收盘价口径
    w_d_c = find_close_to_close_windows(df_d_30cal)
    print(f"\n📌 【日K级别 · 收盘价口径】: 共识别出 {len(w_d_c)} 个窗口")
    for idx, item in enumerate(w_d_c, 1):
        s_time = str(item['start_time'])[:10]
        p_time = str(item['peak_time'])[:10]
        print(f"  {idx}. 起点: {s_time} (${item['start_price']:.2f}) -> 峰值: {p_time} (${item['peak_price']:.2f}) | 涨幅: +{item['gain_pct']:.2f}% | 期间回撤: {item['max_dd_pct']:.2f}%")

    # 2. 60分钟K口径
    w_60 = find_non_overlapping_windows(df_60_30cal, time_col='time_key')
    print(f"\n📌 【60分钟K级别 · 极值波段口径】: 共识别出 {len(w_60)} 个窗口")
    for idx, item in enumerate(w_60, 1):
        s_time = str(item['start_time'])[:16]
        p_time = str(item['peak_time'])[:16]
        b_time = str(item['break_time'])[:16]
        print(f"  {idx}. 起点: {s_time} (${item['start_price']:.2f}) -> 峰值: {p_time} (${item['peak_price']:.2f}) | 涨幅: +{item['gain_pct']:.2f}% | 期间回撤: {item['max_dd_pct']:.2f}% | 持续: {item['bars_count']}小时")

    # 3. 5分钟K口径
    w_5m = find_non_overlapping_windows(df_5m_30cal, time_col='time_key')
    print(f"\n📌 【5分钟K级别 · 精细微观波段口径】: 共识别出 {len(w_5m)} 个窗口")
    for idx, item in enumerate(w_5m, 1):
        s_time = str(item['start_time'])[:16]
        p_time = str(item['peak_time'])[:16]
        b_time = str(item['break_time'])[:16]
        print(f"  {idx}. 起点: {s_time} (${item['start_price']:.2f}) -> 峰值: {p_time} (${item['peak_price']:.2f}) | 涨幅: +{item['gain_pct']:.2f}% | 期间回撤: {item['max_dd_pct']:.2f}% | 持续: {item['bars_count']*5}分钟")

    print("\n" + "#" * 60)
    print(f"二、对照口径：最近 30 个交易日 ({str(start_trd_date)[:10]} ~ 2026-10-07)")
    print("#" * 60)
    w_60_trd = find_non_overlapping_windows(df_60_30trd, time_col='time_key')
    print(f"📌 【60分钟K级别 · 最近30交易日】: 共识别出 {len(w_60_trd)} 个窗口")
    for idx, item in enumerate(w_60_trd, 1):
        s_time = str(item['start_time'])[:16]
        p_time = str(item['peak_time'])[:16]
        print(f"  {idx}. 起点: {s_time} (${item['start_price']:.2f}) -> 峰值: {p_time} (${item['peak_price']:.2f}) | 涨幅: +{item['gain_pct']:.2f}% | 期间回撤: {item['max_dd_pct']:.2f}%")

if __name__ == '__main__':
    main()
