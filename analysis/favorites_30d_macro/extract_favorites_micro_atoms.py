#!/usr/bin/env python3
"""
Extract 2-hour micro atomic features (A-01 ~ A-12) from 5-minute K-lines
and annotate them onto the 30-day macro trend series for all user Favorites.

Outputs:
  analysis/favorites_30d_macro/favorites_30d_data.json (enriched with micro_atoms)
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import futu as ft

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FAV_DATA_JSON = ROOT_DIR / "analysis" / "favorites_30d_macro" / "favorites_30d_data.json"
MICRO_ENGINE_JSON = ROOT_DIR / "analysis" / "micro_atomic_motif_engine" / "data.json"
MARKET_DATA_5M = ROOT_DIR / "market_data" / "us_5m"

# 12 Micro Atomic Descriptors and Visual Palette
ATOM_INFO = {
    "A-01": {"name": "震荡过渡整理", "color": "#8b5cf6", "type": "中继盘整", "bull": False},
    "A-02": {"name": "倒V诱多暴跌", "color": "#dc2626", "type": "假突破见顶", "bull": False},
    "A-03": {"name": "稳步单边攀升", "color": "#10b981", "type": "健康主升", "bull": True},
    "A-04": {"name": "深V探底回升", "color": "#06b6d4", "type": "强力反转", "bull": True},
    "A-05": {"name": "恐慌单边杀跌", "color": "#b91c1c", "type": "单边破位", "bull": False},
    "A-06": {"name": "脉冲爆量强攻", "color": "#059669", "type": "起爆突破", "bull": True},
    "A-07": {"name": "高位双顶滞涨", "color": "#f97316", "type": "双顶遇阻", "bull": False},
    "A-08": {"name": "阶梯筑底蓄势", "color": "#f59e0b", "type": "横盘筑底", "bull": True},
    "A-09": {"name": "急跌抽升高抛", "color": "#ea580c", "type": "超跌反抽", "bull": False},
    "A-10": {"name": "窄幅微波沉寂", "color": "#64748b", "type": "极度缩量", "bull": False},
    "A-11": {"name": "破位二次下探", "color": "#ef4444", "type": "破位寻底", "bull": False},
    "A-12": {"name": "尾盘脉冲突袭", "color": "#047857", "type": "尾盘拉升", "bull": True}
}

def load_micro_codebook():
    with open(MICRO_ENGINE_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    codebook = d["codebook"]
    centroids = np.array([c["curve"] for c in codebook])
    print(f"Loaded {len(centroids)} micro centroids (shape: {centroids.shape}) from micro_atomic_motif_engine")
    return codebook, centroids

def fetch_5m_bars_from_futu(quote_ctx, code):
    """Fetch 30-day 5m bars from Futu OpenD with pagination."""
    all_dfs = []
    page_key = None
    while True:
        ret, df, page_key = quote_ctx.request_history_kline(
            code,
            start="2026-08-17",
            end="2026-09-29",
            ktype=ft.KLType.K_5M,
            autype=ft.AuType.QFQ,
            max_count=1000,
            page_req_key=page_key
        )
        if ret == ft.RET_OK and len(df) > 0:
            all_dfs.append(df)
            if not page_key or len(page_key) == 0:
                break
        else:
            break
        time.sleep(0.1)

    if not all_dfs:
        return pd.DataFrame()
    total_df = pd.concat(all_dfs).drop_duplicates(subset=["time_key"]).sort_values("time_key").reset_index(drop=True)
    return total_df

def extract_micro_atoms_for_stock(df_5m, macro_bars, codebook, centroids, stock_code=""):
    """Extract salient 2-hour micro atomic features and map them to 60m macro bars."""
    if len(df_5m) < 24:
        return []

    # Filter regular hours
    df_5m["time"] = df_5m["time_key"].astype(str).str.split(" ").str[1]
    df_5m["date"] = df_5m["time_key"].astype(str).str.split(" ").str[0]
    
    # Check if US or HK
    is_hk = stock_code.startswith("HK.")
    reg_df = df_5m[(df_5m["time"] >= "09:30:00") & (df_5m["time"] <= "16:00:00")].sort_values("time_key").reset_index(drop=True)

    all_dates = sorted(reg_df["date"].unique())
    recent_30_dates = all_dates[-30:] if len(all_dates) >= 30 else all_dates
    reg_df = reg_df[reg_df["date"].isin(recent_30_dates)].reset_index(drop=True)

    c_all = reg_df["close"].values
    t_all = reg_df["time_key"].values
    L = len(c_all)
    if L < 24:
        return []

    # Time lookup for 60m macro bars: map timestamp to macro bar index
    macro_times = [b["time"] for b in macro_bars]

    def find_nearest_macro_idx(target_time):
        target_d = target_time[:10]
        # Find closest macro bar on same day
        candidates = [i for i, t in enumerate(macro_times) if t.startswith(target_d)]
        if not candidates:
            # Fallback to closest overall
            return min(range(len(macro_times)), key=lambda i: abs((pd.to_datetime(macro_times[i]) - pd.to_datetime(target_time)).total_seconds()))
        # Choose closest hour
        target_dt = pd.to_datetime(target_time)
        return min(candidates, key=lambda i: abs((pd.to_datetime(macro_times[i]) - target_dt).total_seconds()))

    # Sliding 2-hour window (24 5m bars), step 3 bars (15 mins)
    raw_matches = []
    for i in range(23, L, 3):
        sub_c = c_all[i - 23 : i + 1]
        sub_t = t_all[i - 23 : i + 1]
        # Must be intraday within same session
        if sub_t[0][:10] != sub_t[-1][:10]:
            continue

        p_base = sub_c[0]
        if p_base <= 0:
            continue
        amp = (np.max(sub_c) - np.min(sub_c)) / p_base * 100.0
        if amp < 1.6: # filter non-volatile noise
            continue

        std = float(np.std(sub_c))
        if std < 1e-6:
            continue
        z = (sub_c - np.mean(sub_c)) / std
        dists = np.linalg.norm(centroids - z, axis=1)
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        sim = max(0.0, 1.0 - (best_dist**2) / 48.0) * 100.0

        if sim >= 82.0:
            atom_code = codebook[best_idx]["code"]
            info = ATOM_INFO.get(atom_code, {"name": atom_code, "color": "#64748b", "type": "微观波形", "bull": False})
            ret = float((sub_c[-1] - p_base) / p_base * 100.0)
            
            raw_matches.append({
                "date": sub_t[0][:10],
                "start_time": sub_t[0],
                "end_time": sub_t[-1],
                "start_price": round(float(p_base), 2),
                "end_price": round(float(sub_c[-1]), 2),
                "atom": atom_code,
                "name": info["name"],
                "type": info["type"],
                "color": info["color"],
                "bull": info["bull"],
                "sim": round(sim, 1),
                "amplitude": round(amp, 2),
                "ret": round(ret, 2),
                "salience": sim * (1.0 + amp / 8.0) # combination of similarity and volatility
            })

    # Sort by salience
    raw_matches.sort(key=lambda x: x["salience"], reverse=True)

    # Non-Maximum Suppression (NMS): at most 1 salient feature per day, max 4-5 features across 30 days
    selected_days = set()
    selected = []
    for m in raw_matches:
        if m["date"] in selected_days:
            continue
        selected_days.add(m["date"])
        
        # Map to 60m macro bar indices for drawing on macro chart
        s_macro = find_nearest_macro_idx(m["start_time"])
        e_macro = find_nearest_macro_idx(m["end_time"])
        if e_macro <= s_macro:
            e_macro = min(len(macro_bars) - 1, s_macro + 2)

        m["macro_start_idx"] = s_macro
        m["macro_end_idx"] = e_macro

        # Compute macro forward performance strictly anchored at the micro interval's start price!
        p_base_micro = float(m["start_price"])
        perf = {}
        for h_bars, key in [(7, 'h7'), (14, 'h14'), (21, 'h21')]:
            end_fwd_idx = min(len(macro_bars) - 1, s_macro + h_bars)
            fwd_bars = macro_bars[s_macro:end_fwd_idx + 1]
            if len(fwd_bars) > 1:
                highs = [(b.get('high', b['close']), idx + s_macro) for idx, b in enumerate(fwd_bars)]
                lows = [(b.get('low', b['close']), idx + s_macro) for idx, b in enumerate(fwd_bars)]
                max_high, peak_idx = max(highs, key=lambda x: x[0])
                min_low, trough_idx = min(lows, key=lambda x: x[0])
                end_close = fwd_bars[-1]['close']
                
                # Base is STRICTLY micro start price p_base_micro!
                gain_pct = round((max_high / p_base_micro - 1.0) * 100.0, 2)
                dd_pct = round((min_low / p_base_micro - 1.0) * 100.0, 2)
                end_ret_pct = round((end_close / p_base_micro - 1.0) * 100.0, 2)
                
                perf[key] = {
                    'base_price': round(float(p_base_micro), 2),
                    'bars_count': len(fwd_bars),
                    'end_idx': end_fwd_idx,
                    'max_gain': gain_pct,
                    'max_dd': dd_pct,
                    'end_ret': end_ret_pct,
                    'hit_5pct': gain_pct >= 5.0,
                    'hit_3pct': gain_pct >= 3.0,
                    'peak_idx': peak_idx,
                    'peak_price': round(float(max_high), 2),
                    'trough_idx': trough_idx,
                    'trough_price': round(float(min_low), 2)
                }
        m['macro_perf'] = perf
        selected.append(m)
        if len(selected) >= 5:
            break

    # Sort chronologically
    selected.sort(key=lambda x: x["start_time"])
    return selected

def main():
    t0 = time.time()
    print("=" * 70)
    print("EXTRACTING 2-HOUR MICRO ATOMIC MOTIFS (A-01 ~ A-12) FOR FAVORITES")
    print("=" * 70)

    codebook, centroids = load_micro_codebook()

    with open(FAV_DATA_JSON, "r", encoding="utf-8") as f:
        fav_payload = json.load(f)

    stocks = fav_payload["stocks"]
    print(f"Loaded {len(stocks)} stocks from {FAV_DATA_JSON.name}")

    ft.SysConfig.enable_proto_encrypt(False)
    quote_ctx = None

    try:
        quote_ctx = ft.OpenQuoteContext(host="127.0.0.1", port=11111)
        print("Connected to Futu OpenD for missing 5m data.")
    except Exception as e:
        print(f"Notice: Cannot connect to Futu OpenD ({e}), relying on local 5m parquets.")

    total_atoms_extracted = 0

    for i, s in enumerate(stocks, 1):
        ticker = s["ticker"]
        code = s["code"]
        macro_bars = s["bars"]

        df_5m = None
        local_p = MARKET_DATA_5M / ticker / "2026.parquet"

        if local_p.exists():
            try:
                df_5m = pd.read_parquet(local_p)
                if "code" not in df_5m.columns:
                    df_5m["code"] = code
            except Exception as e:
                df_5m = None

        if df_5m is None and quote_ctx is not None:
            print(f"[{i}/{len(stocks)}] {code:10s} - Fetching 5m from Futu OpenD...")
            df_5m = fetch_5m_bars_from_futu(quote_ctx, code)
            if len(df_5m) > 0:
                # Save locally for caching
                ticker_dir = MARKET_DATA_5M / ticker
                ticker_dir.mkdir(parents=True, exist_ok=True)
                df_5m.to_parquet(ticker_dir / "2026.parquet")

        if df_5m is None or len(df_5m) == 0:
            print(f"[{i}/{len(stocks)}] {code:10s} - ⚠️ No 5m data available")
            s["micro_atoms"] = []
            continue

        # Extract 2-hour micro atomic features
        micro_atoms = extract_micro_atoms_for_stock(df_5m, macro_bars, codebook, centroids, code)
        s["micro_atoms"] = micro_atoms
        total_atoms_extracted += len(micro_atoms)

        atom_summary = ", ".join([f"{a['atom']}({a['sim']}%)" for a in micro_atoms])
        print(f"[{i}/{len(stocks)}] {code:10s} | Found {len(micro_atoms)} micro atoms: {atom_summary}")

    if quote_ctx:
        quote_ctx.close()

    # Compute empirical summary across all micro atoms
    results = []
    for s in stocks:
        for a in s.get('micro_atoms', []):
            for h, h_name in [(7, 'h7'), (14, 'h14'), (21, 'h21')]:
                p = a.get('macro_perf', {}).get(h_name)
                if p:
                    results.append({
                        'atom': a['atom'],
                        'name': a['name'],
                        'color': a['color'],
                        'horizon': h_name,
                        'hit_5pct': 1 if p['hit_5pct'] else 0,
                        'max_gain': p['max_gain'],
                        'max_dd': p['max_dd'],
                        'end_ret': p['end_ret']
                    })

    summary = []
    curve_map = {c['code']: [round(float(v), 2) for v in c['curve']] for c in codebook}
    desc_map = {c['code']: c.get('descriptor', '') for c in codebook}

    if len(results) > 0:
        df_res = pd.DataFrame(results)
        for (atom, name, color), g in df_res.groupby(['atom', 'name', 'color']):
            cnt = len(g) // 3
            h7_hit = g[g['horizon'] == 'h7']['hit_5pct'].mean() * 100
            h14_hit = g[g['horizon'] == 'h14']['hit_5pct'].mean() * 100
            h21_hit = g[g['horizon'] == 'h21']['hit_5pct'].mean() * 100
            h21_gain = g[g['horizon'] == 'h21']['max_gain'].mean()
            h21_dd = g[g['horizon'] == 'h21']['max_dd'].mean()
            h21_ret = g[g['horizon'] == 'h21']['end_ret'].mean()
            
            summary.append({
                'atom': atom,
                'name': name,
                'color': color,
                'count': cnt,
                'descriptor': desc_map.get(atom, ''),
                'curve': curve_map.get(atom, []),
                'h7_hit': round(h7_hit, 1),
                'h14_hit': round(h14_hit, 1),
                'h21_hit': round(h21_hit, 1),
                'h21_gain': round(h21_gain, 1),
                'h21_dd': round(h21_dd, 1),
                'h21_ret': round(h21_ret, 1)
            })
        summary.sort(key=lambda x: x['h21_hit'], reverse=True)

    fav_payload["macro_summary"] = summary
    fav_payload["micro_atoms_total"] = total_atoms_extracted
    with open(FAV_DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(fav_payload, f, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"SUCCESS: Enriched {len(stocks)} stocks with {total_atoms_extracted} 2-hour micro atoms ({time.time() - t0:.1f}s)")
    print("=" * 70)

if __name__ == "__main__":
    main()
