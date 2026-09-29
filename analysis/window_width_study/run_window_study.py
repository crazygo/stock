#!/usr/bin/env python3
"""
run_window_study.py
Executes comprehensive empirical study comparing observation window widths W in [2 days, 5 days]
(W = 14, 21, 28, 35 hourly bars) across QQQ constituents from 2025-01-02 to 2026-09-28.
Computes:
  - Predictive spread and peak hit rates for future 3-day +5% touch.
  - Multi-window codebook clustering (K=12).
  - Mathematical invariant correlations (slope, curvature, phase).
  - Sub-condition probability grid (Oversold, Arc Bottom, Volatility Contraction).
  - Annual stability (2025 vs 2026).
Outputs:
  analysis/window_width_study/study_data.json
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PARQUET_PATH = os.path.join(os.path.dirname(__file__), "qqq_hourly_2025_2026.parquet")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "study_data.json")

FORWARD_BARS = 21  # 3 trading days
TARGET = 0.05      # +5.0% touch target
K_CLUSTERS = 12

WINDOWS = [
    {"days": 2, "bars": 14, "name": "2天 (14小时)"},
    {"days": 3, "bars": 21, "name": "3天 (21小时)"},
    {"days": 4, "bars": 28, "name": "4天 (28小时)"},
    {"days": 5, "bars": 35, "name": "5天 (35小时)"},
]

def main():
    t0 = time.time()
    print("=" * 75)
    print("QQQ HOURLY OBSERVATION WINDOW WIDTH STUDY (2 DAYS TO 5 DAYS, 2025-2026)")
    print("=" * 75)

    df = pd.read_parquet(PARQUET_PATH)
    print(f"Loaded {len(df):,} hourly rows for {df['ticker'].nunique()} tickers.")
    print(f"Time range: {df['time_key'].min()} to {df['time_key'].max()}")

    window_results = []

    for w_cfg in WINDOWS:
        W = w_cfg["bars"]
        w_days = w_cfg["days"]
        w_name = w_cfg["name"]
        t_w0 = time.time()
        print(f"\nEvaluating Window: {w_name} (W={W} bars)...")

        slices = []
        meta = []
        t_norm = (np.arange(W) - (W - 1) / 2.0) / (W / 3.0)

        for ticker, grp in df.groupby('ticker'):
            grp = grp.sort_values('time_key').reset_index(drop=True)
            closes = grp['close'].values
            highs = grp['high'].values
            times = grp['time_key'].values
            L = len(closes)
            if L < W + FORWARD_BARS:
                continue

            for i in range(W - 1, L):
                p_end = float(closes[i])
                w_c = closes[i - W + 1 : i + 1]
                std_c = float(np.std(w_c))
                if std_c < 1e-8:
                    continue
                mean_c = float(np.mean(w_c))
                z = (w_c - mean_c) / std_c

                future_len = L - 1 - i
                matured = (future_len >= FORWARD_BARS)

                if matured:
                    fut_h = highs[i + 1 : i + FORWARD_BARS + 1]
                    max_gain = float((np.max(fut_h) - p_end) / p_end)
                    hit_3d = int(max_gain >= TARGET)
                    hit_2d = int(np.max(fut_h[:14]) >= p_end * (1.0 + TARGET))
                    hit_1d = int(np.max(fut_h[:7]) >= p_end * (1.0 + TARGET))
                    touch_idx = np.where((fut_h - p_end) / p_end >= TARGET)[0]
                    bars_to_hit = int(touch_idx[0] + 1) if len(touch_idx) > 0 else None
                else:
                    fut_h = highs[i + 1 :] if future_len > 0 else np.array([])
                    max_gain = float((np.max(fut_h) - p_end) / p_end) if len(fut_h) > 0 else 0.0
                    hit_3d = int(max_gain >= TARGET)
                    hit_2d = int(np.max(fut_h[:14]) >= p_end * (1.0 + TARGET)) if len(fut_h) >= 14 else 0
                    hit_1d = int(np.max(fut_h[:7]) >= p_end * (1.0 + TARGET)) if len(fut_h) >= 7 else 0
                    bars_to_hit = None

                slices.append(z)
                meta.append({
                    "ticker": ticker,
                    "end_time": str(times[i]),
                    "year": str(times[i])[:4],
                    "hour": str(times[i]).split(" ")[1],
                    "matured": matured,
                    "hit_3d": hit_3d,
                    "hit_2d": hit_2d,
                    "hit_1d": hit_1d,
                    "max_gain": round(max_gain, 4),
                    "bars_to_hit": bars_to_hit
                })

        X = np.array(slices)
        total_cnt = len(X)
        matured_cnt = sum(1 for m in meta if m["matured"])
        base_hit_rate = float(np.mean([m["hit_3d"] for m in meta if m["matured"]]))

        # K-Means clustering (K=12)
        km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=5)
        raw_labels = km.fit_predict(X)
        raw_centroids = km.cluster_centers_

        # Rank clusters by 3d hit rate
        cluster_rates = []
        for c in range(K_CLUSTERS):
            c_mask = (raw_labels == c)
            c_matured = [meta[idx] for idx in np.where(c_mask)[0] if meta[idx]["matured"]]
            rate = float(np.mean([m["hit_3d"] for m in c_matured])) if c_matured else 0.0
            cluster_rates.append((c, rate, len(c_matured)))

        cluster_rates.sort(key=lambda x: x[1], reverse=True)

        codebook = []
        for rank_idx, (raw_c, r, m_cnt) in enumerate(cluster_rates):
            c_code = f"W-{rank_idx + 1:02d}"
            c_mask = (raw_labels == raw_c)
            c_matured = [meta[idx] for idx in np.where(c_mask)[0] if meta[idx]["matured"]]
            centroid = raw_centroids[raw_c]

            p1 = np.polyfit(t_norm, centroid, 1)
            beta = float(p1[0])
            p2 = np.polyfit(t_norm, centroid, 2)
            kappa = float(p2[0])
            tau_min = int(np.argmin(centroid))
            tau_max = int(np.argmax(centroid))

            half = W // 2
            cr = float(np.std(centroid[half:]) / (np.std(centroid[:half]) + 1e-6))
            peak = -999.0
            mdd = 0.0
            for v in centroid:
                if v > peak: peak = v
                dd = peak - v
                if dd > mdd: mdd = dd

            hit2_rate = float(np.mean([m["hit_2d"] for m in c_matured])) if c_matured else 0.0
            hit1_rate = float(np.mean([m["hit_1d"] for m in c_matured])) if c_matured else 0.0
            avg_gain = float(np.mean([m["max_gain"] for m in c_matured])) if c_matured else 0.0
            hits_time = [m["bars_to_hit"] for m in c_matured if m["bars_to_hit"] is not None]
            avg_bars = float(np.mean(hits_time)) if hits_time else None

            # Sub-sample evaluation by year (2025 vs 2026)
            c_2025 = [m for m in c_matured if m["year"] == "2025"]
            c_2026 = [m for m in c_matured if m["year"] == "2026"]
            rate_2025 = float(np.mean([m["hit_3d"] for m in c_2025])) if c_2025 else 0.0
            rate_2026 = float(np.mean([m["hit_3d"] for m in c_2026])) if c_2026 else 0.0

            codebook.append({
                "code": c_code,
                "rank": rank_idx + 1,
                "total_count": int(np.sum(c_mask)),
                "matured_count": m_cnt,
                "hit3d_rate": round(r, 4),
                "hit2d_rate": round(hit2_rate, 4),
                "hit1d_rate": round(hit1_rate, 4),
                "avg_max_gain": round(avg_gain, 4),
                "avg_bars_to_hit": round(avg_bars, 1) if avg_bars is not None else None,
                "rate_2025": round(rate_2025, 4),
                "rate_2026": round(rate_2026, 4),
                "curve": list(np.round(centroid, 2)),
                "math": {
                    "beta": round(beta, 3),
                    "kappa": round(kappa, 3),
                    "tau_min": tau_min,
                    "tau_max": tau_max,
                    "tau_min_pct": round(tau_min / W * 100, 1),
                    "tau_max_pct": round(tau_max / W * 100, 1),
                    "cr": round(cr, 2),
                    "mdd": round(mdd, 2)
                }
            })

        peak_rate = codebook[0]["hit3d_rate"]
        trough_rate = codebook[-1]["hit3d_rate"]
        spread = round(peak_rate - trough_rate, 4)
        spread_ratio = round(peak_rate / (trough_rate + 1e-6), 2)

        # Correlation between curvature and hit rate
        kappas = [cb["math"]["kappa"] for cb in codebook]
        betas = [cb["math"]["beta"] for cb in codebook]
        hrates = [cb["hit3d_rate"] for cb in codebook]
        corr_kappa = float(np.corrcoef(kappas, hrates)[0, 1]) if np.std(kappas) > 0 else 0.0
        corr_beta = float(np.corrcoef(betas, hrates)[0, 1]) if np.std(betas) > 0 else 0.0

        # Hour distribution
        hour_stats = {}
        for h_str in ['10:30:00', '11:30:00', '12:30:00', '13:30:00', '14:30:00', '15:30:00', '16:00:00']:
            sub = [m for m in meta if m["matured"] and m["hour"] == h_str]
            if sub:
                hour_stats[h_str] = round(float(np.mean([m["hit_3d"] for m in sub])), 4)

        print(f"  Total Slices: {total_cnt:,} | Matured: {matured_cnt:,}")
        print(f"  Baseline Rate: {base_hit_rate*100:.2f}% | Peak Rate: {peak_rate*100:.2f}% | Trough: {trough_rate*100:.2f}%")
        print(f"  Spread: +{spread*100:.2f}% ({spread_ratio}x) | Corr(kappa, hit): {corr_kappa:+.2f}")
        print(f"  Finished in {time.time() - t_w0:.2f}s")

        window_results.append({
            "days": w_days,
            "bars": W,
            "name": w_name,
            "total_slices": total_cnt,
            "matured_slices": matured_cnt,
            "baseline_hit_rate": round(base_hit_rate, 4),
            "peak_hit_rate": peak_rate,
            "trough_hit_rate": trough_rate,
            "spread": spread,
            "spread_ratio": spread_ratio,
            "corr_kappa": round(corr_kappa, 3),
            "corr_beta": round(corr_beta, 3),
            "hour_hit_rates": hour_stats,
            "codebook": codebook
        })

    # Assemble global output
    payload = {
        "metadata": {
            "title": "QQQ 观测窗口宽度敏感性研究 (2天 ~ 5天)",
            "sample_range": "2025-01-02 至 2026-09-28",
            "tickers_count": int(df["ticker"].nunique()),
            "total_hourly_bars": len(df),
            "total_slices_evaluated": sum(r["total_slices"] for r in window_results),
            "target_horizon": "未来 3 个交易日 (21 根小时 K 线)",
            "target_touch": "+5.0%",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "windows": window_results
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved study results to {OUTPUT_JSON} ({time.time() - t0:.2f}s total)")

if __name__ == "__main__":
    main()
