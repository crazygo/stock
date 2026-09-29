#!/usr/bin/env python3
"""
build_motif_codebook.py
Extracts 35-hour (5 trading days x 7 regular hours) rolling waveform slices from 104 QQQ stocks,
normalizes via Z-score, clusters into a codebook of distinct wave codes (W-01 ~ W-12),
computes mathematical invariants, forward 3-day +5% touch hit rates,
and precomputes cross-stock similarity mappings.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONSTITUENTS_PATH = os.path.join(ROOT_DIR, "analysis/qqq_constituents.json")
MARKET_DATA_DIR = os.path.join(ROOT_DIR, "market_data/us_60m")
OUTPUT_JSON_PATH = os.path.join(os.path.dirname(__file__), "data.json")

REGULAR_HOURS = ['10:30:00', '11:30:00', '12:30:00', '13:30:00', '14:30:00', '15:30:00', '16:00:00']
WINDOW_BARS = 35       # 5 trading days * 7 regular bars/day
FORWARD_BARS = 21      # 3 trading days * 7 regular bars/day
FORWARD_TARGET = 0.05  # +5.0% touch target
K_CLUSTERS = 12

def main():
    start_time = time.time()
    print("=" * 70)
    print("QQQ HOURLY MOTIF CODEBOOK GENERATOR (N=5 Days / 35 Bars)")
    print("=" * 70)

    # 1. Load constituents
    with open(CONSTITUENTS_PATH, "r", encoding="utf-8") as f:
        qqq_data = json.load(f)
    tickers = qqq_data["tickers"]
    print(f"Loaded {len(tickers)} QQQ constituent tickers.")

    # 2. Extract slices from each stock
    all_slices = []
    meta_list = []
    stock_raw_data = {}

    for ticker in tickers:
        parquet_path = os.path.join(MARKET_DATA_DIR, ticker, "2026.parquet")
        if not os.path.exists(parquet_path):
            continue

        df = pd.read_parquet(parquet_path)
        df["time"] = df["time_key"].str.split(" ").str[1]
        df["date"] = df["time_key"].str.split(" ").str[0]
        df_reg = df[df["time"].isin(REGULAR_HOURS)].sort_values("time_key").reset_index(drop=True)

        if len(df_reg) < WINDOW_BARS + 1:
            continue

        closes = df_reg["close"].values
        highs = df_reg["high"].values
        lows = df_reg["low"].values
        times = df_reg["time_key"].values
        names = df_reg["name"].values if "name" in df_reg.columns else [ticker] * len(df_reg)

        stock_raw_data[ticker] = {
            "name": str(names[0]),
            "closes": closes,
            "highs": highs,
            "lows": lows,
            "times": times,
            "len": len(closes)
        }

        # Rolling window
        for i in range(WINDOW_BARS - 1, len(df_reg)):
            end_time = str(times[i])
            end_date = end_time.split(" ")[0]
            # Focus on slices ending in recent 30 trading days (2026-08-14 to 2026-09-25)
            if end_date < "2026-08-14":
                continue

            w_close = closes[i - WINDOW_BARS + 1 : i + 1]
            std_c = np.std(w_close)
            if std_c < 1e-8:
                continue
            mean_c = np.mean(w_close)
            z = (w_close - mean_c) / std_c

            future_len = len(closes) - 1 - i
            matured = (future_len >= FORWARD_BARS)

            p_end = float(closes[i])
            if matured:
                fut_h = highs[i + 1 : i + FORWARD_BARS + 1]
                fut_c = closes[i + 1 : i + FORWARD_BARS + 1]
                max_gain = float((np.max(fut_h) - p_end) / p_end)
                hit_3d = int(max_gain >= FORWARD_TARGET)
                hit_2d = int(np.max(fut_h[:14]) >= p_end * (1.0 + FORWARD_TARGET))
                hit_1d = int(np.max(fut_h[:7]) >= p_end * (1.0 + FORWARD_TARGET))

                touch_arr = np.where((fut_h - p_end) / p_end >= FORWARD_TARGET)[0]
                bars_to_hit = int(touch_arr[0] + 1) if len(touch_arr) > 0 else None
                fut_z = list(np.round((fut_c - mean_c) / std_c, 2))
            else:
                fut_h = highs[i + 1 :] if future_len > 0 else np.array([])
                fut_c = closes[i + 1 :] if future_len > 0 else np.array([])
                max_gain = float((np.max(fut_h) - p_end) / p_end) if len(fut_h) > 0 else 0.0
                hit_3d = int(max_gain >= FORWARD_TARGET)
                hit_2d = int(np.max(fut_h[:14]) >= p_end * (1.0 + FORWARD_TARGET)) if len(fut_h) >= 14 else 0
                hit_1d = int(np.max(fut_h[:7]) >= p_end * (1.0 + FORWARD_TARGET)) if len(fut_h) >= 7 else 0
                bars_to_hit = None
                fut_z = list(np.round((fut_c - mean_c) / std_c, 2)) if len(fut_c) > 0 else []

            all_slices.append(z)
            meta_list.append({
                "ticker": ticker,
                "name": str(names[0]),
                "start_time": str(times[i - WINDOW_BARS + 1]),
                "end_time": end_time,
                "end_price": round(p_end, 2),
                "matured": bool(matured),
                "hit_3d_5pct": hit_3d,
                "hit_2d_5pct": hit_2d,
                "hit_1d_5pct": hit_1d,
                "max_gain": round(max_gain, 4),
                "bars_to_hit": bars_to_hit,
                "z_curve": list(np.round(z, 2)),
                "fut_z": fut_z
            })

    X = np.array(all_slices)
    total_slices = len(X)
    print(f"Extracted {total_slices} rolling slices (shape: {X.shape}).")

    # 3. K-Means clustering
    print(f"Running K-Means (K={K_CLUSTERS})...")
    km = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
    raw_labels = km.fit_predict(X)
    raw_centroids = km.cluster_centers_

    # 4. Rank clusters by 3-day +5% hit rate descending
    rank_metrics = []
    for c in range(K_CLUSTERS):
        c_mask = (raw_labels == c)
        c_matured = [meta_list[idx] for idx in np.where(c_mask)[0] if meta_list[idx]["matured"]]
        hit_rate = float(np.mean([m["hit_3d_5pct"] for m in c_matured])) if c_matured else 0.0
        rank_metrics.append((c, hit_rate))

    rank_metrics.sort(key=lambda x: x[1], reverse=True)
    # Mapping: raw_c -> new_code W-01..W-12
    code_map = {}
    for rank_idx, (raw_c, r) in enumerate(rank_metrics):
        code_name = f"W-{rank_idx + 1:02d}"
        code_map[raw_c] = code_name

    # Assign new codes to each slice
    for idx, raw_c in enumerate(raw_labels):
        meta_list[idx]["wave_code"] = code_map[raw_c]

    # 5. Build Codebook Definition
    t_axis = np.arange(WINDOW_BARS)
    t_norm = (t_axis - 17.0) / 10.0

    codebook = []
    print("\n" + "=" * 90)
    print("CODEBOOK DEFINITIONS & EMPIRICAL HIT RATES (NO SEMANTIC NAMES)")
    print("=" * 90)
    print("Code | Total | Matured | Hit3d% | Hit2d% | Hit1d% | AvgGain% | Slope(beta) | Curv(kappa) | TauMin | TauMax | CR   | MDD")
    print("-" * 115)

    sorted_raw_clusters = [x[0] for x in rank_metrics]
    
    for rank_idx, raw_c in enumerate(sorted_raw_clusters):
        code_name = f"W-{rank_idx + 1:02d}"
        c_indices = np.where(raw_labels == raw_c)[0]
        c_mask = (raw_labels == raw_c)
        c_matured = [meta_list[idx] for idx in c_indices if meta_list[idx]["matured"]]

        centroid = raw_centroids[raw_c]
        c_curve = list(np.round(centroid, 2))

        # Mathematical invariants
        p1 = np.polyfit(t_norm, centroid, 1)
        beta = float(p1[0])
        p2 = np.polyfit(t_norm, centroid, 2)
        kappa = float(p2[0])
        tau_min = int(np.argmin(centroid))
        tau_max = int(np.argmax(centroid))
        cr = float(np.std(centroid[17:]) / (np.std(centroid[:17]) + 1e-6))

        peak = -999.0
        mdd = 0.0
        for v in centroid:
            if v > peak:
                peak = v
            dd = peak - v
            if dd > mdd:
                mdd = dd

        # Stats
        total_cnt = int(np.sum(c_mask))
        mat_cnt = len(c_matured)
        hit3d_rate = float(np.mean([m["hit_3d_5pct"] for m in c_matured])) if c_matured else 0.0
        hit2d_rate = float(np.mean([m["hit_2d_5pct"] for m in c_matured])) if c_matured else 0.0
        hit1d_rate = float(np.mean([m["hit_1d_5pct"] for m in c_matured])) if c_matured else 0.0
        avg_gain = float(np.mean([m["max_gain"] for m in c_matured])) if c_matured else 0.0
        bars_to_hit_vals = [m["bars_to_hit"] for m in c_matured if m["bars_to_hit"] is not None]
        avg_bars_to_hit = float(np.mean(bars_to_hit_vals)) if bars_to_hit_vals else None

        print(f"{code_name} | {total_cnt:5d} | {mat_cnt:7d} | {hit3d_rate*100:6.1f}% | {hit2d_rate*100:6.1f}% | {hit1d_rate*100:6.1f}% | {avg_gain*100:7.2f}% | {beta:+11.3f} | {kappa:+11.3f} | {tau_min:6d} | {tau_max:6d} | {cr:4.2f} | {mdd:4.2f}")

        # Top 12 diverse representative matured historical samples
        # Sort by distance to centroid
        c_distances = np.linalg.norm(X[c_indices] - centroid, axis=1)
        sorted_pairs = sorted(zip(c_indices, c_distances), key=lambda x: x[1])

        top_samples = []
        seen_tickers = set()
        for idx_val, dist_val in sorted_pairs:
            m = meta_list[idx_val]
            if not m["matured"]:
                continue
            if m["ticker"] in seen_tickers and len(seen_tickers) < 10:
                continue
            seen_tickers.add(m["ticker"])
            top_samples.append({
                "ticker": m["ticker"],
                "name": m["name"],
                "start_time": m["start_time"],
                "end_time": m["end_time"],
                "end_price": m["end_price"],
                "distance": round(float(dist_val), 3),
                "hit_3d_5pct": m["hit_3d_5pct"],
                "hit_2d_5pct": m["hit_2d_5pct"],
                "hit_1d_5pct": m["hit_1d_5pct"],
                "max_gain": m["max_gain"],
                "bars_to_hit": m["bars_to_hit"],
                "z_curve": m["z_curve"],
                "fut_z": m["fut_z"]
            })
            if len(top_samples) >= 12:
                break

        codebook.append({
            "code": code_name,
            "rank": rank_idx + 1,
            "curve": c_curve,
            "math": {
                "beta": round(beta, 3),
                "kappa": round(kappa, 3),
                "tau_min": tau_min,
                "tau_max": tau_max,
                "cr": round(cr, 2),
                "mdd": round(mdd, 2)
            },
            "stats": {
                "total_count": total_cnt,
                "matured_count": mat_cnt,
                "hit3d_rate": round(hit3d_rate, 4),
                "hit2d_rate": round(hit2d_rate, 4),
                "hit1d_rate": round(hit1d_rate, 4),
                "avg_max_gain": round(avg_gain, 4),
                "avg_bars_to_hit": round(avg_bars_to_hit, 1) if avg_bars_to_hit is not None else None
            },
            "representative_samples": top_samples
        })

    # 6. Latest 5-day state for each of the 104 stocks
    stock_latest_map = {}
    print("\nComputing latest state and cross-stock matches for 104 stocks...")

    # We also keep an index of matured slices by wave_code for fast cross-stock matching
    slices_by_code = {cb["code"]: [] for cb in codebook}
    for m in meta_list:
        if m["matured"]:
            slices_by_code[m["wave_code"]].append(m)

    for ticker, sdata in stock_raw_data.items():
        closes = sdata["closes"]
        times = sdata["times"]
        highs = sdata["highs"]
        L = len(closes)
        if L < WINDOW_BARS:
            continue

        # Latest 35 bars
        latest_c = closes[-WINDOW_BARS:]
        latest_t = times[-WINDOW_BARS:]
        mean_c = np.mean(latest_c)
        std_c = np.std(latest_c)
        if std_c < 1e-8:
            continue
        latest_z = (latest_c - mean_c) / std_c

        # Find closest cluster centroid
        min_dist = 999.0
        best_code = None
        best_cb = None
        for cb in codebook:
            c_curve = np.array(cb["curve"])
            dist = float(np.linalg.norm(latest_z - c_curve))
            if dist < min_dist:
                min_dist = dist
                best_code = cb["code"]
                best_cb = cb

        # Mathematical properties of this stock's latest curve
        p1 = np.polyfit(t_norm, latest_z, 1)
        p2 = np.polyfit(t_norm, latest_z, 2)
        local_beta = float(p1[0])
        local_kappa = float(p2[0])
        local_tau_min = int(np.argmin(latest_z))
        local_tau_max = int(np.argmax(latest_z))

        # Cross-stock matching: find top 6 closest historical slices in OTHER stocks
        # Search among matured slices in the same code (or overall)
        candidate_pool = slices_by_code[best_code]
        # Filter out slices from the same ticker
        other_candidates = [m for m in candidate_pool if m["ticker"] != ticker]
        if len(other_candidates) < 6:
            # expand to all matured slices
            other_candidates = [m for m in meta_list if m["matured"] and m["ticker"] != ticker]

        # Calculate distance to latest_z
        cand_dists = []
        for m in other_candidates:
            d = float(np.linalg.norm(latest_z - np.array(m["z_curve"])))
            cand_dists.append((m, d))

        cand_dists.sort(key=lambda x: x[1])
        top_cross_matches = []
        seen_cross_tickers = set()
        for cand_m, cand_d in cand_dists:
            if cand_m["ticker"] in seen_cross_tickers:
                continue
            seen_cross_tickers.add(cand_m["ticker"])
            top_cross_matches.append({
                "ticker": cand_m["ticker"],
                "name": cand_m["name"],
                "start_time": cand_m["start_time"],
                "end_time": cand_m["end_time"],
                "end_price": cand_m["end_price"],
                "distance": round(cand_d, 3),
                "matched_code": cand_m["wave_code"],
                "hit_3d_5pct": cand_m["hit_3d_5pct"],
                "hit_2d_5pct": cand_m["hit_2d_5pct"],
                "hit_1d_5pct": cand_m["hit_1d_5pct"],
                "max_gain": cand_m["max_gain"],
                "bars_to_hit": cand_m["bars_to_hit"],
                "z_curve": cand_m["z_curve"],
                "fut_z": cand_m["fut_z"]
            })
            if len(top_cross_matches) >= 6:
                break

        stock_latest_map[ticker] = {
            "ticker": ticker,
            "name": sdata["name"],
            "latest_time": str(latest_t[-1]),
            "latest_price": round(float(latest_c[-1]), 2),
            "matched_code": best_code,
            "distance": round(min_dist, 3),
            "code_hit3d_rate": best_cb["stats"]["hit3d_rate"],
            "code_avg_gain": best_cb["stats"]["avg_max_gain"],
            "z_curve": list(np.round(latest_z, 2)),
            "raw_prices": [round(float(p), 2) for p in latest_c],
            "raw_times": [str(t) for t in latest_t],
            "math": {
                "beta": round(local_beta, 3),
                "kappa": round(local_kappa, 3),
                "tau_min": local_tau_min,
                "tau_max": local_tau_max
            },
            "cross_matches": top_cross_matches
        }

    # 7. Assemble full payload
    payload = {
        "metadata": {
            "title": "QQQ Constituents Hourly Motif Codebook (N=5 Trading Days)",
            "n_days": 5,
            "window_bars": WINDOW_BARS,
            "forward_bars": FORWARD_BARS,
            "target_touch": "+5.0%",
            "k_clusters": K_CLUSTERS,
            "total_slices": total_slices,
            "matured_slices": sum(1 for m in meta_list if m["matured"]),
            "pending_slices": sum(1 for m in meta_list if not m["matured"]),
            "tickers_count": len(stock_raw_data),
            "date_range": "2026-08-14 to 2026-09-25",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "codebook": codebook,
        "stocks": stock_latest_map
    }

    # Write output JSON
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(OUTPUT_JSON_PATH) / (1024 * 1024)
    print(f"\nGenerated {OUTPUT_JSON_PATH}")
    print(f"File size: {file_size_mb:.2f} MB in {elapsed:.2f}s")
    print("Done!")

if __name__ == "__main__":
    main()
