#!/usr/bin/env python3
"""
build_micro_atomic_engine.py
Builds the Multi-Scale Hierarchical Micro-Atomic Motif Engine:
  1. Segments each trading day into 2-hour micro atoms using 5-minute K-lines (24 bars each):
     - S1 (09:30~11:30, 24 bars)
     - S2 (11:30~13:30, 24 bars)
     - S3 (13:30~16:00, 30 bars resampled to 24 points)
  2. Z-Score dimensionless normalization into unified R^24 space.
  3. K-Means clustering (K=12) into a dictionary of micro-atoms (A-01 ~ A-12).
  4. Sequence Chaining: Daily triplets (S1, S2, S3) and 2~3 day sequences.
  5. Opening Surge Branching Analysis: Isolates opening spikes and evaluates future 3-day +5% outcomes.
Outputs:
  analysis/micro_atomic_motif_engine/data.json
"""

import os
import sys
import glob
import json
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PARTS_DIR = os.path.join(ROOT_DIR, "market_data/model_training_history_v1/parts")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "data.json")

FORWARD_DAYS = 3
TARGET = 0.05
K_ATOMS = 12

def main():
    t0 = time.time()
    print("=" * 75)
    print("BUILDING MICRO-ATOMIC MOTIF ENGINE (2-HOUR / 24-BAR 5M K-LINES)")
    print("=" * 75)

    tickers = sorted([d for d in os.listdir(PARTS_DIR) if not d.startswith('.')])
    print(f"Discovered {len(tickers)} tickers in history store.")

    all_segments = []
    seg_meta = []
    daily_records = []

    for t_idx, ticker in enumerate(tickers):
        files = sorted(glob.glob(f"{PARTS_DIR}/{ticker}/2025-*/bars.parquet") + glob.glob(f"{PARTS_DIR}/{ticker}/2026-*/bars.parquet"))
        if not files:
            continue

        for f in files:
            df = pd.read_parquet(f, columns=['session_date', 'session_type', 'start_at_et', 'close', 'high', 'low'])
            reg = df[df['session_type'] == 'regular'].copy()
            if reg.empty:
                continue

            for d, grp in reg.groupby('session_date'):
                if len(grp) >= 72:  # valid full trading day
                    c = grp['close'].values
                    h = grp['high'].values
                    l = grp['low'].values

                    s1_c = c[:24]
                    s1_h = h[:24]
                    s1_l = l[:24]

                    s2_c = c[24:48]
                    s2_h = h[24:48]
                    s2_l = l[24:48]

                    s3_raw_c = c[48:]
                    s3_raw_h = h[48:]
                    s3_raw_l = l[48:]

                    # Uniform 24-point resampling for S3
                    s3_c = np.interp(np.linspace(0, len(s3_raw_c)-1, 24), np.arange(len(s3_raw_c)), s3_raw_c)
                    s3_h = np.interp(np.linspace(0, len(s3_raw_h)-1, 24), np.arange(len(s3_raw_h)), s3_raw_h)
                    s3_l = np.interp(np.linspace(0, len(s3_raw_l)-1, 24), np.arange(len(s3_raw_l)), s3_raw_l)

                    day_segs = [
                        (1, s1_c, s1_h, s1_l),
                        (2, s2_c, s2_h, s2_l),
                        (3, s3_c, s3_h, s3_l)
                    ]

                    seg_indices = []
                    for seg_num, sc, sh, sl in day_segs:
                        std_val = float(np.std(sc))
                        mean_val = float(np.mean(sc))
                        if std_val < 1e-8:
                            std_val = 1.0

                        z_curve = (sc - mean_val) / std_val
                        ret = float((sc[-1] - sc[0]) / sc[0]) if sc[0] > 1e-4 else 0.0
                        peak_bar = int(np.argmax(sc))
                        trough_bar = int(np.argmin(sc))

                        # Fade ratio: if peak is in first 16 bars (first 80 min)
                        max_p = float(np.max(sc))
                        min_p = float(np.min(sc))
                        fade_ratio = float((max_p - sc[-1]) / (max_p - min_p + 1e-6)) if peak_bar < 16 else 0.0

                        seg_id = len(all_segments)
                        all_segments.append(z_curve)
                        seg_meta.append({
                            "ticker": ticker,
                            "date": str(d),
                            "seg_num": seg_num,
                            "ret": round(ret, 4),
                            "peak_min": peak_bar * 5,
                            "trough_min": trough_bar * 5,
                            "fade_ratio": round(fade_ratio, 3),
                            "raw_curve": list(np.round(z_curve, 2))
                        })
                        seg_indices.append(seg_id)

                    daily_records.append({
                        "ticker": ticker,
                        "date": str(d),
                        "open_p": float(c[0]),
                        "close_p": float(c[-1]),
                        "max_high": float(np.max(h)),
                        "min_low": float(np.min(l)),
                        "seg_ids": seg_indices
                    })

    X = np.array(all_segments)
    print(f"Extracted {len(X):,} 2-hour 5m micro segments across {len(daily_records):,} daily records in {time.time() - t0:.2f}s")

    # K-Means clustering (K=12)
    print(f"Clustering into K={K_ATOMS} micro-atoms...")
    km = KMeans(n_clusters=K_ATOMS, random_state=42, n_init=5)
    cluster_labels = km.fit_predict(X)
    centroids = km.cluster_centers_

    # Map labels to metadata
    for i, lab in enumerate(cluster_labels):
        seg_meta[i]["atom_id"] = int(lab)

    for d in daily_records:
        d["atoms"] = [seg_meta[sid]["atom_id"] for sid in d["seg_ids"]]

    # Characterize each atomic cluster
    t_norm = (np.arange(24) - 11.5) / 8.0
    codebook = []
    print("\n" + "=" * 90)
    print("2-HOUR 5-MINUTE MICRO-ATOMIC CODEBOOK (K=12)")
    print("=" * 90)
    print("Code | Total | MeanRet% | PeakMin | TroughMin | FadeRatio | Head(Z) | Peak(Z) | End(Z) | 特征几何形态")
    print("-" * 105)

    for c in range(K_ATOMS):
        c_code = f"A-{c+1:02d}"
        c_indices = np.where(cluster_labels == c)[0]
        c_centroid = centroids[c]
        c_curve = list(np.round(c_centroid, 2))

        c_rets = [seg_meta[i]["ret"] for i in c_indices]
        c_peaks = [seg_meta[i]["peak_min"] for i in c_indices]
        c_troughs = [seg_meta[i]["trough_min"] for i in c_indices]
        c_fades = [seg_meta[i]["fade_ratio"] for i in c_indices]

        mean_ret = float(np.mean(c_rets))
        median_peak = int(np.median(c_peaks))
        median_trough = int(np.median(c_troughs))
        mean_fade = float(np.mean(c_fades))

        head_z = float(c_centroid[0])
        peak_z = float(np.max(c_centroid))
        end_z = float(c_centroid[-1])
        peak_bar = int(np.argmax(c_centroid))

        # Qualitative descriptor
        if end_z > 0.8 and peak_bar >= 18:
            desc = "真突破稳健单边攻坚"
        elif peak_bar <= 8 and (peak_z - end_z) > 1.2:
            desc = "开盘暴冲诱多倒V回落"
        elif peak_bar in [10, 11, 12, 13] and end_z > 0.3:
            desc = "脉冲后高位锁筹横盘"
        elif end_z < -0.8 and peak_bar <= 2:
            desc = "开盘即巅峰持续阴跌"
        elif int(np.argmin(c_centroid)) <= 8 and end_z > 0.5:
            desc = "急跌诱空后深V反弹"
        elif np.std(c_centroid) < 0.6:
            desc = "窄幅缩量中枢横盘"
        else:
            desc = "震荡整理过渡波"

        print(f"{c_code} | {len(c_indices):5d} | {mean_ret*100:+7.2f}% | {median_peak:6d}m | {median_trough:8d}m | {mean_fade:8.2f} | {head_z:+6.2f} | {peak_z:+6.2f} | {end_z:+6.2f} | {desc}")

        codebook.append({
            "code": c_code,
            "id": c,
            "total_count": len(c_indices),
            "mean_ret": round(mean_ret, 4),
            "median_peak_min": median_peak,
            "median_trough_min": median_trough,
            "mean_fade_ratio": round(mean_fade, 3),
            "descriptor": desc,
            "curve": c_curve,
            "head_z": round(head_z, 2),
            "peak_z": round(peak_z, 2),
            "end_z": round(end_z, 2)
        })

    # Sequence Chaining & Forward 3-Day Evaluation
    print("\nChaining atoms into sequences and evaluating future 3-day +5% touch hit rates...")
    daily_by_ticker = defaultdict(list)
    for d in daily_records:
        daily_by_ticker[d["ticker"]].append(d)

    # Sort each ticker's days
    for t in daily_by_ticker:
        daily_by_ticker[t].sort(key=lambda x: x["date"])

    # 1-Day Triplets: (A_open -> A_mid -> A_close)
    triplet_stats = defaultdict(list)
    # 2-Day Sequences: 6 atoms (Day1 A1..A3 -> Day2 A1..A3)
    # Opening Branch Analysis: When Day1 Opening is Surge, how does afternoon split?
    opening_branch_stats = defaultdict(list)

    seq_samples = []

    for ticker, days in daily_by_ticker.items():
        L = len(days)
        for i in range(L - FORWARD_DAYS):
            cur = days[i]
            p_entry = cur["close_p"]
            fut_days = days[i+1 : i+1+FORWARD_DAYS]
            max_fut_h = max(f["max_high"] for f in fut_days)
            hit_3d = int((max_fut_h - p_entry) / p_entry >= TARGET)

            # Daily triplet
            a1, a2, a3 = cur["atoms"]
            triplet_key = f"A-{a1+1:02d} -> A-{a2+1:02d} -> A-{a3+1:02d}"
            triplet_stats[triplet_key].append(hit_3d)

            # Opening branch (a1 -> (a2, a3))
            branch_key = (f"A-{a1+1:02d}", f"A-{a2+1:02d}", f"A-{a3+1:02d}")
            opening_branch_stats[branch_key].append(hit_3d)

            # 2-Day chain if i >= 1
            if i >= 1:
                prev = days[i-1]
                chain_2d = f"A-{prev['atoms'][0]+1:02d}->A-{cur['atoms'][0]+1:02d}->A-{cur['atoms'][2]+1:02d}"
                seq_samples.append((chain_2d, hit_3d))

    # Evaluate Opening Surge Branches
    # Find which atoms are "Surge" (high peak in first half)
    surge_atoms = [cb["code"] for cb in codebook if cb["descriptor"] in ["开盘暴冲诱多倒V回落", "真突破稳健单边攻坚", "脉冲后高位锁筹横盘"] or cb["mean_ret"] > 0.008]
    print(f"\nOpening Surge Atoms identified: {surge_atoms}")

    branch_results = []
    for (a_open, a_mid, a_close), hits in opening_branch_stats.items():
        if a_open in surge_atoms and len(hits) >= 20:
            rate = float(np.mean(hits))
            # Categorize afternoon behavior
            mid_cb = next(c for c in codebook if c["code"] == a_mid)
            close_cb = next(c for c in codebook if c["code"] == a_close)
            
            if mid_cb["end_z"] < -0.5 or close_cb["end_z"] < -0.5:
                aft_type = "午后一路跳水走弱 (Dump/Fade)"
            elif mid_cb["end_z"] > 0.5 and close_cb["end_z"] > 0.5:
                aft_type = "高位稳健锁筹抗跌 (High Hold)"
            elif close_cb["end_z"] > 0.8:
                aft_type = "尾盘放量二次突破 (Second Push)"
            else:
                aft_type = "中枢震荡消化 (Chop)"

            branch_results.append({
                "open_atom": a_open,
                "mid_atom": a_mid,
                "close_atom": a_close,
                "sequence": f"{a_open} -> {a_mid} -> {a_close}",
                "afternoon_type": aft_type,
                "sample_count": len(hits),
                "hit3d_rate": round(rate, 4)
            })

    branch_results.sort(key=lambda x: x["hit3d_rate"], reverse=True)
    print("\n=== TOP OPENING SURGE BRANCHES (AFTERNOON BEHAVIOR IMPACT) ===")
    for b in branch_results[:8]:
        print(f"  {b['sequence']:25s} | 类型: {b['afternoon_type']:20s} | 3天+5%达成率: {b['hit3d_rate']*100:.2f}% (样本: {b['sample_count']})")

    # Evaluate Top Daily Triplets (Sample >= 30)
    top_triplets = []
    for trip_seq, hits in triplet_stats.items():
        if len(hits) >= 30:
            rate = float(np.mean(hits))
            top_triplets.append({
                "sequence": trip_seq,
                "sample_count": len(hits),
                "hit3d_rate": round(rate, 4)
            })

    top_triplets.sort(key=lambda x: x["hit3d_rate"], reverse=True)

    # Global Payload
    payload = {
        "metadata": {
            "title": "QQQ 双尺度微观原子波形与时序语法链引擎",
            "micro_resolution": "5分钟 K 线 (2小时 = 24 根)",
            "sample_range": "2025-01-02 至 2026-09-25",
            "tickers_count": len(tickers),
            "total_micro_segments": len(all_segments),
            "total_trading_days": len(daily_records),
            "k_atoms": K_ATOMS,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "codebook": codebook,
        "opening_branches": branch_results,
        "top_daily_triplets": top_triplets
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\nSaved micro-atomic engine payload to {OUTPUT_JSON} ({time.time() - t0:.2f}s total)")

if __name__ == "__main__":
    main()
