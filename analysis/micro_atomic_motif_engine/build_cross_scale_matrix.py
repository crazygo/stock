#!/usr/bin/env python3
"""
build_cross_scale_matrix.py
Computes the empirical Cross-Scale Joint Distribution Matrix:
Macro Waveforms (W-01 ~ W-12, 5-day / 35-hour 60m scale)
x Micro Atoms (A-01 ~ A-12, 2-hour / 24-bar 5m scale)
across all 104 QQQ constituent stocks.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MACRO_JSON_PATH = os.path.join(ROOT_DIR, "analysis/qqq_hourly_motif_codebook/data.json")
MICRO_JSON_PATH = os.path.join(ROOT_DIR, "analysis/micro_atomic_motif_engine/data.json")
QQQ_PATH = os.path.join(ROOT_DIR, "analysis/qqq_constituents.json")
OUTPUT_JSON_PATH = os.path.join(os.path.dirname(__file__), "cross_scale_matrix.json")

REGULAR_HOURS = ['10:30:00', '11:30:00', '12:30:00', '13:30:00', '14:30:00', '15:30:00', '16:00:00']

def main():
    t0 = time.time()
    print("=" * 70)
    print("COMPUTING MACRO (W-01~W-12) x MICRO (A-01~A-12) CROSS-SCALE MATRIX")
    print("=" * 70)

    # 1. Load macro codebook
    with open(MACRO_JSON_PATH, "r", encoding="utf-8") as f:
        macro_data = json.load(f)
    macro_codebook = macro_data["codebook"]
    macro_centroids = np.array([c["curve"] for c in macro_codebook])
    macro_codes = [c["code"] for c in macro_codebook]

    # 2. Load micro codebook
    with open(MICRO_JSON_PATH, "r", encoding="utf-8") as f:
        micro_data = json.load(f)
    micro_codebook = micro_data["codebook"]
    micro_centroids = np.array([c["curve"] for c in micro_codebook])
    micro_codes = [c["code"] for c in micro_codebook]

    # 3. Load QQQ tickers
    with open(QQQ_PATH, "r", encoding="utf-8") as f:
        qqq_tickers = json.load(f)["tickers"]

    print(f"Loaded {len(macro_codes)} Macro Waveforms, {len(micro_codes)} Micro Atoms, {len(qqq_tickers)} QQQ Tickers.")

    matrix_counts = np.zeros((12, 12), dtype=int)
    matrix_hits = np.zeros((12, 12), dtype=int)
    total_pairs = 0

    for ticker in qqq_tickers:
        p60 = os.path.join(ROOT_DIR, f"market_data/us_60m/{ticker}/2026.parquet")
        p5 = os.path.join(ROOT_DIR, f"market_data/us_5m/{ticker}/2026.parquet")
        if not (os.path.exists(p60) and os.path.exists(p5)):
            continue

        df60 = pd.read_parquet(p60)
        df60["time"] = df60["time_key"].str.split(" ").str[1]
        df60["date"] = df60["time_key"].str.split(" ").str[0]
        df60 = df60[df60["time"].isin(REGULAR_HOURS)].sort_values("time_key").reset_index(drop=True)
        df5 = pd.read_parquet(p5).sort_values("time_key").reset_index(drop=True)

        c60 = df60["close"].values
        h60 = df60["high"].values
        t60 = df60["time_key"].values
        c5 = df5["close"].values
        t5 = df5["time_key"].values
        t5_to_idx = {t: i for i, t in enumerate(t5)}

        L60 = len(df60)
        if L60 < 56:  # need 35 window + 21 forward
            continue

        for i in range(34, L60 - 21):
            end_t = t60[i]
            sub60 = c60[i - 34 : i + 1]
            std60 = np.std(sub60)
            if std60 < 1e-8:
                continue
            z60 = (sub60 - np.mean(sub60)) / std60
            m_idx = int(np.argmin(np.linalg.norm(macro_centroids - z60, axis=1)))

            # 3-day (21 hours) forward touch +5%
            p_curr = c60[i]
            hit = int(np.max(h60[i + 1 : i + 22]) >= p_curr * 1.05)

            # Match terminal 2-hour 5m slice ending at end_t
            if end_t in t5_to_idx:
                idx5 = t5_to_idx[end_t]
                if idx5 >= 23:
                    sub5 = c5[idx5 - 23 : idx5 + 1]
                    std5 = np.std(sub5)
                    if std5 > 1e-8:
                        z5 = (sub5 - np.mean(sub5)) / std5
                        a_idx = int(np.argmin(np.linalg.norm(micro_centroids - z5, axis=1)))
                        matrix_counts[m_idx, a_idx] += 1
                        if hit:
                            matrix_hits[m_idx, a_idx] += 1
                        total_pairs += 1

    print(f"Computed {total_pairs} pairs across 104 QQQ stocks in {time.time()-t0:.2f}s.")

    # Build rich profiles
    macro_profiles = []
    for m_idx in range(12):
        m_code = macro_codes[m_idx]
        m_cb = macro_codebook[m_idx]
        tot = int(np.sum(matrix_counts[m_idx]))
        hits = int(np.sum(matrix_hits[m_idx]))
        base_rate = round(float(hits / max(1, tot)), 4)

        row_counts = matrix_counts[m_idx].tolist()
        row_hits = matrix_hits[m_idx].tolist()
        row_rates = [round(float(row_hits[a] / max(1, row_counts[a])), 4) for a in range(12)]
        row_pcts = [round(float(row_counts[a] / max(1, tot) * 100.0), 1) for a in range(12)]

        # Top frequent micro atoms
        sorted_by_freq = sorted(range(12), key=lambda a: row_counts[a], reverse=True)
        # Top hit rate micro atoms (with at least 50 samples)
        valid_a = [a for a in range(12) if row_counts[a] >= 50]
        sorted_by_rate = sorted(valid_a, key=lambda a: row_rates[a], reverse=True)

        macro_profiles.append({
            "macro_code": m_code,
            "macro_rank": m_idx + 1,
            "curve": m_cb["curve"],
            "total_count": tot,
            "matured_hits": hits,
            "base_hit3d_rate": base_rate,
            "row_counts": row_counts,
            "row_hits": row_hits,
            "row_hit_rates": row_rates,
            "row_freq_pcts": row_pcts,
            "top_frequent_atoms": [
                {
                    "atom": micro_codes[a],
                    "desc": micro_codebook[a]["descriptor"],
                    "count": row_counts[a],
                    "freq_pct": row_pcts[a],
                    "hit3d_rate": row_rates[a]
                }
                for a in sorted_by_freq[:4]
            ],
            "top_hit_atoms": [
                {
                    "atom": micro_codes[a],
                    "desc": micro_codebook[a]["descriptor"],
                    "count": row_counts[a],
                    "freq_pct": row_pcts[a],
                    "hit3d_rate": row_rates[a],
                    "excess": round(row_rates[a] - base_rate, 4)
                }
                for a in sorted_by_rate[:3]
            ],
            "worst_hit_atoms": [
                {
                    "atom": micro_codes[a],
                    "desc": micro_codebook[a]["descriptor"],
                    "count": row_counts[a],
                    "freq_pct": row_pcts[a],
                    "hit3d_rate": row_rates[a],
                    "excess": round(row_rates[a] - base_rate, 4)
                }
                for a in sorted_by_rate[-2:]
            ]
        })

    payload = {
        "metadata": {
            "title": "QQQ 纳斯达克 100 宏微双尺度形态交叉分布矩阵 (Macro x Micro)",
            "macro_scale": "5 个交易日 = 35 根 60m K 线 (W-01 ~ W-12)",
            "micro_scale": "2 小时 = 24 根 5m K 线 (A-01 ~ A-12)",
            "sample_stocks": 104,
            "total_pairs": total_pairs,
            "target": "未来 3 个交易日 (21 小时) 触及 +5.0%",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "macro_codes": macro_codes,
        "micro_codes": micro_codes,
        "micro_codebook": [
            {
                "code": c["code"],
                "descriptor": c["descriptor"],
                "curve": c["curve"]
            }
            for c in micro_codebook
        ],
        "macro_profiles": macro_profiles
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved cross-scale matrix to {OUTPUT_JSON_PATH} ({os.path.getsize(OUTPUT_JSON_PATH)/1024:.1f} KB)")

if __name__ == "__main__":
    main()
