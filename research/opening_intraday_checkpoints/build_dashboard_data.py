#!/usr/bin/env python3
"""Build aggregated JSON data for multi-column (3D, 5D, 10D) and multi-row per date table."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINTS_PARQUET = BASE_DIR / "checkpoints_dataset.parquet"
PREDICTIONS_PATH = RESULTS_DIR / "predictions_oos_multihorizon.parquet"
OUTPUT_JSON = RESULTS_DIR / "dashboard_data.json"


def main():
    print("Building multi-column table dashboard data...")

    # Load 5m checkpoint dataset
    df_raw = pd.read_parquet(CHECKPOINTS_PARQUET)
    df_cp10 = df_raw[(df_raw["checkpoint_min"] == 10) & (df_raw["session_date"] >= "2026-07-01")].copy()

    # Load predictions
    preds_df = pd.read_parquet(PREDICTIONS_PATH)
    p_3d = preds_df[(preds_df["horizon_days"] == 3) & (preds_df["checkpoint_min"] == 10)][["symbol", "session_date", "pred_prob"]]

    df_merged = pd.merge(df_cp10, p_3d, on=["symbol", "session_date"])
    df_merged["daily_rank"] = df_merged.groupby("session_date")["pred_prob"].rank(ascending=False, method="first")

    # 1. Summary KPIs for 1d (5%), 3d (5%), 5d (10%), 10d (10%)
    if "touch_1d" not in df_merged.columns:
        df_merged["touch_1d"] = (df_merged["forward_mfe_1d"] >= 0.05).astype(int)
    if "touch_3d" not in df_merged.columns:
        df_merged["touch_3d"] = (df_merged["forward_mfe_3d"] >= 0.05).astype(int)
    if "touch_5d" not in df_merged.columns:
        df_merged["touch_5d"] = (df_merged["forward_mfe_5d"] >= 0.10).astype(int)
    if "touch_10d" not in df_merged.columns:
        df_merged["touch_10d"] = (df_merged["forward_mfe_10d"] >= 0.10).astype(int)

    sub_1d = df_merged.dropna(subset=["touch_1d"])
    sub_3d = df_merged.dropna(subset=["touch_3d"])
    sub_5d = df_merged.dropna(subset=["touch_5d"])
    sub_10d = df_merged.dropna(subset=["touch_10d"])

    t1_1d = sub_1d[sub_1d["daily_rank"] == 1]
    t1_3d = sub_3d[sub_3d["daily_rank"] == 1]
    t1_5d = sub_5d[sub_5d["daily_rank"] == 1]
    t1_10d = sub_10d[sub_10d["daily_rank"] == 1]

    t3_1d = sub_1d[sub_1d["daily_rank"] <= 3]
    t3_3d = sub_3d[sub_3d["daily_rank"] <= 3]
    t3_5d = sub_5d[sub_5d["daily_rank"] <= 3]
    t3_10d = sub_10d[sub_10d["daily_rank"] <= 3]

    t5_1d = sub_1d[sub_1d["daily_rank"] <= 5]
    t5_3d = sub_3d[sub_3d["daily_rank"] <= 5]
    t5_5d = sub_5d[sub_5d["daily_rank"] <= 5]
    t5_10d = sub_10d[sub_10d["daily_rank"] <= 5]

    kpis = {
        "top1_touch": {
            "1d": float(t1_1d["touch_1d"].mean()),
            "3d": float(t1_3d["touch_3d"].mean()),
            "5d": float(t1_5d["touch_5d"].mean()),
            "10d": float(t1_10d["touch_10d"].mean()),
        },
        "top1_mfe": {
            "1d": float(t1_1d["forward_mfe_1d"].mean()),
            "3d": float(t1_3d["forward_mfe_3d"].mean()),
            "5d": float(t1_5d["forward_mfe_5d"].mean()),
            "10d": float(t1_10d["forward_mfe_10d"].mean()),
        },
        "top3_any_hit": {
            "1d": float(t3_1d.groupby("session_date")["touch_1d"].max().mean()),
            "3d": float(t3_3d.groupby("session_date")["touch_3d"].max().mean()),
            "5d": float(t3_5d.groupby("session_date")["touch_5d"].max().mean()),
            "10d": float(t3_10d.groupby("session_date")["touch_10d"].max().mean()),
        },
        "top5_any_hit": {
            "1d": float(t5_1d.groupby("session_date")["touch_1d"].max().mean()),
            "3d": float(t5_3d.groupby("session_date")["touch_3d"].max().mean()),
            "5d": float(t5_5d.groupby("session_date")["touch_5d"].max().mean()),
            "10d": float(t5_10d.groupby("session_date")["touch_10d"].max().mean()),
        },
    }

    # 2. Score Calibration Scale
    bins = [0.0, 0.50, 0.60, 0.70, 0.80, 1.0]
    labels = ["<50%", "50-60%", "60-70%", "70-80%", ">=80%"]
    df_merged["score_bin"] = pd.cut(df_merged["pred_prob"], bins=bins, labels=labels)

    scale_list = []
    for lbl, g in df_merged.groupby("score_bin", observed=False):
        tot = len(g)
        if tot == 0:
            continue
        scale_list.append({
            "bin": str(lbl),
            "count": tot,
            "touch_1d": float(g["touch_1d"].mean()) if g["touch_1d"].count() > 0 else 0.0,
            "mfe_1d": float(g["forward_mfe_1d"].mean()) if g["forward_mfe_1d"].count() > 0 else 0.0,
            "touch_3d": float(g["touch_3d"].mean()) if g["touch_3d"].count() > 0 else 0.0,
            "mfe_3d": float(g["forward_mfe_3d"].mean()) if g["forward_mfe_3d"].count() > 0 else 0.0,
            "touch_5d": float(g["touch_5d"].mean()) if g["touch_5d"].count() > 0 else 0.0,
            "mfe_5d": float(g["forward_mfe_5d"].mean()) if g["forward_mfe_5d"].count() > 0 else 0.0,
            "touch_10d": float(g["touch_10d"].mean()) if g["touch_10d"].count() > 0 else 0.0,
            "mfe_10d": float(g["forward_mfe_10d"].mean()) if g["forward_mfe_10d"].count() > 0 else 0.0,
        })

    # 3. Daily Multi-Row Table Data
    dates = sorted(df_merged["session_date"].unique(), reverse=True)
    daily_records = []

    for d in dates:
        is_latest_trading_day = (d == dates[0])
        day_df = df_merged[df_merged["session_date"] == d].sort_values("daily_rank").head(5)
        stocks = []
        for _, r in day_df.iterrows():
            stocks.append({
                "rank": int(r["daily_rank"]),
                "symbol": str(r["symbol"]),
                "entry_price": round(float(r["entry_price"]), 2),
                "score": round(float(r["pred_prob"]), 3),
                # 1D Column (Target: >= +5%)
                "mfe_1d": round(float(r["forward_mfe_1d"]), 4) if pd.notnull(r["forward_mfe_1d"]) else None,
                "touch_1d": int(r["touch_1d"]) if pd.notnull(r["touch_1d"]) else None,
                "is_matured_1d": 0 if is_latest_trading_day else 1,
                # 3D Column (Target: >= +5%)
                "mfe_3d": round(float(r["forward_mfe_3d"]), 4) if pd.notnull(r["forward_mfe_3d"]) else None,
                "touch_3d": int(r["touch_3d"]) if pd.notnull(r["touch_3d"]) else None,
                "is_matured_3d": int(r["is_matured_3d"]) if "is_matured_3d" in r and pd.notnull(r["is_matured_3d"]) else (1 if pd.notnull(r["forward_mfe_3d"]) else 0),
                # 5D Column (Target: >= +10%)
                "mfe_5d": round(float(r["forward_mfe_5d"]), 4) if pd.notnull(r["forward_mfe_5d"]) else None,
                "touch_5d": int(r["touch_5d"]) if pd.notnull(r["touch_5d"]) else None,
                "is_matured_5d": int(r["is_matured_5d"]) if "is_matured_5d" in r and pd.notnull(r["is_matured_5d"]) else (1 if pd.notnull(r["forward_mfe_5d"]) else 0),
                # 10D Column (Target: >= +10%)
                "mfe_10d": round(float(r["forward_mfe_10d"]), 4) if pd.notnull(r["forward_mfe_10d"]) else None,
                "touch_10d": int(r["touch_10d"]) if pd.notnull(r["touch_10d"]) else None,
                "is_matured_10d": int(r["is_matured_10d"]) if "is_matured_10d" in r and pd.notnull(r["is_matured_10d"]) else (1 if pd.notnull(r["forward_mfe_10d"]) else 0),
                # Micro Tags
                "atr_pct": round(float(r["atr_20d_pct"]) * 100, 1),
                "qqq_vwap_dev": round(float(r["qqq_vwap_dev"]) * 100, 2),
                "bar_range": round(float(r["intra_curr_bar_range"]) * 100, 2),
            })
        daily_records.append({
            "date": str(d),
            "stocks": stocks,
        })

    payload = {
        "generated_at": "2026-09-30",
        "checkpoint": "T=10m (09:40 ET)",
        "kpis": kpis,
        "scale": scale_list,
        "dates_data": daily_records,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved dashboard data to {OUTPUT_JSON} ({OUTPUT_JSON.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
