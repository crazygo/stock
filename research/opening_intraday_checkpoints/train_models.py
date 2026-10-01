#!/usr/bin/env python3
"""Train isolated LightGBM models across opening checkpoints and multi-day holding horizons (1d, 3d, 5d, 10d).

Walk-Forward Time Series Validation:
- In-Sample Train: 2026-01-02 to 2026-06-30
- Out-of-Sample Test: 2026-07-01 to 2026-09-25

For each Horizon H in {1, 3, 5, 10} days:
Trains separate models for Checkpoints T in {5, 10, 15, 20, 25, 30} min.
Evaluates out-of-sample touch/win rates, precision lift, MFE, and feature importance.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "checkpoints_dataset.parquet"
MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    # Dim 1: Pre-Market Geometry
    "pm_gap",
    "pm_intra_gain",
    "pm_gain",
    "pm_surge_retention",
    "pm_peak_to_open_mins",
    "pm_final_velocity_10m",
    # Dim 2: Pre-Market Liquidity
    "pm_volume",
    "pm_turnover",
    "pm_vwap_dev",
    "pm_rvol",
    # Dim 3: Intraday Opening Trajectory
    "intra_return_from_open",
    "intra_high_from_open",
    "intra_low_from_open",
    "intra_range_pct",
    "intra_range_retention",
    "intra_vwap_dev",
    "intra_upbar_ratio",
    "intra_curr_bar_ret",
    "intra_curr_bar_range",
    "intra_breakout_pm_high",
    "intra_dist_to_pm_high",
    # Dim 4: Opening Order Flow & Volume
    "intra_volume",
    "intra_turnover",
    "intra_rvol",
    "intra_vol_acceleration",
    # Dim 5: Market Benchmark & Alpha
    "qqq_gap",
    "qqq_intra_ret",
    "alpha_vs_qqq",
    "qqq_vwap_dev",
    # Dim 6: Volatility Profile
    "atr_20d_pct",
]

CHECKPOINTS = [5, 10, 15, 20, 25, 30]
HORIZONS = [1, 3, 5, 10]
HORIZON_CONFIG = {
    1: {"target_pct": 0.05, "target_col": "y_hit_1d", "touch_col": "touch_1d", "label": "1D ≥ +5%"},
    3: {"target_pct": 0.05, "target_col": "y_hit_3d", "touch_col": "touch_3d", "label": "3D ≥ +5%"},
    5: {"target_pct": 0.10, "target_col": "y_hit_5d", "touch_col": "touch_5d", "label": "5D ≥ +10%"},
    10: {"target_pct": 0.10, "target_col": "y_hit_10d", "touch_col": "touch_10d", "label": "10D ≥ +10%"},
}
SPLIT_DATE = "2026-07-01"


def train_and_evaluate(
    horizon: int, cp_min: int, df_all: pd.DataFrame
) -> Tuple[Dict, pd.DataFrame]:
    """Train and evaluate model for a specific horizon and checkpoint."""
    cfg = HORIZON_CONFIG[horizon]
    target_col = cfg["target_col"]
    touch_col = cfg["touch_col"]
    target_pct = cfg["target_pct"]
    mfe_col = f"forward_mfe_{horizon}d"
    mae_col = f"forward_mae_{horizon}d"
    close_col = f"forward_close_ret_{horizon}d"

    # Filter checkpoint
    df_sub = df_all[df_all["checkpoint_min"] == cp_min].copy().sort_values("session_date").reset_index(drop=True)

    # Train only on matured rows before SPLIT_DATE
    train_mask = (df_sub["session_date"] < SPLIT_DATE) & (df_sub[target_col].notnull())
    # Test on ALL rows on or after SPLIT_DATE (including recent dates)
    test_mask = df_sub["session_date"] >= SPLIT_DATE

    df_train = df_sub[train_mask]
    df_test = df_sub[test_mask]

    X_train = df_train[FEATURE_COLS].fillna(0.0)
    y_train = df_train[target_col].astype(int)

    X_test = df_test[FEATURE_COLS].fillna(0.0)

    n_pos_train = int(y_train.sum())
    n_neg_train = len(y_train) - n_pos_train
    scale_pos = max(1.0, float(n_neg_train) / max(1, n_pos_train))

    model = lgb.LGBMClassifier(
        n_estimators=180,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=min(scale_pos, 4.0),
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(X_train, y_train)

    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Metrics on matured subset
    matured_test_mask = df_test[target_col].notnull()
    df_test_matured = df_test[matured_test_mask].copy()
    y_test_matured = df_test_matured[target_col].astype(int)
    y_prob_matured = y_prob_test[matured_test_mask.values]

    test_total = len(df_test)
    matured_total = len(df_test_matured)
    matured_pos = int(y_test_matured.sum())
    base_valid_rate = (matured_pos / matured_total) if matured_total > 0 else 0.0
    base_touch_rate = float(df_test_matured[touch_col].mean()) if matured_total > 0 else 0.0

    roc_auc = float(roc_auc_score(y_test_matured, y_prob_matured)) if len(np.unique(y_test_matured)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_test_matured, y_prob_matured)) if len(np.unique(y_test_matured)) > 1 else 0.0
    brier = float(brier_score_loss(y_test_matured, y_prob_matured)) if matured_total > 0 else 0.0

    df_test_res = df_test.copy()
    df_test_res["horizon_days"] = horizon
    df_test_res["pred_prob"] = y_prob_test
    df_test_res["active_mfe"] = df_test[mfe_col]
    df_test_res["active_mae"] = df_test[mae_col]
    df_test_res["active_close_ret"] = df_test[close_col]
    df_test_res["active_target"] = df_test[target_col]
    df_test_res["active_touch"] = df_test[touch_col]

    # Top 10% on matured subset
    k_10 = max(1, int(0.10 * matured_total))
    top_10 = df_test_matured.copy()
    top_10["pred_prob"] = y_prob_matured
    top_10 = top_10.sort_values("pred_prob", ascending=False).head(k_10)
    prec_top10 = float(top_10[target_col].mean()) if len(top_10) > 0 else 0.0
    touch_top10 = float(top_10[touch_col].mean()) if len(top_10) > 0 else 0.0
    top10_mfe = float(top_10[mfe_col].mean()) if len(top_10) > 0 else 0.0
    top10_close_ret = float(top_10[close_col].mean()) if len(top_10) > 0 else 0.0

    # Top 20% on matured subset
    k_20 = max(1, int(0.20 * matured_total))
    top_20 = df_test_matured.copy()
    top_20["pred_prob"] = y_prob_matured
    top_20 = top_20.sort_values("pred_prob", ascending=False).head(k_20)
    prec_top20 = float(top_20[target_col].mean()) if len(top_20) > 0 else 0.0
    touch_top20 = float(top_20[touch_col].mean()) if len(top_20) > 0 else 0.0
    top20_mfe = float(top_20[mfe_col].mean()) if len(top_20) > 0 else 0.0
    top20_close_ret = float(top_20[close_col].mean()) if len(top_20) > 0 else 0.0
    lift_top20 = (prec_top20 / base_valid_rate) if base_valid_rate > 0 else 1.0

    # Bot 80% on matured subset
    bot_80 = df_test_matured.copy()
    bot_80["pred_prob"] = y_prob_matured
    bot_80 = bot_80.sort_values("pred_prob", ascending=False).iloc[k_20:]
    bot80_prec = float(bot_80[target_col].mean()) if len(bot_80) > 0 else 0.0
    bot80_mfe = float(bot_80[mfe_col].mean()) if len(bot_80) > 0 else 0.0
    bot80_close_ret = float(bot_80[close_col].mean()) if len(bot_80) > 0 else 0.0

    # Feature Importance
    importance_gain = model.booster_.feature_importance(importance_type="gain")
    importance_split = model.booster_.feature_importance(importance_type="split")
    tot_gain = sum(importance_gain) if sum(importance_gain) > 0 else 1.0

    feat_imp = []
    for f_name, gain, split in zip(FEATURE_COLS, importance_gain, importance_split):
        feat_imp.append({
            "feature": f_name,
            "gain": float(gain),
            "gain_share": float(gain / tot_gain),
            "split_count": int(split),
        })
    feat_imp = sorted(feat_imp, key=lambda x: x["gain"], reverse=True)

    metrics = {
        "horizon_days": horizon,
        "target_pct": target_pct,
        "checkpoint_min": cp_min,
        "train_samples": len(df_train),
        "test_samples": test_total,
        "base_touch_rate": float(base_touch_rate),
        "base_valid_rate": float(base_valid_rate),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "precision_top10": float(prec_top10),
        "touch_top10": float(touch_top10),
        "top10_avg_mfe": float(top10_mfe),
        "top10_avg_close_ret": float(top10_close_ret),
        "precision_top20": float(prec_top20),
        "touch_top20": float(touch_top20),
        "top20_avg_mfe": float(top20_mfe),
        "top20_avg_close_ret": float(top20_close_ret),
        "lift_factor_top20": float(lift_top20),
        "bot80_precision": float(bot80_prec),
        "bot80_avg_mfe": float(bot80_mfe),
        "bot80_avg_close_ret": float(bot80_close_ret),
        "feature_importance": feat_imp,
    }

    # Save model file
    model_path = MODELS_DIR / f"lgbm_h{horizon}d_cp{cp_min}m.txt"
    model.booster_.save_model(str(model_path))

    return metrics, df_test_res


def main():
    print("=== Training Multi-Horizon Checkpoint Models (1d, 3d, 5d, 10d) ===")
    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}. Run extract_features.py first.")
        sys.exit(1)

    df_all = pd.read_parquet(DATASET_PATH)
    print(f"Loaded dataset: {len(df_all)} total rows.")

    all_metrics = []
    all_test_dfs = []

    for h in HORIZONS:
        print(f"\n=======================================================")
        print(f"🌟 HOLDING HORIZON: {h} TRADING DAYS")
        print(f"=======================================================")
        for cp_min in CHECKPOINTS:
            metrics, df_test_res = train_and_evaluate(h, cp_min, df_all)
            all_metrics.append(metrics)
            all_test_dfs.append(df_test_res)

            print(
                f"[{h:2d}d Horizon] T={cp_min:2d}m | AUC: {metrics['roc_auc']:.3f} | Base Touch: {metrics['base_touch_rate']*100:4.1f}% | Valid: {metrics['base_valid_rate']*100:4.1f}% "
                f"| Top20% Touch: {metrics['touch_top20']*100:4.1f}% | Top20% Valid: {metrics['precision_top20']*100:4.1f}% "
                f"| Lift: {metrics['lift_factor_top20']:.2f}x | Top20% MFE: {metrics['top20_avg_mfe']*100:+5.2f}%"
            )

    # Save all test predictions
    df_combined_test = pd.concat(all_test_dfs, ignore_index=True)
    df_combined_test.to_parquet(RESULTS_DIR / "predictions_oos_multihorizon.parquet", index=False)

    # Save summary JSON
    summary_path = RESULTS_DIR / "evaluation_summary_multihorizon.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved multi-horizon evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
