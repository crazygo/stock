"""Frozen-data, exposed-history development for the B-with-groups LightGBM route."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from research.after_open_3d5pct.train_multiscale_v6 import _folds, project_monotone, tabular


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DATASET = ROOT / "research/after_open_3d5pct/runs/multiscale_groups_v61_dataset_20260926_r2"
BASE = ROOT / "research/after_open_3d5pct/runs/multiscale_groups_v61_fit_20260926"
CONFIG = json.loads((BASE / "config.json").read_text())
TARGETS = [f"{h}_{t}" for h in range(3) for t in range(3)]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def load():
    rows = pd.read_parquet(DATASET / "rows.parquet")
    with np.load(DATASET / "features.npz") as source:
        data = {key: source[key] for key in source.files}
    folds = _folds(rows, CONFIG)
    assert len(rows) == len(data["y"]) == 9222
    assert {k: len(v) for k, v in folds.items()} == {"train": 5606, "inner": 1521, "outer": 1140}
    assert digest(DATASET / "manifest.json") == "56b3d04def58b72e388f98e41c0d72df1b996aef3ecf7c8ee6325349082a143b"
    return rows, data, folds


def metrics(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return {"n": len(y), "actual": float(y.mean()), "mean_p": float(p.mean()),
            "brier": float(np.mean((p-y)**2)), "logloss": float(log_loss(y, p, labels=[0, 1])),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None}


def curve_features(data):
    """Observed 09:30-11:30 5m prefix only; no future padded slots."""
    x = data["x5"][:, 7, 66:90, :].astype(float)
    valid = x[:, :, 10] > 0
    assert valid.all(), "Frozen sample contract requires complete opening prefix"
    r = x[:, :, 0] + x[:, :, 3]
    # The first cross-bar channel reaches outside this regular-session window.
    r[:, 0] = x[:, 0, 0]
    path = np.cumsum(r, axis=1)
    block = [r[:, i:i+8].sum(axis=1) for i in (0, 8, 16)]
    peak = np.maximum.accumulate(np.column_stack((np.zeros(len(x)), path)), axis=1)
    trough = np.minimum.accumulate(np.column_stack((np.zeros(len(x)), path)), axis=1)
    drawdown = np.min(np.column_stack((np.zeros(len(x)), path)) - peak, axis=1)
    runup = np.max(np.column_stack((np.zeros(len(x)), path)) - trough, axis=1)
    eff = path[:, -1] / (np.abs(r).sum(axis=1) + 1e-5)
    early_vol = np.std(r[:, :12], axis=1)
    late_vol = np.std(r[:, 12:], axis=1)
    early_dollar = np.mean(x[:, :12, 4], axis=1)
    late_dollar = np.mean(x[:, 12:, 4], axis=1)
    return pd.DataFrame(np.column_stack((*block, drawdown, runup, eff,
                                         late_vol - early_vol, late_dollar - early_dollar)),
                        columns=["open40_ret", "mid40_ret", "late40_ret", "prefix_drawdown",
                                 "prefix_runup", "prefix_efficiency", "late_early_vol_delta",
                                 "late_early_logdollar_delta"])


def group_relative_features(data):
    """Current peer/QQQ relative state, preserving six fixed group type slots."""
    seq = data["group_seq"][:, -1].astype(float)
    x = data["x5"][:, 7, 66:90, :]
    own = x[:, :, 0].sum(axis=1) + x[:, 1:, 3].sum(axis=1)
    out, names = [], []
    for slot, typ in enumerate(("trend15", "trend63", "trend126", "volatility", "liquidity", "qqq")):
        valid = seq[:, slot, 9] > 0
        out.append(np.where(valid, own - seq[:, slot, 3], np.nan))
        names.append(f"own_minus_{typ}_cutoff_return")
    return pd.DataFrame(np.column_stack(out), columns=names)


def make_x(data, variant):
    x = tabular(data, group=True)
    if variant in ("curve", "curve_group", "curve_group_cal"):
        x = pd.concat((x, curve_features(data)), axis=1)
    if variant in ("curve_group", "curve_group_cal"):
        x = pd.concat((x, group_relative_features(data)), axis=1)
    return x


def reference_predictions(x, ids):
    arr = np.column_stack([lgb.Booster(model_file=str(BASE / "models" / f"B_lgbm_group_{j}.txt")).predict(x.iloc[ids], num_threads=1)
                           for j in range(9)])
    return project_monotone(arr).reshape(-1, 9)


def diagnostic():
    rows, data, folds = load()
    x = make_x(data, "base")
    y = data["y"].reshape(-1, 9)
    p_train = reference_predictions(x, folds["train"])
    p_inner = reference_predictions(x, folds["inner"])
    imp = []
    for j in range(9):
        model = lgb.Booster(model_file=str(BASE / "models" / f"B_lgbm_group_{j}.txt"))
        imp.append(model.feature_importance(importance_type="gain"))
    imp = np.stack(imp)
    valid = data["group_seq"][..., 9] > 0
    result = {"dataset_sha256": digest(DATASET / "manifest.json"),
              "fold_rows": {k: len(v) for k, v in folds.items()},
              "base_features": x.shape[1],
              "all_missing_features_train": int(x.iloc[folds["train"]].isna().all().sum()),
              "constant_features_train": int((x.iloc[folds["train"]].nunique(dropna=True) <= 1).sum()),
              "group_valid_slot_rate": valid.mean(axis=(0, 1)).tolist(),
              "daily_full_rate": {k: float(rows.iloc[v].full_126_prior_days.mean()) for k, v in folds.items()},
              "R0": {split: {name: metrics(y[folds[split], j], p[:, j])
                                     for j, name in enumerate(TARGETS)}
                     for split, p in (("train", p_train), ("inner", p_inner))},
              "primary_gain_by_family": {"5m": float(imp[4, :104].sum()),
                                         "60m": float(imp[4, 104:507].sum()),
                                         "daily": float(imp[4, 507:585].sum()),
                                         "group": float(imp[4, 585:].sum())},
              "primary_trees": int(lgb.Booster(model_file=str(BASE / "models" / "B_lgbm_group_4.txt")).num_trees())}
    # Each 5m day contributes 13 summary columns. Current day is the eighth.
    current_start = 7 * 13
    result["current_5m_last_half_missing_rate"] = float(x.iloc[:, current_start+12].isna().mean())
    (HERE / "diagnosis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"primary": {k: result["R0"][k]["1_1"] for k in ("train", "inner")},
                      "missing": result["all_missing_features_train"],
                      "constant": result["constant_features_train"],
                      "gain": result["primary_gain_by_family"],
                      "current_missing": result["current_5m_last_half_missing_rate"]}, indent=2))


def fit(round_id, variant, trees, leaves, min_leaf, lr, half_life_sessions=0):
    rows, data, folds = load()
    x = make_x(data, variant)
    y = data["y"].reshape(-1, 9)
    output = ROOT / f"research/after_open_3d5pct/runs/v7_b_group_{round_id}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    (output / "models").mkdir()
    (output / "experiment_source.py").write_bytes(Path(__file__).read_bytes())
    x.columns.to_series().to_json(output / "feature_schema.json", orient="values")
    settings = {"round": round_id, "variant": variant, "max_trees": trees, "num_leaves": leaves,
                "min_child_samples": min_leaf, "learning_rate": lr, "reg_lambda": 10,
                "half_life_sessions": half_life_sessions,
                "dataset_sha256": digest(DATASET / "manifest.json"), "code_sha256": digest(Path(__file__)),
                "base_code_sha256": digest(ROOT / "research/after_open_3d5pct/train_multiscale_v6.py"),
                "config_sha256": digest(BASE / "config.json"), "python": os.sys.version,
                "lightgbm": lgb.__version__}
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    pi, po, pt = np.zeros((len(folds["inner"]), 9)), np.zeros((len(folds["outer"]), 9)), np.zeros((len(folds["train"]), 9))
    train_weight = None
    if half_life_sessions:
        train_dates = rows.iloc[folds["train"]].session_date.to_numpy()
        ordered = sorted(set(train_dates))
        age = np.array([len(ordered) - 1 - ordered.index(day) for day in train_dates], float)
        train_weight = 0.5 ** (age / half_life_sessions)
        settings["weight_effective_n"] = float(train_weight.sum() ** 2 / np.square(train_weight).sum())
        settings["weight_min_max"] = [float(train_weight.min()), float(train_weight.max())]
        (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    logs = []
    started = time.monotonic()
    for j in range(9):
        model = lgb.LGBMClassifier(n_estimators=trees, num_leaves=leaves, max_depth=-1,
                                   min_child_samples=min_leaf, learning_rate=lr, reg_lambda=10,
                                   verbosity=-1, n_jobs=1, random_state=3566)
        curve = {}
        model.fit(x.iloc[folds["train"]], y[folds["train"], j], sample_weight=train_weight,
                  eval_set=[(x.iloc[folds["train"]], y[folds["train"], j]),
                            (x.iloc[folds["inner"]], y[folds["inner"], j])],
                  eval_names=["train", "inner"], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(35, verbose=False), lgb.record_evaluation(curve)])
        model.booster_.save_model(str(output / "models" / f"target_{j}.txt"))
        limit = model.best_iteration_
        pt[:, j] = model.predict_proba(x.iloc[folds["train"]], num_iteration=limit)[:, 1]
        pi[:, j] = model.predict_proba(x.iloc[folds["inner"]], num_iteration=limit)[:, 1]
        po[:, j] = model.predict_proba(x.iloc[folds["outer"]], num_iteration=limit)[:, 1]
        logs.append({"target": TARGETS[j], "best_trees": int(limit),
                     "curve": {split: {metric: [float(v) for v in values]
                                       for metric, values in scores.items()} for split, scores in curve.items()}})
    for split, raw in (("train", pt), ("inner", pi), ("outer", po)):
        p = project_monotone(raw).reshape(-1, 9)
        frame = rows.iloc[folds[split]][["sample_id", "symbol", "session_date"]].copy()
        for j, name in enumerate(TARGETS):
            frame[f"y_{name}"] = y[folds[split], j]
            frame[f"p_raw_{name}"] = raw[:, j]
            frame[f"p_{name}"] = p[:, j]
        frame.to_parquet(output / f"predictions_{split}.parquet", index=False)
    info = {"status": "complete", "fit_seconds": time.monotonic() - started,
            "fold_rows": {k: len(v) for k, v in folds.items()},
            "targets": {name: {split: metrics(y[folds[split], j], p[:, j])
                               for split, p in (("train", project_monotone(pt).reshape(-1, 9)),
                                                ("inner", project_monotone(pi).reshape(-1, 9)))}
                        for j, name in enumerate(TARGETS)}, "learning": logs}
    (output / "inner_result.json").write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"round": round_id, "primary_inner": info["targets"]["1_1"]["inner"],
                      "primary_train": info["targets"]["1_1"]["train"],
                      "trees": [v["best_trees"] for v in logs], "seconds": info["fit_seconds"]}, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("diagnose")
    train = sub.add_parser("fit")
    train.add_argument("--round", required=True)
    train.add_argument("--variant", required=True)
    train.add_argument("--trees", type=int, default=300)
    train.add_argument("--leaves", type=int, default=7)
    train.add_argument("--min-leaf", type=int, default=100)
    train.add_argument("--lr", type=float, default=0.035)
    train.add_argument("--half-life-sessions", type=int, default=0)
    args = cli.parse_args()
    if args.command == "diagnose":
        diagnostic()
    else:
        fit(args.round, args.variant, args.trees, args.leaves, args.min_leaf, args.lr,
            args.half_life_sessions)
