"""Train one new model per registry family. Checkpoints in model_registry_v1 stay put."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.pugh_ge5.build_rows import GROUPS, ROOT
from research.after_open_3d5pct.models.pugh_ge5.walk import FEATURES
from research.after_open_3d5pct.models.weekly_scale.run import (
    MACRO,
    MICRO,
    RELATION,
    WEEK,
    _auc,
    _fit,
    _metrics_by_block,
    _score_block,
    attach,
    matured_before,
    week_blocks,
)
from research.after_open_3d5pct.models.pugh_ge5.build_rows import build_panel

TRAIN_WEEKS = 8
MIN_TRAIN = 40
OUT = ROOT / "research/after_open_3d5pct/runs/break_v1"

B_NO_GROUP = list(FEATURES) + MICRO + MACRO + RELATION
B_GROUP = list(FEATURES) + ["gap", "burst_5m", "peer_morning", "gap_x_peer", "burst_x_peer"]
C_NO_GROUP = list(FEATURES) + ["own_base"] + MICRO
C_GROUP = list(FEATURES) + [
    "burst_pos", "micro_peak", "micro_fade", "peer_morning", "burst_pos_x_peer", "fade_x_peer",
]
C_NO_DAILY = list(FEATURES) + MACRO
PARENTS = {
    "B_no_group": B_NO_GROUP,
    "B_group": B_GROUP,
    "C_no_group": C_NO_GROUP,
    "C_group": C_GROUP,
    "C_no_daily": C_NO_DAILY,
}


def peer_morning(frame: pd.DataFrame) -> pd.Series:
    total = frame.groupby("session_date")["morning_return"].transform("sum")
    count = frame.groupby("session_date")["morning_return"].transform("count")
    peer = (total - frame["morning_return"]) / (count - 1)
    return peer.where(count > 1)


def add_peer(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["peer_morning"] = peer_morning(out)
    out["gap_x_peer"] = out["gap"] * out["peer_morning"]
    out["burst_x_peer"] = out["burst_5m"] * out["peer_morning"]
    out["burst_pos_x_peer"] = out["burst_pos"] * out["peer_morning"]
    out["fade_x_peer"] = out["micro_fade"] * out["peer_morning"]
    return out


def window_days(weeks: list[list[str]], index: int, train_weeks: int = TRAIN_WEEKS) -> tuple[list[str], list[str], list[str]]:
    train = [day for block in weeks[index - train_weeks:index] for day in block]
    return train, weeks[index], weeks[index + 1]


def walk(frame: pd.DataFrame, features: list[str], dates: list[str], label: str = "y_3d_5pct", min_train: int = MIN_TRAIN) -> dict:
    weeks = week_blocks(dates)
    pooled = {"test": [], "validation": [], "train_auc": []}
    used = 0
    skipped = 0
    last = None
    last_blocks = {}
    for index in range(TRAIN_WEEKS, len(weeks) - 1):
        train_days, test_days, val_days = window_days(weeks, index)
        if set(train_days) & set(test_days) or set(test_days) & set(val_days):
            raise RuntimeError("train, test, and validation weeks overlap")
        before = pd.Timestamp(f"{test_days[0]} 11:30")
        train = matured_before(frame, train_days, before)
        if label != "y_3d_5pct":
            train["y_3d_5pct"] = train[label].to_numpy()
        test = frame.loc[frame["session_date"].isin(test_days)].copy()
        validation = frame.loc[frame["session_date"].isin(val_days)].copy()
        if len(train) < min_train or train[label].nunique() < 2 or test.empty or validation.empty:
            skipped += 1
            continue
        model = _fit(train, features)
        level = float(np.quantile(model.predict_proba(train[features])[:, 1], 0.80))
        train_score = model.predict_proba(train[features])[:, 1]
        pooled["train_auc"].append(_auc(train[label].to_numpy(int), train_score))
        scored_test = _score_block(model, features, test, level)
        scored_validation = _score_block(model, features, validation, level)
        scored_test["y_3d_5pct"] = scored_test[label]
        scored_validation["y_3d_5pct"] = scored_validation[label]
        pooled["test"].append(scored_test)
        pooled["validation"].append(scored_validation)
        last = {"train": [train_days[0], train_days[-1]], "test": [test_days[0], test_days[-1]], "validation": [val_days[0], val_days[-1]]}
        last_blocks = {"test": scored_test, "validation": scored_validation}
        used += 1
    test = pd.concat(pooled["test"], ignore_index=True) if pooled["test"] else pd.DataFrame()
    validation = pd.concat(pooled["validation"], ignore_index=True) if pooled["validation"] else pd.DataFrame()
    train_aucs = [item for item in pooled["train_auc"] if item is not None]
    return {
        "fitted_triplets": used,
        "skipped_triplets": skipped,
        "mean_train_auc": None if not train_aucs else float(np.mean(train_aucs)),
        "last_triplet": last,
        "test": _metrics_by_block(test),
        "validation": _metrics_by_block(validation),
        "last_triplet_metrics": {
            "test": _metrics_by_block(last_blocks["test"]) if last_blocks else {},
            "validation": _metrics_by_block(last_blocks["validation"]) if last_blocks else {},
        },
    }


def _snxx() -> dict:
    from research.after_open_3d5pct.models.SNXX_3d3pct.labels import build_rows
    from research.after_open_3d5pct.models.SNXX_3d3pct.train import BARS, FEATURES as BASE
    if not BARS.exists():
        return {"status": "missing_bars"}
    rows, excluded = build_rows(pd.read_parquet(BARS), barrier=0.03)
    if rows.empty:
        return {"status": "no_rows", "excluded": excluded}
    rows = rows.copy()
    rows["symbol"] = "SNXX"
    rows["gap_x_slope"] = rows["overnight_gap"] * rows["r_20"]
    dates = sorted(rows["session_date"].unique())
    done = walk(rows, list(BASE) + ["gap_x_slope"], dates, label="y", min_train=30)
    done["status"] = "ok"
    done["excluded"] = excluded
    return done


def _soxs() -> dict:
    from research.after_open_3d5pct.models.SOXS_SOXX_3d5pct.labels import build_rows
    from research.after_open_3d5pct.models.SOXS_SOXX_3d5pct.train import FEATURES as BASE
    soxs = ROOT / "market_data/us_5m/SOXS/2026.parquet"
    soxx = ROOT / "market_data/us_5m/SOXX/2026.parquet"
    if not soxs.exists() or not soxx.exists():
        return {"status": "missing_bars"}
    rows, excluded = build_rows(pd.read_parquet(soxs), pd.read_parquet(soxx), barrier=0.05)
    if rows.empty:
        return {"status": "no_rows", "excluded": excluded}
    rows = rows.copy()
    rows["spread_x_slow"] = rows["morning_spread"] * rows["r12_spread"]
    features = list(BASE) + ["spread_x_slow"]
    dates = sorted(rows["session_date"].unique())
    weeks = week_blocks(dates)
    test_parts = []
    val_parts = []
    used = 0
    skipped = 0
    last = None
    for index in range(TRAIN_WEEKS, len(weeks) - 1):
        train_days, test_days, val_days = window_days(weeks, index)
        before = pd.Timestamp(f"{test_days[0]} 11:30")
        train = matured_before(rows, train_days, before)
        test = rows.loc[rows["session_date"].isin(test_days)].copy()
        validation = rows.loc[rows["session_date"].isin(val_days)].copy()
        if len(train) < 30 or test.empty or validation.empty:
            skipped += 1
            continue
        if train["y_SOXS"].nunique() < 2 or train["y_SOXX"].nunique() < 2:
            skipped += 1
            continue
        scored = {"test": test, "validation": validation}
        for key, block in scored.items():
            block = block.copy()
            probs = {}
            for name in ("SOXS", "SOXX"):
                import lightgbm as lgb
                model = lgb.LGBMClassifier(
                    n_estimators=40, num_leaves=7, max_depth=3, min_child_samples=10,
                    learning_rate=0.05, reg_lambda=10.0, n_jobs=1, verbosity=-1, random_state=3566,
                )
                model.fit(train[features], train[f"y_{name}"].to_numpy(int))
                probs[name] = model.predict_proba(block[features])[:, 1]
            chosen = np.array(["flat"] * len(block), dtype=object)
            buy = (probs["SOXS"] >= 0.5) | (probs["SOXX"] >= 0.5)
            chosen[buy & (probs["SOXS"] >= probs["SOXX"])] = "SOXS"
            chosen[buy & (probs["SOXX"] > probs["SOXS"])] = "SOXX"
            block["chosen"] = chosen
            block["hit"] = np.where(chosen == "SOXS", block["y_SOXS"], np.where(chosen == "SOXX", block["y_SOXX"], np.nan))
            if key == "test":
                test_parts.append(block)
            else:
                val_parts.append(block)
        last = {"train": [train_days[0], train_days[-1]], "test": [test_days[0], test_days[-1]], "validation": [val_days[0], val_days[-1]]}
        used += 1
    def _pack(parts: list[pd.DataFrame]) -> dict:
        if not parts:
            return {"rows": 0, "accuracy": None, "buys": 0, "precision": None}
        frame = pd.concat(parts, ignore_index=True)
        buy = frame["chosen"].isin(["SOXS", "SOXX"])
        hits = frame.loc[buy, "hit"].to_numpy(float)
        return {
            "rows": int(len(frame)),
            "accuracy": float((frame["chosen"] == frame["action"]).mean()),
            "buys": int(buy.sum()),
            "precision": None if not buy.any() else float(np.nanmean(hits)),
            "base_rate": float(((frame["y_SOXS"] == 1) | (frame["y_SOXX"] == 1)).mean()),
        }
    return {
        "status": "ok",
        "excluded": excluded,
        "fitted_triplets": used,
        "skipped_triplets": skipped,
        "last_triplet": last,
        "test": _pack(test_parts),
        "validation": _pack(val_parts),
    }


def run() -> dict:
    panel, coverage = build_panel("2024")
    frame = add_peer(attach(panel))
    dates = sorted(frame["session_date"].unique())
    models = {}
    for name, features in PARENTS.items():
        models[name] = walk(frame, features, dates)
        print(json.dumps({"model": name, "test": models[name]["test"], "validation": models[name]["validation"]}, default=str), flush=True)
        for group, symbols in GROUPS.items():
            key = f"{name}_{group}"
            models[key] = walk(frame.loc[frame["symbol"].isin(symbols)].copy(), features, dates)
            print(json.dumps({"model": key, "test_precision": models[key]["test"].get("precision"), "test_auc": models[key]["test"].get("auc")}, default=str), flush=True)
    models["SNXX_3d3pct"] = _snxx()
    print(json.dumps({"model": "SNXX_3d3pct", "status": models["SNXX_3d3pct"].get("status"), "test": models["SNXX_3d3pct"].get("test")}, default=str), flush=True)
    models["SOXS_SOXX_3d5pct"] = _soxs()
    print(json.dumps({"model": "SOXS_SOXX_3d5pct", "status": models["SOXS_SOXX_3d5pct"].get("status"), "test": models["SOXS_SOXX_3d5pct"].get("test")}, default=str), flush=True)
    payload = {
        "train_weeks": TRAIN_WEEKS,
        "week_length": WEEK,
        "coverage": {key: coverage[key] for key in ("rows", "first", "last")},
        "models": models,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload


if __name__ == "__main__":
    done = run()
    print(json.dumps({"models": list(done["models"])}))
