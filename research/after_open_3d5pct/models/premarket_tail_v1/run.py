"""Eight-week walk for the premarket-tail entry. The 80th percentile stays put."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.break_v1.run import MIN_TRAIN, TRAIN_WEEKS, window_days
from research.after_open_3d5pct.models.premarket_tail_v1.rows import FEATURES, FOLLOW_UP, build_frame
from research.after_open_3d5pct.models.pugh_ge5.build_rows import ROOT
from research.after_open_3d5pct.models.weekly_scale.run import (
    _auc,
    _fit,
    _metrics_by_block,
    _score_block,
    matured_before,
    week_blocks,
)

OUT = ROOT / "research/after_open_3d5pct/runs/premarket_tail_v1"
MIN_BUYS = 15


def decision_cutoff(test_days: list[str]) -> pd.Timestamp:
    return pd.Timestamp(f"{test_days[0]} 09:30")


def beats_everyday(block: dict) -> bool:
    precision = block.get("precision")
    base = block.get("base_rate")
    buys = block.get("buys") or 0
    if precision is None or base is None:
        return False
    return int(buys) >= MIN_BUYS and float(precision) > float(base)


def beats_prior(newer: dict, older: dict) -> bool:
    if (newer.get("precision") is None) or (older.get("precision") is None):
        return False
    return int(newer.get("buys") or 0) >= MIN_BUYS and float(newer["precision"]) > float(older["precision"])


def sign_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"rows": 0, "base_rate": None, "buys": 0, "precision": None, "lift": None}
    base = float(frame["y_3d_5pct"].mean())
    bought = frame.loc[frame["path_impulse"].gt(0)]
    precision = None if bought.empty else float(bought["y_3d_5pct"].mean())
    return {
        "rows": int(len(frame)),
        "base_rate": base,
        "buys": int(len(bought)),
        "precision": precision,
        "lift": None if precision is None else precision - base,
    }


def coverage(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"rows": 0, "overnight_finite": 0, "premarket_finite": 0, "both_finite": 0, "both_share": None}
    overnight = frame["overnight_return"].notna()
    premarket = frame["premarket_return"].notna()
    both = overnight & premarket
    return {
        "rows": int(len(frame)),
        "overnight_finite": int(overnight.sum()),
        "premarket_finite": int(premarket.sum()),
        "both_finite": int(both.sum()),
        "both_share": float(both.mean()),
    }


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def walk(frame: pd.DataFrame, features: list[str]) -> dict:
    dates = sorted(frame["session_date"].unique())
    weeks = week_blocks(dates)
    pooled = {"test": [], "validation": [], "train_auc": []}
    used = 0
    skipped = 0
    last = None
    last_blocks: dict = {}
    for index in range(TRAIN_WEEKS, len(weeks) - 1):
        train_days, test_days, val_days = window_days(weeks, index)
        if set(train_days) & set(test_days) or set(test_days) & set(val_days):
            raise RuntimeError("train, test, and validation weeks overlap")
        before = decision_cutoff(test_days)
        train = matured_before(frame, train_days, before)
        test = frame.loc[frame["session_date"].isin(test_days)].copy()
        validation = frame.loc[frame["session_date"].isin(val_days)].copy()
        if train.empty or train["session_date"].max() >= test_days[0] or test_days[-1] >= val_days[0]:
            raise RuntimeError("week order collapsed")
        if len(train) < MIN_TRAIN or train["y_3d_5pct"].nunique() < 2 or test.empty or validation.empty:
            skipped += 1
            continue
        model = _fit(train, features)
        level = float(np.quantile(model.predict_proba(train[features])[:, 1], 0.80))
        train_score = model.predict_proba(train[features])[:, 1]
        pooled["train_auc"].append(_auc(train["y_3d_5pct"].to_numpy(int), train_score))
        scored_test = _score_block(model, features, test, level)
        scored_validation = _score_block(model, features, validation, level)
        pooled["test"].append(scored_test)
        pooled["validation"].append(scored_validation)
        last = {
            "train": [train_days[0], train_days[-1]],
            "test": [test_days[0], test_days[-1]],
            "validation": [val_days[0], val_days[-1]],
        }
        last_blocks = {"test": scored_test, "validation": scored_validation}
        used += 1
        if used % 20 == 0:
            print(json.dumps({"fitted_triplets": used, "through": test_days[-1]}), flush=True)
    test = pd.concat(pooled["test"], ignore_index=True) if pooled["test"] else pd.DataFrame()
    validation = pd.concat(pooled["validation"], ignore_index=True) if pooled["validation"] else pd.DataFrame()
    train_aucs = [item for item in pooled["train_auc"] if item is not None]
    return {
        "features": list(features),
        "fitted_triplets": used,
        "skipped_triplets": skipped,
        "weeks": len(weeks),
        "mean_train_auc": None if not train_aucs else float(np.mean(train_aucs)),
        "last_triplet": last,
        "test": _metrics_by_block(test),
        "validation": _metrics_by_block(validation),
        "test_sign": sign_metrics(test),
        "validation_sign": sign_metrics(validation),
        "test_coverage": coverage(test),
        "validation_coverage": coverage(validation),
        "last_triplet_metrics": {
            "test": _metrics_by_block(last_blocks["test"]) if last_blocks else {},
            "validation": _metrics_by_block(last_blocks["validation"]) if last_blocks else {},
        },
    }


def run() -> dict:
    frame = build_frame()
    if frame.empty:
        raise RuntimeError("no premarket-tail rows")
    first = walk(frame, FEATURES)
    opened = beats_everyday(first["test"]) and beats_everyday(first["validation"])
    payload = {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()),
        "first": frame["session_date"].min(),
        "last": frame["session_date"].max(),
        "round1": first,
        "round2_opened": opened,
        "bound": "round1",
    }
    if opened:
        second = walk(frame, FEATURES + FOLLOW_UP)
        payload["round2"] = second
        if beats_prior(second["test"], first["test"]) and beats_prior(second["validation"], first["validation"]):
            payload["bound"] = "round2"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(_plain(payload), indent=2))
    print(json.dumps(_plain({
        "bound": payload["bound"],
        "round2_opened": opened,
        "test": first["test"],
        "validation": first["validation"],
        "test_sign": first["test_sign"],
        "validation_sign": first["validation_sign"],
        "test_coverage": first["test_coverage"],
        "validation_coverage": first["validation_coverage"],
    }), default=str), flush=True)
    return payload


if __name__ == "__main__":
    run()
