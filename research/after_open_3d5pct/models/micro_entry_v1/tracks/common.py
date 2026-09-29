"""Shared scoring for one 2-hour sliding track. Track code stays outside this file."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.break_v1.run import MIN_TRAIN, TRAIN_WEEKS, window_days
from research.after_open_3d5pct.models.micro_entry_v1.rows import FEATURES, build_frame
from research.after_open_3d5pct.models.pugh_ge5.build_rows import ROOT
from research.after_open_3d5pct.models.weekly_scale.run import (
    _auc,
    _fit,
    _metrics_by_block,
    _score_block,
    week_blocks,
)

FRAME = ROOT / "research/after_open_3d5pct/runs/micro_entry_v1/frame_2h.parquet"
BASELINE_SCORED = ROOT / "research/after_open_3d5pct/runs/micro_entry_v1/tracks/baseline_scored.parquet"
BASELINE_META = ROOT / "research/after_open_3d5pct/runs/micro_entry_v1/tracks/baseline.json"


def load_frame() -> pd.DataFrame:
    frame = pd.read_parquet(FRAME)
    for column in ("decision_at", "entry_at", "label_end"):
        frame[column] = pd.to_datetime(frame[column])
    return frame


def first_trigger(scored: pd.DataFrame) -> dict:
    if scored.empty:
        return {"days": 0, "buys": 0, "precision": None, "base_rate": None, "lift": None}
    ordered = scored.sort_values(["symbol", "session_date", "decision_at"])
    first = ordered.groupby(["symbol", "session_date"], sort=False).head(1)
    base = float(first["y_3d_5pct"].mean())
    taken = ordered.loc[ordered["score"] >= ordered["level"]]
    taken = taken.groupby(["symbol", "session_date"], sort=False).head(1)
    precision = None if taken.empty else float(taken["y_3d_5pct"].mean())
    return {
        "days": int(len(first)),
        "first_clock_base": base,
        "buys": int(len(taken)),
        "precision": precision,
        "lift": None if precision is None else precision - base,
    }


def beats(candidate: dict, reference: dict) -> bool:
    """Both exam weeks must rise, and each week must still have at least 15 buys."""
    for key in ("test_first", "validation_first"):
        left = candidate[key]
        right = reference[key]
        if left["precision"] is None or right["precision"] is None:
            return False
        if left["buys"] < 15 or not left["precision"] > right["precision"]:
            return False
    return True


def walk(frame: pd.DataFrame, features: list[str], annotate) -> dict:
    """Fit the frozen tree. annotate(train, test, validation, model, features) returns scored test and validation."""
    dates = sorted(frame["session_date"].unique())
    weeks = week_blocks(dates)
    pooled = {"test": [], "validation": [], "train_auc": []}
    used = 0
    skipped = 0
    last = None
    for index in range(TRAIN_WEEKS, len(weeks) - 1):
        train_days, test_days, val_days = window_days(weeks, index)
        test = frame.loc[frame["session_date"].isin(test_days)].copy()
        validation = frame.loc[frame["session_date"].isin(val_days)].copy()
        if test.empty or validation.empty:
            skipped += 1
            continue
        before = pd.Timestamp(test["entry_at"].min())
        train = frame.loc[frame["session_date"].isin(train_days) & frame["label_end"].lt(before)].copy()
        if len(train) < MIN_TRAIN or train["y_3d_5pct"].nunique() < 2:
            skipped += 1
            continue
        model = _fit(train, features)
        train_score = model.predict_proba(train[features])[:, 1]
        pooled["train_auc"].append(_auc(train["y_3d_5pct"].to_numpy(int), train_score))
        scored_test, scored_validation = annotate(train, test, validation, model, features)
        scored_test = scored_test.copy()
        scored_validation = scored_validation.copy()
        scored_test["triplet"] = index
        scored_validation["triplet"] = index
        pooled["test"].append(scored_test)
        pooled["validation"].append(scored_validation)
        last = {
            "train": [train_days[0], train_days[-1]],
            "test": [test_days[0], test_days[-1]],
            "validation": [val_days[0], val_days[-1]],
        }
        used += 1
    scored = {
        key: pd.concat(pooled[key], ignore_index=True) if pooled[key] else pd.DataFrame()
        for key in ("test", "validation")
    }
    train_aucs = [item for item in pooled["train_auc"] if item is not None]
    return {
        "fitted_triplets": used,
        "skipped_triplets": skipped,
        "mean_train_auc": None if not train_aucs else float(np.mean(train_aucs)),
        "last_triplet": last,
        "test": _metrics_by_block(scored["test"]),
        "validation": _metrics_by_block(scored["validation"]),
        "test_first": first_trigger(scored["test"]),
        "validation_first": first_trigger(scored["validation"]),
        "scored_test": scored["test"],
        "scored_validation": scored["validation"],
    }


def annotate_current(train, test, validation, model, features):
    level = float(np.quantile(model.predict_proba(train[features])[:, 1], 0.80))
    return _score_block(model, features, test, level), _score_block(model, features, validation, level)


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    return value


def pack(result: dict) -> dict:
    kept = {key: result[key] for key in (
        "fitted_triplets", "skipped_triplets", "mean_train_auc", "last_triplet",
        "test", "validation", "test_first", "validation_first",
    )}
    return _plain(kept)


def ensure_baseline() -> dict:
    if BASELINE_META.exists() and BASELINE_SCORED.exists() and FRAME.exists():
        return json.loads(BASELINE_META.read_text())
    FRAME.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_SCORED.parent.mkdir(parents=True, exist_ok=True)
    if not FRAME.exists():
        frame = build_frame(24, step=1)
        frame.to_parquet(FRAME, index=False)
    frame = load_frame()
    done = walk(frame, list(FEATURES), annotate_current)
    scored = pd.concat([
        done["scored_test"].assign(split="test"),
        done["scored_validation"].assign(split="validation"),
    ], ignore_index=True)
    scored.to_parquet(BASELINE_SCORED, index=False)
    payload = pack(done)
    BASELINE_META.write_text(json.dumps(payload, indent=2, default=str))
    return payload
