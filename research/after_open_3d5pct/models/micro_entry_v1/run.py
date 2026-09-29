"""Two frozen rounds: enter after a 2-hour slide, then after a 30-minute slide."""
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

OUT = ROOT / "research/after_open_3d5pct/runs/micro_entry_v1"
ROUNDS = (
    {"name": "r1_2h", "length": 24},
    {"name": "r2_30m", "length": 6},
)


def first_trigger(scored: pd.DataFrame) -> dict:
    """One action per symbol-day: the earliest window whose score clears the line."""
    if scored.empty:
        return {"days": 0, "buys": 0, "precision": None, "base_rate": None, "lift": None}
    ordered = scored.sort_values(["symbol", "session_date", "decision_at"])
    first = ordered.groupby(["symbol", "session_date"], sort=False).head(1)
    base = float(first["y_3d_5pct"].mean())
    taken = ordered.loc[ordered["score"] >= ordered["level"]]
    taken = taken.groupby(["symbol", "session_date"], sort=False).head(1)
    precision = None if taken.empty else float(taken["y_3d_5pct"].mean())
    return {
        "days": int(first["session_date"].count()),
        "first_clock_base": base,
        "buys": int(len(taken)),
        "precision": precision,
        "lift": None if precision is None else precision - base,
    }


def walk(frame: pd.DataFrame, dates: list[str]) -> dict:
    weeks = week_blocks(dates)
    pooled = {"test": [], "validation": [], "train_auc": []}
    used = 0
    skipped = 0
    last = None
    for index in range(TRAIN_WEEKS, len(weeks) - 1):
        train_days, test_days, val_days = window_days(weeks, index)
        test = frame.loc[frame["session_date"].isin(test_days)]
        validation = frame.loc[frame["session_date"].isin(val_days)]
        if test.empty or validation.empty:
            skipped += 1
            continue
        before = pd.Timestamp(test["entry_at"].min())
        train = frame.loc[frame["session_date"].isin(train_days) & frame["label_end"].lt(before)]
        if len(train) < MIN_TRAIN or train["y_3d_5pct"].nunique() < 2:
            skipped += 1
            continue
        model = _fit(train, FEATURES)
        level = float(np.quantile(model.predict_proba(train[FEATURES])[:, 1], 0.80))
        train_score = model.predict_proba(train[FEATURES])[:, 1]
        pooled["train_auc"].append(_auc(train["y_3d_5pct"].to_numpy(int), train_score))
        pooled["test"].append(_score_block(model, FEATURES, test, level))
        pooled["validation"].append(_score_block(model, FEATURES, validation, level))
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
    }


def run() -> dict:
    payload = {"train_weeks": TRAIN_WEEKS, "rounds": {}}
    for spec in ROUNDS:
        frame = build_frame(spec["length"], step=1)
        dates = sorted(frame["session_date"].unique())
        clock = frame.loc[frame["decision_at"].dt.strftime("%H:%M").eq("11:30")].copy()
        slide = walk(frame, dates)
        fixed = walk(clock, dates)
        payload["rounds"][spec["name"]] = {
            "length": spec["length"],
            "rows": int(len(frame)),
            "clock_rows": int(len(clock)),
            "slide": slide,
            "clock_1130": fixed,
        }
        print(json.dumps({
            "round": spec["name"],
            "slide_test": slide["test"].get("precision"),
            "slide_first": slide["test_first"].get("precision"),
            "clock_test": fixed["test"].get("precision"),
            "clock_first": fixed["test_first"].get("precision"),
        }, default=str), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload


if __name__ == "__main__":
    run()
