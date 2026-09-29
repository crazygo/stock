"""Buy when the morning holds a late high. Cutoffs are fixed before the 2026 score.

The 12-shape codebook separates a high made in the last part of a two-hour
window from a high made in the first 40 minutes and then given back. At 11:30
only the first window exists. This rule uses that split and does not search
cutoffs on the confirm window.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.pugh_ge5.build_rows import ROOT, _regular_from_2024, build_panel

BARS = 24
LATE_PEAK_BAR = 16
MAX_FADE = 0.10
EARLY_PEAK_BAR = 8
MIN_FADE = 0.70
MIN_BUYS = 15
PRECISION_BAR = 0.90
CONFIRM_START, CONFIRM_END = "2026-01-02", "2026-09-18"
OUT = ROOT / "research/after_open_3d5pct/runs/per_stock_90"


def _fade(close: np.ndarray) -> tuple[int, float]:
    peak = int(np.argmax(close))
    span = float(np.max(close) - np.min(close))
    fade = 0.0 if span <= 1e-8 else float((close[peak] - close[-1]) / span)
    return peak, fade


def is_true_push(close: np.ndarray) -> bool:
    if len(close) != BARS or np.any(close <= 0) or not np.isfinite(close).all():
        return False
    peak, fade = _fade(close)
    return peak >= LATE_PEAK_BAR and fade <= MAX_FADE


def is_inverted_v(close: np.ndarray) -> bool:
    if len(close) != BARS or np.any(close <= 0) or not np.isfinite(close).all():
        return False
    peak, fade = _fade(close)
    return peak <= EARLY_PEAK_BAR and fade >= MIN_FADE


def morning_flags(bars: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if bars.empty:
        return pd.DataFrame(columns=["session_date", "true_push", "inverted_v"])
    dated = bars.copy()
    dated["session_date"] = dated["start"].dt.strftime("%Y-%m-%d")
    for day, frame in dated.groupby("session_date", sort=True):
        done = frame.loc[frame["end"].dt.time.le(pd.Timestamp("11:30").time())]
        close = done["close"].to_numpy(float)
        rows.append({
            "session_date": day,
            "true_push": is_true_push(close),
            "inverted_v": is_inverted_v(close),
        })
    return pd.DataFrame(rows)


def _slice_stats(window: pd.DataFrame, column: str) -> dict:
    days = int(len(window))
    bought = window.loc[window[column]]
    count = int(len(bought))
    base = None if days == 0 else float(window["y_3d_5pct"].mean())
    precision = None if count == 0 else float(bought["y_3d_5pct"].mean())
    return {
        "days": days,
        "buys": count,
        "buy_fraction": None if days == 0 else count / days,
        "base_rate": base,
        "precision": precision,
        "lift": None if precision is None or base is None else precision - base,
    }


def score_symbol(frame: pd.DataFrame) -> dict:
    ordered = frame.sort_values("session_date")
    window = ordered.loc[ordered["session_date"].ge(CONFIRM_START) & ordered["session_date"].le(CONFIRM_END)]
    push = _slice_stats(window, "true_push")
    fade = _slice_stats(window, "inverted_v")
    push["passed"] = bool(
        push["buys"] >= MIN_BUYS and push["precision"] is not None and push["precision"] >= PRECISION_BAR
    )
    return {"true_push": push, "inverted_v": fade}


def run() -> dict:
    panel, coverage = build_panel("2024")
    flags = []
    for symbol in sorted(panel["symbol"].unique()):
        bars, _floor = _regular_from_2024(symbol)
        shaped = morning_flags(bars)
        shaped["symbol"] = symbol
        flags.append(shaped)
    shaped = pd.concat(flags, ignore_index=True) if flags else pd.DataFrame()
    merged = panel.merge(shaped, on=["symbol", "session_date"], how="left")
    merged["true_push"] = merged["true_push"].eq(True)
    merged["inverted_v"] = merged["inverted_v"].eq(True)
    rows = []
    for symbol, frame in merged.groupby("symbol", sort=True):
        result = score_symbol(frame)
        result["symbol"] = symbol
        rows.append(result)
        print(json.dumps(result, default=str), flush=True)
    payload = {
        "rule": "11:30 morning has 24 closes, high at bar 16 or later, fade at most 0.10",
        "contrast": "high at bar 8 or earlier and fade at least 0.70; reported, not switched to after the score",
        "confirm": [CONFIRM_START, CONFIRM_END],
        "coverage": {key: coverage[key] for key in ("rows", "first", "last")},
        "symbols": rows,
        "passed": [item["symbol"] for item in rows if item["true_push"]["passed"]],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "morning_shape.json").write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    done = run()
    print(json.dumps({"passed": done["passed"]}))
