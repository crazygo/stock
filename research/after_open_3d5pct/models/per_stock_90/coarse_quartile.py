"""Buy only when today's morning vol is inside the stock's own past top quartile."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.pugh_ge5.build_rows import ROOT, build_panel

MIN_PAST = 40
QUANTILE = 0.75
MAX_BUY_FRACTION = 0.50
MIN_BUYS = 15
CONFIRM_START, CONFIRM_END = "2026-01-02", "2026-09-18"
OUT = ROOT / "research/after_open_3d5pct/runs/per_stock_90"


def top_quartile_mask(vol: np.ndarray, minimum: int = MIN_PAST, quantile: float = QUANTILE) -> np.ndarray:
    buy = np.zeros(len(vol), dtype=bool)
    for index in range(minimum, len(vol)):
        level = float(np.quantile(vol[:index], quantile))
        buy[index] = vol[index] > level
    return buy


def score_symbol(frame: pd.DataFrame) -> dict:
    ordered = frame.sort_values("session_date").copy()
    ordered["buy"] = top_quartile_mask(ordered["morning_vol"].to_numpy(float))
    window = ordered.loc[ordered["session_date"].ge(CONFIRM_START) & ordered["session_date"].le(CONFIRM_END)].copy()
    days = int(len(window))
    buys = window.loc[window["buy"]]
    count = int(len(buys))
    base = None if days == 0 else float(window["y_3d_5pct"].mean())
    precision = None if count == 0 else float(buys["y_3d_5pct"].mean())
    fraction = None if days == 0 else count / days
    lift = None if precision is None or base is None else precision - base
    clears_cap = fraction is not None and fraction <= MAX_BUY_FRACTION
    beats_random = bool(count >= MIN_BUYS and clears_cap and lift is not None and lift > 0)
    return {
        "days": days,
        "buys": count,
        "buy_fraction": fraction,
        "base_rate": base,
        "precision": precision,
        "lift": lift,
        "beats_random": beats_random,
    }


def run() -> dict:
    panel, coverage = build_panel("2024")
    rows = []
    for symbol, frame in panel.groupby("symbol", sort=True):
        result = score_symbol(frame)
        result["symbol"] = symbol
        rows.append(result)
        print(json.dumps(result, default=str), flush=True)
    payload = {
        "rule": "morning_vol above the stock's own prior 75th percentile, at least 40 prior days",
        "confirm": [CONFIRM_START, CONFIRM_END],
        "coverage": {key: coverage[key] for key in ("rows", "first", "last")},
        "symbols": rows,
        "beats_random": [item["symbol"] for item in rows if item["beats_random"]],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coarse_quartile.json").write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    done = run()
    print(json.dumps({"beats_random": done["beats_random"]}))
