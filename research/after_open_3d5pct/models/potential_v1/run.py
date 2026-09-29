"""Train the morning-push times 30-day-trend hypothesis on the eight-week walk."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.break_v1.run import (
    MIN_TRAIN,
    TRAIN_WEEKS,
    walk,
    window_days,
)
from research.after_open_3d5pct.models.curve_split.run import curve_geometry, zscore
from research.after_open_3d5pct.models.pugh_ge5.build_rows import ROOT, build_panel
from research.after_open_3d5pct.models.pugh_ge5.walk import FEATURES
from research.after_open_3d5pct.models.weekly_scale.run import (
    MACRO,
    MICRO,
    RELATION,
    _load_all,
    attach,
    matured_before,
    week_blocks,
)

MACRO_DAYS = 30
OUT = ROOT / "research/after_open_3d5pct/runs/potential_v1"
POTENTIAL = ["impulse", "macro_30d_beta", "macro_30d_kappa", "potential"]
RELATION_FEATURES = list(FEATURES) + MICRO + MACRO + RELATION
VARIANTS = {
    "baseline": list(FEATURES),
    "potential": list(FEATURES) + POTENTIAL,
    "relation": RELATION_FEATURES,
    "both": RELATION_FEATURES + POTENTIAL,
}


def impulse_from_path(start_open: float, closes: np.ndarray) -> float:
    """Signed morning move, shrunk when the 5-minute path chops."""
    if start_open <= 0 or len(closes) < 2:
        return np.nan
    path = np.concatenate([[float(start_open)], np.asarray(closes, float)])
    if np.any(path <= 0) or not np.isfinite(path).all():
        return np.nan
    net = float(path[-1] / path[0] - 1)
    steps = np.diff(path) / path[:-1]
    travelled = float(np.abs(steps).sum())
    if travelled < 1e-12 or not np.isfinite(net):
        return np.nan
    cleanliness = min(abs(net) / travelled, 1.0)
    return net * cleanliness


def macro_shape(closes_before_today: np.ndarray) -> tuple[float, float] | None:
    """Slope and curvature of the last 30 daily closes. Today is not included."""
    if len(closes_before_today) < MACRO_DAYS:
        return None
    chosen = np.asarray(closes_before_today[-MACRO_DAYS:], float)
    scaled = zscore(chosen)
    if scaled is None:
        return None
    beta, kappa, _peak, _trough, _fade = curve_geometry(scaled)
    return beta, kappa


def add_potential(frame: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for symbol, group in frame.groupby("symbol", sort=True):
        bars = _load_all(symbol)
        if bars.empty:
            continue
        regular = bars.loc[bars["session_type"].eq("regular")].copy()
        regular["day"] = regular["start"].dt.strftime("%Y-%m-%d")
        daily = (
            regular.groupby("day", sort=True)
            .agg(close=("close", "last"))
            .reset_index()
        )
        day_index = {day: index for index, day in enumerate(daily["day"])}
        closes = daily["close"].to_numpy(float)
        rows = []
        for day in group["session_date"]:
            decision = pd.Timestamp(f"{day} 11:30")
            morning = regular.loc[regular["day"].eq(day) & regular["end"].le(decision)].sort_values("end")
            impulse = np.nan
            if len(morning) == 24 and pd.Timestamp(morning["end"].iloc[-1]) == decision:
                impulse = impulse_from_path(float(morning["open"].iloc[0]), morning["close"].to_numpy(float))
            beta = kappa = potential = np.nan
            index = day_index.get(day)
            if index is not None:
                shape = macro_shape(closes[:index])
                if shape is not None and np.isfinite(impulse):
                    beta, kappa = shape
                    potential = float(impulse) * float(beta)
                elif shape is not None:
                    beta, kappa = shape
            rows.append({
                "symbol": symbol,
                "session_date": day,
                "impulse": impulse,
                "macro_30d_beta": beta,
                "macro_30d_kappa": kappa,
                "potential": potential,
            })
        built = pd.DataFrame(rows)
        finite = int(np.isfinite(built["potential"]).sum())
        print(json.dumps({"symbol": symbol, "potential_rows": finite, "rows": int(len(built))}), flush=True)
        frames.append(built)
    extra = pd.concat(frames, ignore_index=True)
    return frame.merge(extra, on=["symbol", "session_date"], how="left")


def sign_hits(frame: pd.DataFrame, dates: list[str]) -> dict:
    """Buy only when the morning push and the 30-day slope are both up."""
    weeks = week_blocks(dates)
    pooled = {"test": [], "validation": []}
    used = 0
    for index in range(TRAIN_WEEKS, len(weeks) - 1):
        train_days, test_days, val_days = window_days(weeks, index)
        before = pd.Timestamp(f"{test_days[0]} 11:30")
        train = matured_before(frame, train_days, before)
        if len(train) < MIN_TRAIN or train["y_3d_5pct"].nunique() < 2:
            continue
        used += 1
        for key, days in (("test", test_days), ("validation", val_days)):
            pooled[key].append(frame.loc[frame["session_date"].isin(days)])

    def _pack(parts: list[pd.DataFrame]) -> dict:
        if not parts:
            return {"rows": 0, "buys": 0, "precision": None, "base_rate": None, "lift": None}
        block = pd.concat(parts, ignore_index=True)
        buy = block["impulse"].gt(0) & block["macro_30d_beta"].gt(0)
        base = float(block["y_3d_5pct"].mean())
        precision = None if not buy.any() else float(block.loc[buy, "y_3d_5pct"].mean())
        return {
            "rows": int(len(block)),
            "base_rate": base,
            "buys": int(buy.sum()),
            "precision": precision,
            "lift": None if precision is None else precision - base,
        }

    return {"fitted_triplets": used, "test": _pack(pooled["test"]), "validation": _pack(pooled["validation"])}


def run() -> dict:
    panel, coverage = build_panel("2024")
    frame = add_potential(attach(panel))
    dates = sorted(frame["session_date"].unique())
    models = {}
    for name, features in VARIANTS.items():
        models[name] = walk(frame, features, dates)
        print(json.dumps({
            "model": name,
            "test": models[name]["test"],
            "validation": models[name]["validation"],
        }, default=str), flush=True)
    rule = sign_hits(frame, dates)
    print(json.dumps({"model": "sign_rule", **rule}, default=str), flush=True)
    payload = {
        "train_weeks": TRAIN_WEEKS,
        "coverage": {key: coverage[key] for key in ("rows", "first", "last")},
        "potential_finite": int(np.isfinite(frame["potential"]).sum()),
        "rows": int(len(frame)),
        "models": models,
        "sign_rule": rule,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload


if __name__ == "__main__":
    run()
