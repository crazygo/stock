"""Paired SOXS/SOXX labels from regular 5-minute bars."""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

DECISION = time(11, 30)
ENTRY = time(11, 35)
HORIZON_MINUTES = 1170
MIN_HISTORY = 40


def regular_bars(frame: pd.DataFrame) -> pd.DataFrame:
    bars = frame.loc[frame["session_type"].eq("regular")].copy()
    bars["start"] = pd.to_datetime(bars["start_at_et"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    bars["end"] = pd.to_datetime(bars["end_at_et"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    bars["minutes"] = ((bars["end"] - bars["start"]).dt.total_seconds() // 60).astype(int)
    bars = bars.loc[bars["minutes"].gt(0)].sort_values("start")
    return bars.reset_index(drop=True)


def horizon_slice(bars: pd.DataFrame, entry_pos: int) -> pd.DataFrame | None:
    acc = 0
    last = entry_pos - 1
    for i in range(entry_pos, len(bars)):
        step = int(bars.iloc[i]["minutes"])
        if acc + step > HORIZON_MINUTES:
            return None
        acc += step
        last = i
        if acc == HORIZON_MINUTES:
            return bars.iloc[entry_pos:last + 1]
    return None


def _path(completed: pd.DataFrame) -> dict | None:
    if len(completed) < MIN_HISTORY:
        return None
    close = completed["close"].to_numpy(float)
    if np.any(close[-MIN_HISTORY:] <= 0) or not np.isfinite(close[-MIN_HISTORY:]).all():
        return None
    window = completed.iloc[-39:]
    span = float(window["high"].max() - window["low"].min())
    volume = completed["volume"].iloc[-40:-1].to_numpy(float)
    med = float(np.median(volume))
    if span <= 0 or med <= 0:
        return None
    day = completed["start"].iloc[-1].normalize()
    today = completed.loc[completed["start"].dt.normalize().eq(day)]
    opened = today.loc[today["start"].dt.time.eq(time(9, 30))]
    if opened.empty or float(opened["open"].iloc[0]) <= 0:
        return None
    return {
        "r_1": float(close[-1] / close[-2] - 1),
        "r_6": float(close[-1] / close[-7] - 1),
        "r_12": float(close[-1] / close[-13] - 1),
        "r_39": float(close[-1] / close[-40] - 1),
        "morning": float(close[-1] / float(opened["open"].iloc[0]) - 1),
        "range_pos": float((close[-1] - window["low"].min()) / span),
        "volume_ratio": float(completed["volume"].iloc[-1] / med),
    }


def _one(bars: pd.DataFrame, day: pd.Timestamp, barrier: float) -> dict | None:
    decision = day + pd.Timedelta(hours=11, minutes=30)
    entry_at = day + pd.Timedelta(hours=11, minutes=35)
    entry_pos = bars.index[bars["start"].eq(entry_at)]
    if len(entry_pos) != 1:
        return None
    pos = int(entry_pos[0])
    window = horizon_slice(bars, pos)
    if window is None:
        return None
    features = _path(bars.loc[bars["end"].le(decision)])
    if features is None:
        return None
    entry = float(bars.iloc[pos]["open"])
    if entry <= 0:
        return None
    high = float(window["high"].max())
    return {
        **features,
        "entry": entry,
        "hit": int(high >= entry * (1 + barrier)),
        "mfe": high / entry - 1,
        "label_end": pd.Timestamp(window["end"].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S"),
    }


def build_rows(soxs: pd.DataFrame, soxx: pd.DataFrame, barrier: float = 0.05) -> tuple[pd.DataFrame, dict]:
    left = regular_bars(soxs)
    right = regular_bars(soxx)
    days = sorted(set(left.loc[left["start"].dt.time.eq(ENTRY), "start"].dt.normalize())
                  & set(right.loc[right["start"].dt.time.eq(ENTRY), "start"].dt.normalize()))
    rows = []
    excluded = {"incomplete": 0}
    for day in days:
        day = pd.Timestamp(day)
        a = _one(left, day, barrier)
        b = _one(right, day, barrier)
        if a is None or b is None:
            excluded["incomplete"] += 1
            continue
        if not a["hit"] and not b["hit"]:
            action = "flat"
        elif a["hit"] and not b["hit"]:
            action = "SOXS"
        elif b["hit"] and not a["hit"]:
            action = "SOXX"
        else:
            action = "SOXS" if a["mfe"] > b["mfe"] else "SOXX"
        row = {"session_date": day.strftime("%Y-%m-%d"), "action": action,
               "y_SOXS": a["hit"], "y_SOXX": b["hit"], "label_end": max(a["label_end"], b["label_end"])}
        for name, item in (("SOXS", a), ("SOXX", b)):
            for key in ("r_1", "r_6", "r_12", "r_39", "morning", "range_pos", "volume_ratio"):
                row[f"{name}_{key}"] = item[key]
        row["morning_spread"] = a["morning"] - b["morning"]
        row["r12_spread"] = a["r_12"] - b["r_12"]
        rows.append(row)
    return pd.DataFrame(rows), excluded
