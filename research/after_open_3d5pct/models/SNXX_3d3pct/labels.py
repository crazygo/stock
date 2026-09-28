"""Hourly-proxy labels for one symbol. Bars are exchange-local and labeled by start."""
from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
DECISION = time(11, 30)
HORIZON_MINUTES = 1170
MIN_HISTORY = 21


def regular_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Regular-session bars with end taken from the next timestamp, capped at 16:00."""
    bars = frame.copy()
    bars["start"] = pd.to_datetime(bars["time_key"])
    bars = bars.sort_values("start").drop_duplicates("start")
    nxt = bars["start"].shift(-1)
    close_cap = bars["start"].dt.normalize() + pd.Timedelta(hours=16)
    same_day = nxt.dt.date.eq(bars["start"].dt.date)
    bars["end"] = nxt.where(same_day & nxt.le(close_cap), close_cap)
    bars["minutes"] = ((bars["end"] - bars["start"]).dt.total_seconds() // 60).astype(int)
    clock = bars["start"].dt.time
    regular = bars[(clock >= RTH_OPEN) & (clock < RTH_CLOSE) & (bars["minutes"] > 0)]
    return regular.reset_index(drop=True)


def horizon_slice(regular: pd.DataFrame, entry_pos: int) -> pd.DataFrame | None:
    """Bars from entry until exactly 1170 regular minutes. A partial bar past the end drops the row."""
    acc = 0
    last = entry_pos - 1
    for i in range(entry_pos, len(regular)):
        step = int(regular.iloc[i]["minutes"])
        if acc + step > HORIZON_MINUTES:
            return None
        acc += step
        last = i
        if acc == HORIZON_MINUTES:
            return regular.iloc[entry_pos:last + 1]
    return None


def _features(completed: pd.DataFrame, day: pd.Timestamp) -> dict | None:
    if len(completed) < MIN_HISTORY:
        return None
    close = completed["close"].to_numpy(float)
    if np.any(close[-MIN_HISTORY:] <= 0) or np.any(~np.isfinite(close[-MIN_HISTORY:])):
        return None
    window = completed.iloc[-20:]
    span = float(window["high"].max() - window["low"].min())
    if span <= 0:
        return None
    prior_vol = completed["volume"].iloc[-21:-1].to_numpy(float)
    med = float(np.median(prior_vol))
    if med <= 0:
        return None
    today = completed[completed["start"].dt.normalize().eq(day)]
    open_bar = today[today["start"].dt.time.eq(RTH_OPEN)]
    ten_thirty = today[today["start"].dt.time.eq(time(10, 30))]
    if open_bar.empty or ten_thirty.empty:
        return None
    previous = completed[completed["start"].dt.normalize().lt(day)]
    if previous.empty or float(open_bar["open"].iloc[0]) <= 0:
        return None
    last_ranges = ((window["high"] - window["low"]) / window["close"]).iloc[-10:]
    out = {
        "r_1": float(close[-1] / close[-2] - 1),
        "r_5": float(close[-1] / close[-6] - 1),
        "r_10": float(close[-1] / close[-11] - 1),
        "r_20": float(close[-1] / close[-21] - 1),
        "morning_return": float(ten_thirty["close"].iloc[0] / open_bar["open"].iloc[0] - 1),
        "overnight_gap": float(open_bar["open"].iloc[0] / previous["close"].iloc[-1] - 1),
        "range_position_20": float((close[-1] - window["low"].min()) / span),
        "volume_ratio_20": float(completed["volume"].iloc[-1] / med),
        "range_mean_10": float(last_ranges.mean()),
    }
    if not np.isfinite(list(out.values())).all():
        return None
    return out


def build_rows(frame: pd.DataFrame, barrier: float = 0.03) -> tuple[pd.DataFrame, dict]:
    """One row per session that has an 11:30 bar. Immature and short-history sessions are counted, not filled."""
    regular = regular_bars(frame)
    dates = sorted(regular.loc[regular["start"].dt.time.eq(DECISION), "start"].dt.normalize().unique())
    rows = []
    excluded = {"no_entry": 0, "immature": 0, "short_history": 0, "bad_features": 0}
    for day in dates:
        day = pd.Timestamp(day)
        decision = day + pd.Timedelta(hours=11, minutes=30)
        entry_pos = regular.index[regular["start"].eq(decision)]
        if len(entry_pos) != 1:
            excluded["no_entry"] += 1
            continue
        pos = int(entry_pos[0])
        window = horizon_slice(regular, pos)
        if window is None:
            excluded["immature"] += 1
            continue
        completed = regular[regular["end"].le(decision)]
        features = _features(completed, day)
        if features is None:
            excluded["short_history" if len(completed) < MIN_HISTORY else "bad_features"] += 1
            continue
        entry = float(regular.iloc[pos]["open"])
        if entry <= 0:
            excluded["bad_features"] += 1
            continue
        label_end = pd.Timestamp(window["end"].iloc[-1])
        rows.append({
            "session_date": day.strftime("%Y-%m-%d"),
            "entry": entry,
            "y": int(float(window["high"].max()) >= entry * (1 + barrier)),
            "label_end": label_end.strftime("%Y-%m-%d %H:%M:%S"),
            **features,
        })
    return pd.DataFrame(rows), excluded
