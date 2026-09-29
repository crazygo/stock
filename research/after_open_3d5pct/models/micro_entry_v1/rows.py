"""Sliding regular-session windows. The label starts at the next bar's open."""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.pugh_ge5.build_rows import (
    _regular_from_2024,
    label_hits,
    union_symbols,
    window_end,
)
from research.after_open_3d5pct.models.curve_split.run import curve_geometry, zscore
from research.after_open_3d5pct.models.potential_v1.run import impulse_from_path, macro_shape

HORIZON = 1170
WIDTH = HORIZON // 5
FEATURES = [
    "impulse",
    "micro_peak",
    "micro_fade",
    "macro_30d_beta",
    "macro_30d_kappa",
    "potential",
]


def sliding_window_max(values: np.ndarray, width: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if width <= 0 or len(values) < width:
        return out
    queue: deque[int] = deque()
    for index, value in enumerate(values):
        while queue and queue[0] <= index - width:
            queue.popleft()
        while queue and values[queue[-1]] <= value:
            queue.pop()
        queue.append(index)
        if index >= width - 1:
            out[index] = values[queue[0]]
    return out


def horizon_last(minutes: np.ndarray, entry: int, horizon: int = HORIZON) -> int | None:
    """Last bar of an exact regular-minute horizon. Matches window_end."""
    last = entry + horizon // 5 - 1
    if last < len(minutes) and np.all(minutes[entry:last + 1] == 5):
        return last
    return window_end(minutes, entry, horizon)


def rows_from_bars(bars: pd.DataFrame, length: int, step: int, symbol: str) -> pd.DataFrame:
    if bars.empty or length < 2 or step < 1:
        return pd.DataFrame()
    frame = bars.sort_values("start").reset_index(drop=True)
    day = frame["start"].dt.strftime("%Y-%m-%d").to_numpy()
    start = frame["start"].to_numpy()
    end = frame["end"].to_numpy()
    open_ = frame["open"].to_numpy(float)
    high = frame["high"].to_numpy(float)
    close = frame["close"].to_numpy(float)
    minutes = frame["minutes"].to_numpy(int)
    roll_max = sliding_window_max(high, WIDTH)
    daily = frame.groupby(frame["start"].dt.strftime("%Y-%m-%d"), sort=True)["close"].last()
    prior_close = daily.to_numpy(float)
    prior_day = {name: index for index, name in enumerate(daily.index)}
    contiguous = np.zeros(len(frame), dtype=bool)
    contiguous[1:] = start[1:] == end[:-1]
    columns: dict[str, list] = {name: [] for name in (
        "symbol", "session_date", "decision_at", "entry_at", "entry", "y_3d_5pct", "label_end", *FEATURES,
    )}
    left = 0
    while left < len(frame):
        right = left
        while right + 1 < len(frame) and day[right + 1] == day[left]:
            right += 1
        shape = macro_shape(prior_close[:prior_day[day[left]]]) if day[left] in prior_day else None
        beta = kappa = np.nan
        if shape is not None:
            beta, kappa = shape
        for end_index in range(left + length - 1, right, step):
            entry_index = end_index + 1
            window_start = end_index - length + 1
            if entry_index > right or window_start < left:
                continue
            if not contiguous[window_start + 1:end_index + 1].all() or start[entry_index] != end[end_index]:
                continue
            last = horizon_last(minutes, entry_index)
            if last is None:
                continue
            entry = float(open_[entry_index])
            if not np.isfinite(entry) or entry <= 0:
                continue
            if last == entry_index + WIDTH - 1 and np.isfinite(roll_max[last]):
                future_high = float(roll_max[last])
            else:
                future_high = float(np.nanmax(high[entry_index:last + 1]))
            window_close = close[window_start:end_index + 1]
            impulse = impulse_from_path(float(open_[window_start]), window_close)
            scaled = zscore(window_close)
            peak = fade = np.nan
            if scaled is not None:
                _beta, _kappa, peak, _trough, fade = curve_geometry(scaled)
            potential = impulse * beta if np.isfinite(impulse) and np.isfinite(beta) else np.nan
            columns["symbol"].append(symbol)
            columns["session_date"].append(day[left])
            columns["decision_at"].append(end[end_index])
            columns["entry_at"].append(start[entry_index])
            columns["entry"].append(entry)
            columns["y_3d_5pct"].append(float(future_high >= entry * 1.05))
            columns["label_end"].append(end[last])
            columns["impulse"].append(impulse)
            columns["micro_peak"].append(peak)
            columns["micro_fade"].append(fade)
            columns["macro_30d_beta"].append(beta)
            columns["macro_30d_kappa"].append(kappa)
            columns["potential"].append(potential)
        left = right + 1
    return pd.DataFrame(columns)


def build_frame(length: int, step: int = 1) -> pd.DataFrame:
    frames = []
    for symbol in union_symbols():
        bars, _floor = _regular_from_2024(symbol)
        built = rows_from_bars(bars, length, step, symbol)
        print(f"{symbol} length={length} rows={len(built)}", flush=True)
        if len(built):
            frames.append(built)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def agrees_with_label_hits(minutes: np.ndarray, high: np.ndarray, entry_index: int, entry: float) -> bool:
    """Used by tests. The fast horizon and label_hits name the same last bar and the same hit."""
    last = horizon_last(minutes, entry_index)
    labels = label_hits(high, minutes, entry_index, entry)
    if last is None or labels is None:
        return last is None and labels is None
    return int(labels["label_end_pos"]) == last
