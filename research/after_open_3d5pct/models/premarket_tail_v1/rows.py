"""One row per symbol-day. Entry is the 09:30 premarket close."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.after_open_3d5pct.models.curve_split.run import curve_geometry, zscore
from research.after_open_3d5pct.models.potential_v1.run import impulse_from_path
from research.after_open_3d5pct.models.pugh_ge5.build_rows import (
    ROOT,
    _naive_et,
    label_hits,
    union_symbols,
    valid_ohlcv,
)
from research.after_open_3d5pct.models.weekly_scale.run import (
    MIN_OVERNIGHT,
    MIN_PREMARKET,
    _load_all,
)

HORIZON = 1170
FEATURES = ["overnight_return", "premarket_return", "path_impulse", "path_fade"]
FOLLOW_UP = ["path_cleanliness", "path_peak"]
OVERNIGHT_DIR = ROOT / "research/after_open_3d5pct/runs/premarket_tail_v1/overnight"
COLUMNS = [
    "symbol",
    "session_date",
    "decision_at",
    "entry",
    "regular_open",
    "y_3d_5pct",
    "label_end",
    *FEATURES,
    *FOLLOW_UP,
    "n_overnight",
    "n_premarket",
]


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _segment_return(open_: np.ndarray, close: np.ndarray, mask: np.ndarray, minimum: int) -> tuple[float, int]:
    where = np.flatnonzero(mask)
    count = int(len(where))
    if count < minimum:
        return np.nan, count
    first = float(open_[where[0]])
    last = float(close[where[-1]])
    if first <= 0 or last <= 0 or not np.isfinite(first) or not np.isfinite(last):
        return np.nan, count
    return last / first - 1.0, count


def _path_stats(anchor: float, closes: np.ndarray) -> tuple[float, float, float, float]:
    impulse = float(impulse_from_path(anchor, closes))
    cleanliness = np.nan
    fade = np.nan
    peak = np.nan
    if anchor > 0 and len(closes) >= 2:
        path = np.concatenate([[float(anchor)], np.asarray(closes, float)])
        if np.all(np.isfinite(path)) and np.all(path > 0):
            net = float(path[-1] / path[0] - 1.0)
            steps = np.diff(path) / path[:-1]
            travelled = float(np.abs(steps).sum())
            if travelled >= 1e-12 and np.isfinite(net):
                cleanliness = min(abs(net) / travelled, 1.0)
    scaled = zscore(np.asarray(closes, float))
    if scaled is not None and len(scaled) >= 2:
        _beta, _kappa, peak, _trough, fade = curve_geometry(scaled)
    return impulse, cleanliness, fade, peak


def _clock_index(stamps: pd.Series, hour: int, minute: int) -> dict[str, int]:
    found: dict[str, int] = {}
    clock = stamps.dt.hour.eq(hour) & stamps.dt.minute.eq(minute) & stamps.dt.second.eq(0)
    for index in np.flatnonzero(clock.to_numpy()):
        found[pd.Timestamp(stamps.iloc[index]).strftime("%Y-%m-%d")] = int(index)
    return found


def rows_from_bars(symbol: str, bars: pd.DataFrame) -> pd.DataFrame:
    """Evening bars keep their own calendar date. The path is joined by end time."""
    if bars.empty:
        return _empty()
    frame = (
        bars.dropna(subset=["start", "end"])
        .sort_values(["end", "start"])
        .drop_duplicates("start")
        .reset_index(drop=True)
    )
    frame = frame.loc[frame["end"].gt(frame["start"])].reset_index(drop=True)
    if frame.empty:
        return _empty()
    regular = frame.loc[frame["session_type"].eq("regular")].reset_index(drop=True)
    if regular.empty:
        return _empty()
    day_labels = regular["start"].dt.strftime("%Y-%m-%d").to_numpy()
    ordered = list(dict.fromkeys(day_labels))
    last_at = {}
    for index, day in enumerate(day_labels):
        last_at[day] = index
    open_at = _clock_index(regular["start"], 9, 30)
    tail_at: dict[str, int] = {}
    pre_rows = np.flatnonzero(frame["session_type"].eq("pre_market").to_numpy())
    pre_ends = frame["end"].to_numpy()
    for pos in pre_rows:
        ts = pd.Timestamp(pre_ends[pos])
        if ts.hour == 9 and ts.minute == 30 and ts.second == 0:
            tail_at[ts.strftime("%Y-%m-%d")] = int(pos)
    ends = frame["end"].to_numpy()
    session = frame["session_type"].to_numpy()
    open_ = frame["open"].to_numpy(float)
    close = frame["close"].to_numpy(float)
    reg_open = regular["open"].to_numpy(float)
    reg_high = regular["high"].to_numpy(float)
    reg_end = regular["end"].to_numpy()
    reg_close = regular["close"].to_numpy(float)
    reg_minutes = ((regular["end"] - regular["start"]).dt.total_seconds() // 60).to_numpy(dtype=float)
    reg_minutes = np.where(np.isfinite(reg_minutes), reg_minutes, 0).astype(int)
    rows = []
    for index, day in enumerate(ordered):
        if index == 0 or day not in open_at or day not in tail_at:
            continue
        previous = ordered[index - 1]
        anchor = float(reg_close[last_at[previous]])
        entry = float(close[tail_at[day]])
        if not np.isfinite(anchor) or anchor <= 0 or not np.isfinite(entry) or entry <= 0:
            continue
        prev_end = np.datetime64(pd.Timestamp(reg_end[last_at[previous]]))
        decision = np.datetime64(pd.Timestamp(f"{day} 09:30"))
        left = int(np.searchsorted(ends, prev_end, side="right"))
        right = int(np.searchsorted(ends, decision, side="right"))
        chosen = np.zeros(right - left, dtype=bool)
        if right > left:
            kind = session[left:right]
            chosen = (kind == "overnight") | (kind == "pre_market")
        path_open = open_[left:right][chosen]
        path_close = close[left:right][chosen]
        kind = session[left:right][chosen]
        overnight_return, n_overnight = _segment_return(path_open, path_close, kind == "overnight", MIN_OVERNIGHT)
        premarket_return, n_premarket = _segment_return(path_open, path_close, kind == "pre_market", MIN_PREMARKET)
        impulse, cleanliness, fade, peak = _path_stats(anchor, path_close)
        entry_pos = open_at[day]
        hits = label_hits(reg_high, reg_minutes, entry_pos, entry)
        if hits is None:
            continue
        rows.append({
            "symbol": symbol,
            "session_date": day,
            "decision_at": pd.Timestamp(f"{day} 09:30"),
            "entry": entry,
            "regular_open": float(reg_open[entry_pos]),
            "y_3d_5pct": hits["y_3d_5pct"],
            "label_end": pd.Timestamp(reg_end[hits["label_end_pos"]]),
            "overnight_return": overnight_return,
            "premarket_return": premarket_return,
            "path_impulse": impulse,
            "path_fade": fade,
            "path_cleanliness": cleanliness,
            "path_peak": peak,
            "n_overnight": n_overnight,
            "n_premarket": n_premarket,
        })
    if not rows:
        return _empty()
    return pd.DataFrame(rows, columns=COLUMNS)


def _read_overnight(path: Path) -> pd.DataFrame:
    extra = pd.read_parquet(path)
    if extra.empty:
        return extra
    extra = extra.loc[valid_ohlcv(extra)].copy()
    extra["start"] = _naive_et(extra["start_at_et"])
    extra["end"] = _naive_et(extra["end_at_et"])
    return extra.loc[extra["end"].gt(extra["start"])]


def load_symbol(symbol: str) -> pd.DataFrame:
    frames = []
    base = _load_all(symbol)
    if not base.empty:
        frames.append(base)
    if OVERNIGHT_DIR.exists():
        for path in sorted(OVERNIGHT_DIR.glob(f"{symbol}_*.parquet")):
            extra = _read_overnight(path)
            if len(extra):
                frames.append(extra)
        single = OVERNIGHT_DIR / f"{symbol}.parquet"
        if single.exists():
            extra = _read_overnight(single)
            if len(extra):
                frames.append(extra)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    return frame.sort_values("start").drop_duplicates("start").reset_index(drop=True)


def build_frame(symbols: list[str] | None = None) -> pd.DataFrame:
    import json
    frames = []
    for symbol in symbols or union_symbols():
        built = rows_from_bars(symbol, load_symbol(symbol))
        both = 0 if built.empty else int((built["overnight_return"].notna() & built["premarket_return"].notna()).sum())
        print(json.dumps({"symbol": symbol, "rows": int(len(built)), "both_finite": both}), flush=True)
        if len(built):
            frames.append(built)
    if not frames:
        return _empty()
    return pd.concat(frames, ignore_index=True)
