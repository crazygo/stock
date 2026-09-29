"""Point-in-time rows for the frozen within-stock excess week test."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
HISTORY = ROOT / "market_data/model_training_history_v1"
HORIZONS = {"1d": 390, "3d": 1170, "5d": 1950}
BARRIERS = {"3pct": 0.03, "5pct": 0.05, "8pct": 0.08}
PRIMARY = "3d_5pct"
MIN_BASE = 40
MIN_MORNING = 20
GROUPS = {
    "chips": ["ALAB", "AMD", "ARM", "AVGO", "INTC", "MRVL", "NVDA", "QCOM"],
    "optics": ["LITE", "COHR", "CIEN", "AAOI", "FN", "AXTI", "CRDO", "MRVL", "AVGO", "ALAB", "ANET", "CSCO"],
    "storage": ["MU", "SNDK", "STX", "WDC"],
}


def union_symbols() -> list[str]:
    return list(dict.fromkeys(GROUPS["chips"] + GROUPS["optics"] + GROUPS["storage"]))


def window_end(minutes: np.ndarray, start: int, horizon: int) -> int | None:
    acc = 0
    for i in range(start, len(minutes)):
        step = int(minutes[i])
        if step <= 0 or acc + step > horizon:
            return None
        acc += step
        if acc == horizon:
            return i
    return None


def label_hits(high: np.ndarray, minutes: np.ndarray, entry_pos: int, entry: float) -> dict | None:
    ends = {name: window_end(minutes, entry_pos, horizon) for name, horizon in HORIZONS.items()}
    if ends["3d"] is None:
        return None
    out = {}
    for horizon, last in ends.items():
        for barrier, level in BARRIERS.items():
            key = f"y_{horizon}_{barrier}"
            if last is None:
                out[key] = np.nan
            else:
                out[key] = float(np.nanmax(high[entry_pos:last + 1]) >= entry * (1 + level))
    out["label_end_pos"] = int(ends["3d"])
    return out


def expanding_base(label_ends: list[pd.Timestamp], hits: list[float], decision: pd.Timestamp, minimum: int = MIN_BASE) -> float:
    usable = [hit for end, hit in zip(label_ends, hits) if end <= decision and np.isfinite(hit)]
    if len(usable) < minimum:
        return np.nan
    return float(np.mean(usable))


def _naive_et(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)


def load_calendar() -> pd.DataFrame:
    import json
    payload = json.loads((HISTORY / "calendar.json").read_text())
    frame = pd.DataFrame(payload["sessions"])
    frame["session_date"] = frame["session_date"].astype(str)
    return frame.sort_values("session_date").reset_index(drop=True)


def _daily(symbol: str) -> pd.DataFrame:
    frames = [pd.read_parquet(HISTORY / "derived_day" / symbol / f"{year}.parquet") for year in (2025, 2026)]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame["is_complete"].eq(True)].copy()
    frame["session_date"] = frame["session_date"].astype(str)
    return frame.sort_values("session_date").drop_duplicates("session_date").reset_index(drop=True)


def history_floor(required_from: str | None, year_floor: str = "2024-01-01") -> str:
    if required_from and required_from > year_floor:
        return required_from
    return year_floor


def valid_ohlcv(frame: pd.DataFrame) -> pd.Series:
    prices = frame[["open", "high", "low", "close"]]
    return (
        prices.notna().all(axis=1)
        & frame["volume"].notna()
        & frame["open"].gt(0)
        & frame["high"].gt(0)
        & frame["low"].gt(0)
        & frame["close"].gt(0)
        & frame["volume"].ge(0)
        & frame["high"].ge(prices.max(axis=1))
        & frame["low"].le(prices.min(axis=1))
    )


def complete_daily(bars: pd.DataFrame, expected: dict[str, int]) -> pd.DataFrame:
    columns = ["session_date", "open", "high", "low", "close", "volume", "is_complete"]
    if bars.empty:
        return pd.DataFrame(columns=columns)
    grouped = bars.groupby("session_date", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), n=("start", "nunique"),
    ).reset_index()
    grouped["expected"] = grouped["session_date"].map(expected)
    keep = grouped["n"].eq(grouped["expected"])
    out = grouped.loc[keep, ["session_date", "open", "high", "low", "close", "volume"]].copy()
    out["is_complete"] = True
    return out.reset_index(drop=True)


def _read_5m(path: Path) -> pd.DataFrame:
    return pd.read_parquet(
        path, columns=["start_at_et", "end_at_et", "session_type", "session_date", "open", "high", "low", "close", "volume"]
    )


def _regular(symbol: str) -> pd.DataFrame:
    frames = [_read_5m(HISTORY / "us_5m" / symbol / f"{year}.parquet") for year in (2025, 2026)]
    extra = ROOT / "market_data/us_5m" / symbol / "2026.parquet"
    if extra.exists():
        frames.append(_read_5m(extra))
    return _as_regular(pd.concat(frames, ignore_index=True))


def _listing_floors() -> dict[str, str]:
    import json
    path = HISTORY / "security_history.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return {symbol: row["required_from"] for symbol, row in payload.items() if row.get("required_from")}


def _expected_regular_bars() -> dict[str, int]:
    frame = load_calendar()
    return {str(row.session_date): int(row.duration_minutes) // 5 for row in frame.itertuples(index=False)}


def _as_regular(frame: pd.DataFrame, floor: str | None = None) -> pd.DataFrame:
    frame = frame.loc[frame["session_type"].eq("regular")].copy()
    if floor:
        frame = frame.loc[frame["session_date"].astype(str).ge(floor) & valid_ohlcv(frame)]
    frame["start"] = _naive_et(frame["start_at_et"])
    frame["end"] = _naive_et(frame["end_at_et"])
    frame["minutes"] = ((frame["end"] - frame["start"]).dt.total_seconds() // 60).astype(int)
    frame = frame.loc[frame["minutes"].gt(0)].sort_values("start").drop_duplicates("start")
    return frame.reset_index(drop=True)


def _regular_from_2024(symbol: str) -> tuple[pd.DataFrame, str]:
    floor = history_floor(_listing_floors().get(symbol))
    folder = HISTORY / "parts" / symbol
    paths = []
    if folder.exists():
        paths.extend(path for path in sorted(folder.glob("*/bars.parquet")) if path.parent.name >= floor[:7])
    extra = ROOT / "market_data/us_5m" / symbol / "2026.parquet"
    if extra.exists():
        paths.append(extra)
    if not paths:
        return _as_regular(pd.DataFrame(columns=["session_type", "session_date", "start_at_et", "end_at_et", "open", "high", "low", "close", "volume"])), floor
    return _as_regular(pd.concat([_read_5m(path) for path in paths], ignore_index=True), floor), floor


def _prior_features(daily: pd.DataFrame, day: str) -> dict | None:
    prior = daily.loc[daily["session_date"].lt(day)]
    if len(prior) < 126:
        return None
    close = prior["close"].to_numpy(float)
    volume = prior["volume"].to_numpy(float)
    if np.any(close[-60:] <= 0) or not np.isfinite(close[-60:]).all():
        return None
    high20 = float(np.max(prior["high"].to_numpy(float)[-20:]))
    vol20 = float(np.mean(volume[-20:]))
    return {
        "full_126": int(len(prior) >= 126),
        "ret_5": close[-1] / close[-6] - 1,
        "ret_20": close[-1] / close[-21] - 1,
        "ret_60": close[-1] / close[-61] - 1,
        "drawdown_20": close[-1] / high20 - 1,
        "volume_ratio_20": volume[-1] / vol20 if vol20 > 0 else np.nan,
    }


def _morning(day_bars: pd.DataFrame) -> dict | None:
    done = day_bars.loc[day_bars["end"].dt.time.le(pd.Timestamp("11:30").time())]
    if len(done) < MIN_MORNING:
        return None
    close = done["close"].to_numpy(float)
    open0 = float(done["open"].iloc[0])
    if open0 <= 0 or np.any(close <= 0):
        return None
    returns = np.diff(close) / close[:-1]
    return {
        "morning_return": close[-1] / open0 - 1,
        "morning_vol": float(np.std(returns)) if len(returns) else np.nan,
        "last6_return": close[-1] / close[-7] - 1 if len(close) >= 7 else np.nan,
    }


def build_symbol(symbol: str, history: str = "short") -> tuple[pd.DataFrame, dict]:
    floor = None
    if history == "short":
        daily = _daily(symbol)
        bars = _regular(symbol)
    elif history == "2024":
        bars, floor = _regular_from_2024(symbol)
        if len(bars):
            bars["session_date"] = bars["start"].dt.strftime("%Y-%m-%d")
        daily = complete_daily(bars, _expected_regular_bars())
    else:
        raise ValueError(history)
    if history == "short":
        bars["session_date"] = bars["start"].dt.strftime("%Y-%m-%d")
    high = bars["high"].to_numpy(float)
    minutes = bars["minutes"].to_numpy(int)
    ends = list(bars["end"])
    rows = []
    matured_ends: list[pd.Timestamp] = []
    matured_hits: list[float] = []
    decision_days = 0
    ready_days = 0
    by_day = {day: frame for day, frame in bars.groupby("session_date", sort=True)}
    for day, frame in by_day.items():
        entry_rows = frame.loc[frame["start"].dt.time.eq(pd.Timestamp("11:35").time())]
        if entry_rows.empty:
            continue
        entry_pos = int(entry_rows.index[0])
        entry = float(bars.at[entry_pos, "open"])
        if not np.isfinite(entry) or entry <= 0:
            continue
        if day >= "2026-03-02":
            decision_days += 1
            if int((daily["session_date"] < day).sum()) >= 126:
                ready_days += 1
        labels = label_hits(high, minutes, entry_pos, entry)
        features = _prior_features(daily, day)
        morning = _morning(frame)
        decision = pd.Timestamp(f"{day} 11:30")
        base = expanding_base(matured_ends, matured_hits, decision)
        if labels is not None:
            label_end = pd.Timestamp(ends[labels.pop("label_end_pos")])
            matured_ends.append(label_end)
            matured_hits.append(labels["y_3d_5pct"])
        else:
            label_end = pd.NaT
        if labels is None or features is None or morning is None or not features["full_126"]:
            continue
        if not np.isfinite(base):
            continue
        row = {"symbol": symbol, "session_date": day, "decision_at": decision, "entry": entry,
               "own_base": base, "label_end": label_end, **features, **morning, **labels}
        rows.append(row)
    coverage = {
        "symbol": symbol,
        "history": history,
        "floor": floor,
        "decision_days_from_20260302": decision_days,
        "full_126_days": ready_days,
        "complete_days": int(len(daily)),
        "first_complete": None if daily.empty else str(daily["session_date"].min()),
        "last_complete": None if daily.empty else str(daily["session_date"].max()),
        "rows": len(rows),
    }
    return pd.DataFrame(rows), coverage


def build_panel(history: str = "short") -> tuple[pd.DataFrame, dict]:
    built = [build_symbol(symbol, history) for symbol in union_symbols()]
    frames = [frame for frame, _ in built if len(frame)]
    per_symbol = [item for _, item in built]
    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    decision_days = sum(item["decision_days_from_20260302"] for item in per_symbol)
    ready_days = sum(item["full_126_days"] for item in per_symbol)
    coverage = {
        "history": history,
        "symbols": union_symbols(),
        "rows": int(len(panel)),
        "first": None if panel.empty else str(panel["session_date"].min()),
        "last": None if panel.empty else str(panel["session_date"].max()),
        "decision_days_from_20260302": decision_days,
        "full_126_days": ready_days,
        "full_126_fraction": ready_days / decision_days if decision_days else None,
        "per_symbol": per_symbol,
    }
    return panel, coverage
