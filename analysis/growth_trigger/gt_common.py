"""Shared data layer for the growth-trigger study.

All price/volume features and the +8% touch test use the Futu hourly archive only.
The Massive daily archive is used for point-in-time eligibility and for the 8% labels.
"""

from __future__ import annotations

import gzip
import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DAILY_DIR = ROOT / "market_data/us"
HOURLY_DIR = ROOT / "market_data/us_60m"
PINNED_PATH = ROOT / "analysis/preopen_three_week/pinned_stocks.json"

# Futu hourly bars are labelled with the bar END time in US/Eastern.
PRE = ("05:00", "09:30")      # 04:00-09:30 ET
REG = ("10:30", "16:00")     # 09:30-16:00 ET
POST = ("17:00", "20:00")    # 16:00-20:00 ET
PREDICT_LABELS = [f"{h:02d}:{m:02d}" for h, m in
                  [(5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (9, 30), (10, 30), (11, 30),
                   (12, 30), (13, 30), (14, 30), (15, 30), (16, 0), (17, 0), (18, 0),
                   (19, 0), (20, 0)]]
TOUCH_TARGET = 0.08
TOUCH_MIN_DAYS = 3
TOUCH_MAX_DAYS = 14


def _parse(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")


def label_of(ts: str) -> str:
    return ts[11:16]


def session_of(ts: str) -> str:
    lab = label_of(ts)
    if PRE[0] <= lab <= PRE[1]:
        return "pre"
    if REG[0] <= lab <= REG[1]:
        return "reg"
    if POST[0] <= lab <= POST[1]:
        return "post"
    return "night"


@dataclass
class HourlySeries:
    ticker: str
    ts: list[datetime]
    o: list[float]
    h: list[float]
    l: list[float]
    c: list[float]
    v: list[float]
    sess: list[str]
    # sparse table for range-max of highs
    _log: list[int] = field(default_factory=list, repr=False)
    _table: list[list[float]] = field(default_factory=list, repr=False)

    def __post_init__(self):
        n = len(self.ts)
        log = [0] * (n + 1)
        for i in range(2, n + 1):
            log[i] = log[i // 2] + 1
        self._log = log
        table = [list(self.h)]
        j = 1
        while (1 << j) <= n:
            prev = table[-1]
            size = n - (1 << j) + 1
            row = [max(prev[i], prev[i + (1 << (j - 1))]) for i in range(max(0, size))]
            table.append(row)
            j += 1
        self._table = table

    def index_at(self, moment: datetime) -> int | None:
        i = bisect_left(self.ts, moment)
        if i < len(self.ts) and self.ts[i] == moment:
            return i
        return None

    def max_high(self, lo: datetime, hi: datetime) -> float | None:
        a = bisect_left(self.ts, lo)
        b = bisect_right(self.ts, hi)
        if a >= b:
            return None
        b -= 1
        k = self._log[b - a + 1]
        row = self._table[k]
        return max(row[a], row[b - (1 << k) + 1])


@lru_cache(maxsize=None)
def load_hourly(ticker: str) -> HourlySeries:
    path = HOURLY_DIR / ticker / "2026.json.gz"
    if not path.exists():
        return HourlySeries(ticker, [], [], [], [], [], [])
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)
    bars = sorted(payload["bars"], key=lambda b: b["time_key"])
    return HourlySeries(
        ticker=ticker,
        ts=[_parse(b["time_key"]) for b in bars],
        o=[float(b["open"]) for b in bars],
        h=[float(b["high"]) for b in bars],
        l=[float(b["low"]) for b in bars],
        c=[float(b["close"]) for b in bars],
        v=[float(b["volume"]) for b in bars],
        sess=[session_of(b["time_key"]) for b in bars],
    )


@lru_cache(maxsize=None)
def pinned_tickers() -> tuple[str, ...]:
    payload = json.loads(PINNED_PATH.read_text())
    return tuple(payload["tickers"])


@lru_cache(maxsize=None)
def daily_index() -> dict[str, dict[str, dict]]:
    out = {}
    for day in sorted(p.name for p in DAILY_DIR.iterdir() if p.is_dir()):
        with gzip.open(DAILY_DIR / day / "grouped.json.gz", "rt") as handle:
            out[day] = {r["T"]: r for r in json.load(handle)["results"]}
    return out


@lru_cache(maxsize=None)
def load_daily_market():
    """Point-in-time eligibility + quality from the existing backtest engine."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from backtest_ai_strategies import Backtest
    from preopen_three_week import FundMembership

    return Backtest(
        DAILY_DIR, ROOT / "ai_universe_candidates.json",
        ROOT / "data/ai_sec_annual_facts.json",
        extra_tickers=FundMembership(ROOT / "data/ai_fund_holdings.json").all_tickers
        | set(pinned_tickers()),
        extra_snapshot=ROOT / "data/futu_snapshot_2026-09-22.json",
        minimum_history=60,
    )


def regular_session_closes(series: HourlySeries) -> dict[str, dict]:
    """Per ET trading date: regular-session open/high/low/close/volume from hourly bars."""
    out: dict[str, dict] = {}
    cur = None
    for i, ts in enumerate(series.ts):
        if series.sess[i] != "reg":
            continue
        d = ts.strftime("%Y-%m-%d")
        bar = {"o": series.o[i], "h": series.h[i], "l": series.l[i],
               "c": series.c[i], "v": series.v[i]}
        if cur is None or cur["d"] != d:
            cur = {"d": d, "o": bar["o"], "h": bar["h"], "l": bar["l"],
                   "c": bar["c"], "v": bar["v"]}
            out[d] = cur
        else:
            cur["h"] = max(cur["h"], bar["h"])
            cur["l"] = min(cur["l"], bar["l"])
            cur["c"] = bar["c"]
            cur["v"] += bar["v"]
    return out


def touch_outcome(series: HourlySeries, idx: int,
                  target: float = TOUCH_TARGET,
                  min_days: int = TOUCH_MIN_DAYS,
                  max_days: int = TOUCH_MAX_DAYS,
                  basis: str = "high", use_extended: bool = True) -> dict | None:
    """Outcome of buying at the close of bar *idx*: did price reach +target within the window?

    basis="high"  -> best intraday high of each bar (default, user's touch definition)
    basis="close" -> best bar close (stricter, close-only variant)
    use_extended=False restricts to regular-session bars only.
    """
    entry = series.c[idx]
    if entry <= 0:
        return None
    t = series.ts[idx]
    lo = t + timedelta(days=min_days)
    hi = t + timedelta(days=max_days)
    best, best_ts = None, None
    for j in range(idx + 1, len(series.ts)):
        if series.ts[j] < lo:
            continue
        if series.ts[j] > hi:
            break
        if not use_extended and series.sess[j] != "reg":
            continue
        px = series.h[j] if basis == "high" else series.c[j]
        if best is None or px > best:
            best, best_ts = px, series.ts[j]
    if best is None:
        return None
    return {"entry": entry, "best": best, "best_ts": best_ts.isoformat(sep=" "),
            "max_return": best / entry - 1, "hit": best / entry - 1 >= target}
