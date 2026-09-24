"""Observation points and the five target labels (spec_v1.md M0 + M3).

Everything here is label-side: it is allowed to look at the future, because the whole
point is to describe what happened after an observation point.  Features must come from
``causal_features`` / ``causal_state`` instead, which never look ahead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from segments import (Bar, Segment, atr_scale, build_segments, causal_state,
                      trailing_atr_series)

REGULAR_LABELS = ("10:30", "11:30", "12:30", "13:30", "14:30", "15:30", "16:00")
SESSION_MINUTES = 390           # 09:30-16:00 ET
TARGETS = (0.04, 0.05, 0.06, 0.07, 0.08)
DOWN_BARRIERS = (0.03, 0.05, 0.08)
HORIZON_SESSIONS = 3


def regular_bars(bars: list[Bar]) -> list[Bar]:
    """Keep only regular-session bars (labels 10:30 .. 16:00, i.e. 09:30-16:00 ET)."""
    return [b for b in bars if b.ts.strftime("%H:%M") in REGULAR_LABELS]


def bar_open(bar: Bar, prev: Bar | None) -> datetime:
    """Open time of *bar*: label minus the bar's own length.

    Regular-session hourly bars are labelled with their END time in ET.  The last bar
    of a session (label 16:00) covers only 15:30-16:00, so it is 30 minutes long.
    """
    if prev is not None and bar.ts - prev.ts <= timedelta(minutes=30):
        return prev.ts
    return bar.ts - timedelta(hours=1)


def sessions(bars: list[Bar]) -> list[tuple[str, datetime, datetime]]:
    """[(session_date, open, close)] for the regular-session bars, in order."""
    out = []
    for i, b in enumerate(bars):
        d = b.ts.strftime("%Y-%m-%d")
        if out and out[-1][0] == d:
            out[-1] = (d, out[-1][1], b.ts)
        else:
            out.append((d, bar_open(b, bars[i - 1] if i else None), b.ts))
    return out


def horizon_end(entry_open: datetime, sess: list[tuple[str, datetime, datetime]],
                sessions_ahead: int = HORIZON_SESSIONS) -> datetime | None:
    """entry_open + sessions_ahead * 390 minutes of *market* time.

    Walks session by session so nights, weekends and holidays are skipped, and a
    partial first session only consumes the minutes that actually remain in it.
    """
    remaining = sessions_ahead * SESSION_MINUTES
    t = entry_open
    idx = 0
    while idx < len(sess):
        _, so, sc = sess[idx]
        if sc <= t:
            idx += 1
            continue
        if t < so:
            t = so
        avail = int((sc - t).total_seconds() // 60)
        if avail >= remaining:
            return t + timedelta(minutes=remaining)
        remaining -= avail
        t = sc
        idx += 1
    return None            # not enough future sessions in the archive


@dataclass
class Observation:
    ticker: str
    signal_ts: str                 # bar whose completion triggers the signal
    entry_ts: str                  # next tradable bar
    entry_price: float
    session_date: str
    horizon_end: str
    mfe: float | None
    mae: float | None
    mfe_close: float | None
    mae_close: float | None
    labels: dict[str, int] = field(default_factory=dict)
    tau: dict[str, int | None] = field(default_factory=dict)
    tau_down: dict[str, int | None] = field(default_factory=dict)
    tau_ts: dict[str, str | None] = field(default_factory=dict)
    tau_down_ts: dict[str, str | None] = field(default_factory=dict)
    bars_to_mfe: int | None = None
    episode_id: str | None = None
    judged: bool = True
    reason: str | None = None


def build_observations(ticker: str, bars: list[Bar], segments: list[Segment] | None = None,
                       targets=TARGETS, down_barriers=DOWN_BARRIERS,
                       sessions_ahead: int = HORIZON_SESSIONS) -> list[Observation]:
    """One observation per regular-session bar; entry is the next bar's open."""
    bars = regular_bars(bars)
    if len(bars) < 4:
        return []
    sess = sessions(bars)
    # episode id from the (post-hoc) label segmentation, for statistical grouping only
    episode_of = {}
    if segments:
        for s in segments:
            for k in range(s.start_idx, s.end_idx + 1):
                episode_of[k] = s.event_id
    out = []
    for i in range(len(bars) - 1):
        entry_bar = bars[i + 1]
        entry_price = entry_bar.o
        if entry_price <= 0:
            continue
        entry_open = bar_open(entry_bar, bars[i])
        end = horizon_end(entry_open, sess, sessions_ahead)
        if end is None:
            out.append(Observation(ticker, bars[i].ts.isoformat(sep=" "),
                                   entry_bar.ts.isoformat(sep=" "), entry_price,
                                   bars[i].ts.strftime("%Y-%m-%d"), "", None, None, None,
                                   None, judged=False, reason="horizon_beyond_data"))
            continue
        hi = lo = hi_c = lo_c = None
        tau, tau_d, tau_ts, tau_d_ts = {}, {}, {}, {}
        bars_to_mfe, mfe_idx = None, None
        for j in range(i + 1, len(bars)):
            if bars[j].ts > end:
                break
            b = bars[j]
            if hi is None or b.h > hi:
                hi, mfe_idx = b.h, j
            lo = b.l if lo is None else min(lo, b.l)
            hi_c = b.c if hi_c is None else max(hi_c, b.c)
            lo_c = b.c if lo_c is None else min(lo_c, b.c)
            for q in targets:
                if q not in tau and b.h >= entry_price * (1 + q):
                    tau[q] = j - i
                    tau_ts[q] = b.ts.isoformat(sep=" ")
            for d in down_barriers:
                if d not in tau_d and b.l <= entry_price * (1 - d):
                    tau_d[d] = j - i
                    tau_d_ts[d] = b.ts.isoformat(sep=" ")
        if hi is None:
            out.append(Observation(ticker, bars[i].ts.isoformat(sep=" "),
                                   entry_bar.ts.isoformat(sep=" "), entry_price,
                                   bars[i].ts.strftime("%Y-%m-%d"), end.isoformat(sep=" "),
                                   None, None, None, None, judged=False,
                                   reason="no_bars_in_horizon"))
            continue
        mfe = hi / entry_price - 1
        mae = lo / entry_price - 1
        out.append(Observation(
            ticker=ticker, signal_ts=bars[i].ts.isoformat(sep=" "),
            entry_ts=entry_bar.ts.isoformat(sep=" "), entry_price=entry_price,
            session_date=bars[i].ts.strftime("%Y-%m-%d"),
            horizon_end=end.isoformat(sep=" "),
            mfe=mfe, mae=mae,
            mfe_close=(hi_c / entry_price - 1) if hi_c is not None else None,
            mae_close=(lo_c / entry_price - 1) if lo_c is not None else None,
            labels={f"{int(q*100)}%": int(mfe >= q) for q in targets},
            tau={f"{int(q*100)}%": tau.get(q) for q in targets},
            tau_down={f"{int(d*100)}%": tau_d.get(d) for d in down_barriers},
            tau_ts={f"{int(q*100)}%": tau_ts.get(q) for q in targets},
            tau_down_ts={f"{int(d*100)}%": tau_d_ts.get(d) for d in down_barriers},
            bars_to_mfe=(mfe_idx - i) if mfe_idx is not None else None,
            episode_id=episode_of.get(i + 1),
        ))
    return out


def causal_features(bars: list[Bar], when: datetime, delta: float = 2.45,
                    atr_window: int = 20, lookback: int = 20) -> dict:
    """Features knowable when the bar ending at *when* has just closed.

    ``when`` is a timestamp, not an index, so the caller cannot accidentally mix up
    positions between the full bar list and the regular-session subset.  Strictly
    causal: nothing at or after *when* is read, which is what makes the P0 test
    (appending future data never changes past features) meaningful.
    """
    reg = regular_bars(bars)
    idx = None
    for k, b in enumerate(reg):
        if b.ts == when:
            idx = k
            break
    if idx is None:
        return {}
    i = idx
    y = [math.log(b.c) for b in reg[:i + 1]]
    atr = trailing_atr_series(reg[:i + 1], atr_window)
    st = causal_state(y, atr, delta, i)
    c = reg[i].c
    feats = {
        "close": c,
        "ret_1": c / reg[i - 1].c - 1 if i >= 1 else None,
        "ret_4": c / reg[i - 4].c - 1 if i >= 4 else None,
        "ret_20": c / reg[i - 20].c - 1 if i >= 20 else None,
        "atr": atr[i] if i < len(atr) else None,
        "bars_since_pivot": st["bars_since_pivot"],
        "drawdown_from_extreme": st["drawdown_from_extreme"],
        "slope_since_pivot": st["slope_since_pivot"],
        "current_direction": st["current_direction"],
        "n_confirmed_pivots": len(st["pivots"]),
        "last_pivot_kind": st["pivots"][-1][1] if st["pivots"] else None,
    }
    if i >= lookback:
        rets = [y[k] - y[k - 1] for k in range(i - lookback + 1, i + 1)]
        m = sum(rets) / len(rets)
        feats["rv"] = math.sqrt(sum((r - m) ** 2 for r in rets) / len(rets))
    else:
        feats["rv"] = None
    for w in (4, 20):
        feats[f"slope_{w}"] = (y[i] - y[i - w]) / w if i >= w else None
    feats["accel"] = (feats["slope_4"] - feats["slope_20"]
                      if feats["slope_4"] is not None and feats["slope_20"] is not None else None)
    vols = [reg[k].v for k in range(max(0, i - 19), i + 1)]
    prior = vols[:-1]
    feats["vol_ratio"] = (reg[i].v / (sum(prior) / len(prior))
                          if len(prior) >= 4 and sum(prior) > 0 else None)
    sess_dates = [b.ts.strftime("%Y-%m-%d") for b in reg[:i + 1]]
    si = 0
    for k in range(i, -1, -1):
        if sess_dates[k] != sess_dates[i]:
            si = k + 1
            break
    lo_i = max(0, si - 20)
    win_h = [reg[k].h for k in range(lo_i, i + 1)]
    win_l = [reg[k].l for k in range(lo_i, i + 1)]
    feats["px_vs_hi20"] = reg[i].c / max(win_h) - 1 if win_h else None
    feats["px_vs_lo20"] = reg[i].c / min(win_l) - 1 if win_l else None
    feats["sess_ret"] = (reg[i].c / reg[si].c - 1) if reg[si].c > 0 else None
    feats["sess_bars"] = i - si + 1
    if st["pivots"]:
        li = st["pivots"][-1][0]
        window = y[li:i + 1]
        if st["current_direction"] == "up":
            feats["dist_from_running_max"] = max(window) - y[i]
        else:
            feats["dist_from_running_min"] = y[i] - min(window)
    return feats


class CausalWalker:
    """Incremental causal segmentation + features.

    ``push(bar)`` folds one more bar in and returns the features knowable at that bar.
    Equivalent to calling :func:`causal_features` for every prefix, but O(1) per bar
    instead of O(n), which matters for ~23k observation points.
    """

    def __init__(self, delta: float = 2.45, atr_window: int = 20, lookback: int = 20):
        self.delta, self.atr_window, self.lookback = delta, atr_window, lookback
        self.p: list[float] = []          # raw closes
        self.v: list[float] = []          # volumes
        self.h: list[float] = []          # highs
        self.l: list[float] = []          # lows
        self.session_of: list[str] = []   # ET date of each bar
        self.session_start: list[int] = []  # index of the first bar of that session
        self.y: list[float] = []          # log closes, for slopes / ATR
        self.trs: list[float | None] = []
        self.pivots: list[tuple[int, str]] = []
        self.phase = 0
        self.hi_idx = self.lo_idx = 0
        self.hi_val = self.lo_val = None
        self.direction = 0
        self.ext_idx = self.ext_val = 0

    def push(self, bar: Bar) -> dict:
        i = len(self.y)
        self.p.append(float(bar.c))
        self.v.append(float(bar.v))
        self.h.append(float(bar.h))
        self.l.append(float(bar.l))
        d = bar.ts.strftime("%Y-%m-%d")
        self.session_of.append(d)
        self.session_start.append(i if (i == 0 or self.session_of[i - 1] != d) else
                                  self.session_start[i - 1])
        c = math.log(bar.c) if bar.c > 0 else None
        if c is None:
            self.y.append(self.y[-1] if self.y else 0.0)
            self.trs.append(None)
            return self._features(i, atr=None)
        self.y.append(c)
        if i == 0:
            self.trs.append(None)
            self.hi_val = self.lo_val = c
            return self._features(i, atr=None)
        h, l, pc = math.log(bar.h), math.log(bar.l), self.y[i - 1]
        self.trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        vals = [t for t in self.trs[max(0, i - self.atr_window + 1):i + 1] if t is not None]
        atr = sum(vals) / len(vals) if vals else None
        if self.hi_val is None:
            self.hi_val = self.lo_val = c
        if c > self.hi_val:
            self.hi_idx, self.hi_val = i, c
        if c < self.lo_val:
            self.lo_idx, self.lo_val = i, c
        theta = self.delta * atr if atr else None
        if theta is not None:
            if self.phase == 0:
                if self.hi_val - c >= theta and self.hi_idx > 0:
                    self.pivots.append((self.hi_idx, "peak"))
                    self.phase, self.direction = 1, -1
                    self.ext_idx, self.ext_val = self.hi_idx, self.hi_val
                elif c - self.lo_val >= theta and self.lo_idx > 0:
                    self.pivots.append((self.lo_idx, "trough"))
                    self.phase, self.direction = 1, 1
                    self.ext_idx, self.ext_val = self.lo_idx, self.lo_val
            elif self.direction == 1:
                if c > self.ext_val:
                    self.ext_idx, self.ext_val = i, c
                elif self.ext_val - c >= theta:
                    self.pivots.append((self.ext_idx, "peak"))
                    self.direction = -1
                    self.ext_idx, self.ext_val = i, c
            else:
                if c < self.ext_val:
                    self.ext_idx, self.ext_val = i, c
                elif c - self.ext_val >= theta:
                    self.pivots.append((self.ext_idx, "trough"))
                    self.direction = 1
                    self.ext_idx, self.ext_val = i, c
        return self._features(i, atr)

    def _features(self, i: int, atr) -> dict:
        y, p, v = self.y, self.p, self.v
        f = {
            "close": p[i], "atr": atr,
            "ret_1": p[i] / p[i - 1] - 1 if i >= 1 else None,
            "ret_4": p[i] / p[i - 4] - 1 if i >= 4 else None,
            "ret_20": p[i] / p[i - 20] - 1 if i >= 20 else None,
            "bars_since_pivot": (i - self.pivots[-1][0]) if self.pivots else None,
            "current_direction": ("up" if self.direction == 1 else "down")
                                if self.pivots else None,
            "n_confirmed_pivots": len(self.pivots),
            "last_pivot_kind": self.pivots[-1][1] if self.pivots else None,
        }
        if self.pivots:
            li = self.pivots[-1][0]
            window = y[li:i + 1]
            if self.direction == 1:
                dd = max(window) - y[i]
                f["drawdown_from_extreme"] = dd
                f["dist_from_running_max"] = dd
            else:
                dd = y[i] - min(window)
                f["drawdown_from_extreme"] = dd
                f["dist_from_running_min"] = dd
            f["slope_since_pivot"] = (y[i] - y[li]) / max(1, i - li)
        else:
            f["drawdown_from_extreme"] = None
            f["slope_since_pivot"] = None
        if i >= self.lookback:
            rets = [y[k] - y[k - 1] for k in range(i - self.lookback + 1, i + 1)]
            m = sum(rets) / len(rets)
            f["rv"] = math.sqrt(sum((r - m) ** 2 for r in rets) / len(rets))
        else:
            f["rv"] = None
        for w in (4, 20):
            f[f"slope_{w}"] = (y[i] - y[i - w]) / w if i >= w else None
        f["accel"] = (f["slope_4"] - f["slope_20"]
                      if f["slope_4"] is not None and f["slope_20"] is not None else None)
        prior = v[max(0, i - 19):i]
        f["vol_ratio"] = (v[i] / (sum(prior) / len(prior))
                          if len(prior) >= 4 and sum(prior) > 0 else None)
        # rule inputs: position vs the last 20 sessions, and session-to-date return
        lo_i = max(0, self.session_start[i] - 20)
        win_h = self.h[lo_i:i + 1]
        win_l = self.l[lo_i:i + 1]
        f["px_vs_hi20"] = p[i] / max(win_h) - 1 if win_h else None
        f["px_vs_lo20"] = p[i] / min(win_l) - 1 if win_l else None
        si = self.session_start[i]
        f["sess_ret"] = p[i] / p[si] - 1 if p[si] > 0 else None
        f["sess_bars"] = i - si + 1
        return f


def market_features(bars_by_ticker: dict[str, list[Bar]]) -> dict[str, dict]:
    """Per-session market context from QQQ (regular session only)."""
    q = regular_bars(bars_by_ticker.get("QQQ", []))
    out = {}
    closes = [b.c for b in q]
    for i, b in enumerate(q):
        d = b.ts.strftime("%Y-%m-%d")
        out.setdefault(d, {})
        out[d]["qqq_ret_20"] = closes[i] / closes[i - 20] - 1 if i >= 20 else None
        out[d]["qqq_ret_1"] = closes[i] / closes[i - 1] - 1 if i >= 1 else None
    return out
