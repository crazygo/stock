"""Drawdown-tolerant up/down segmentation of a price path on log prices.

Three parameters with three strictly separated jobs:

* ``delta``   - the ONLY structural parameter.  A segment ends when price retraces
  ``delta * ATR`` from its running extreme.  This answers "when did this move end".
* ``min_amp`` - a pure qualification flag applied after segmentation.  It never
  moves a pivot.  Answers "was this move big enough to care about".
* ``min_bars``- a pure duration flag, also applied after segmentation.

The target return (e.g. 8%) is deliberately NOT a parameter of this module: it
belongs to the prediction label (triple-barrier), not to the price structure.

``atr_mode="global"`` uses the whole-sample median true range and is therefore only
valid for *labels*.  ``atr_mode="trailing"`` is causal: every pivot it emits was
confirmable at the moment it was emitted, so the confirmed-pivot sequence grows
monotonically as data arrives and is safe to build features from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Bar:
    ts: datetime
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0          # volume; 0.0 when the source has none


@dataclass
class Segment:
    kind: str                 # "up" | "down"
    start_idx: int
    end_idx: int
    start_ts: str
    end_ts: str
    amplitude: float          # log return over the segment
    bars: int
    calendar_days: int
    v_mean: float
    v1: float
    v2: float
    v3: float
    delta_v: float
    efficiency: float
    mdd: float
    vol: float
    event_id: str = ""        # e.g. "UP#7"; stable identifier of the parent move
    qualified: bool = True    # amplitude >= min_amp * ATR
    censored: bool = False    # last segment of the sample is never complete
    short: bool = False       # shorter than min_bars: excluded from the analysis set
    meta: dict = field(default_factory=dict)


def log_prices(bars: list[Bar]) -> list[float]:
    return [math.log(b.c) for b in bars]


def true_ranges_log(bars: list[Bar]) -> list[float | None]:
    """True range in log space; None for the first bar."""
    out: list[float | None] = [None]
    for i in range(1, len(bars)):
        h, l, pc = math.log(bars[i].h), math.log(bars[i].l), math.log(bars[i - 1].c)
        out.append(max(h - l, abs(h - pc), abs(l - pc)))
    return out


def atr_scale(bars: list[Bar], mode: str = "global", window: int = 20) -> float:
    """A single volatility scale for the whole series (median true range)."""
    trs = [t for t in true_ranges_log(bars) if t is not None]
    if not trs:
        return 0.0
    if mode == "global":
        s = sorted(trs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    if mode == "trailing":
        return sum(trs[-window:]) / len(trs[-window:])
    raise ValueError(mode)


def trailing_atr_series(bars: list[Bar], window: int = 20) -> list[float]:
    """Causal ATR at every bar: atr[i] uses only bars[0..i]."""
    trs = true_ranges_log(bars)
    out: list[float] = []
    for i in range(len(bars)):
        vals = [t for t in trs[max(0, i - window + 1):i + 1] if t is not None]
        out.append(sum(vals) / len(vals) if vals else 0.0)
    return out


def zigzag_pivots(y: list[float], theta: float) -> list[tuple[int, str]]:
    """Alternating trough/peak pivots with a drawdown tolerance of *theta* (log units).

    A pivot is appended only once price has retraced *theta* from the extreme, i.e.
    every emitted pivot was confirmable at the moment it was emitted.  With a causal
    ``theta`` the sequence therefore never rewrites history.

    Phase 1 establishes the first pivot: the first time the path either falls *theta*
    below its running maximum (=> that maximum is a peak) or rises *theta* above its
    running minimum (=> that minimum is a trough).  A pivot at index 0 is rejected
    because there is no data on its left, so a purely monotone window yields nothing.
    """
    n = len(y)
    if n < 3 or theta <= 0:
        return []
    pivots: list[tuple[int, str]] = []
    hi_idx, hi_val = 0, y[0]
    lo_idx, lo_val = 0, y[0]
    first = None
    for i in range(1, n):
        if y[i] > hi_val:
            hi_idx, hi_val = i, y[i]
        if y[i] < lo_val:
            lo_idx, lo_val = i, y[i]
        if hi_val - y[i] >= theta and hi_idx > 0:
            first = (hi_idx, "peak")
            break
        if y[i] - lo_val >= theta and lo_idx > 0:
            first = (lo_idx, "trough")
            break
    if first is None:
        return []
    pivots.append(first)
    direction = 1 if first[1] == "trough" else -1
    ext_idx, ext_val = first[0], y[first[0]]
    for i in range(first[0] + 1, n):
        if direction == 1:                      # riding a rise, hunting the next peak
            if y[i] > ext_val:
                ext_idx, ext_val = i, y[i]
            elif ext_val - y[i] >= theta:
                pivots.append((ext_idx, "peak"))
                direction = -1
                ext_idx, ext_val = i, y[i]
        else:                                   # riding a fall, hunting the next trough
            if y[i] < ext_val:
                ext_idx, ext_val = i, y[i]
            elif y[i] - ext_val >= theta:
                pivots.append((ext_idx, "trough"))
                direction = 1
                ext_idx, ext_val = i, y[i]
    return pivots


def zigzag_pivots_causal(y: list[float], atr: list[float], delta: float,
                          upto: int | None = None) -> list[tuple[int, str]]:
    """Same as :func:`zigzag_pivots` but with a bar-by-bar causal threshold.

    ``atr[i]`` must depend only on ``y[0..i]``.  ``upto`` limits the walk to that bar,
    which makes it possible to ask "what was confirmable at bar *upto*".
    """
    n = len(y) if upto is None else min(upto + 1, len(y))
    if n < 3:
        return []
    # atr[0] has no true range yet and is legitimately 0; it is never used for a
    # decision because the walk starts at bar 1.
    theta = [delta * a if a > 0 else None for a in atr[:n]]
    if all(t is None for t in theta[1:]):
        return []
    pivots: list[tuple[int, str]] = []
    hi_idx, hi_val = 0, y[0]
    lo_idx, lo_val = 0, y[0]
    first = None
    for i in range(1, n):
        if y[i] > hi_val:
            hi_idx, hi_val = i, y[i]
        if y[i] < lo_val:
            lo_idx, lo_val = i, y[i]
        th = theta[i]
        if th is None:
            continue
        if hi_val - y[i] >= th and hi_idx > 0:
            first = (hi_idx, "peak")
            break
        if y[i] - lo_val >= th and lo_idx > 0:
            first = (lo_idx, "trough")
            break
    if first is None:
        return []
    pivots.append(first)
    direction = 1 if first[1] == "trough" else -1
    ext_idx, ext_val = first[0], y[first[0]]
    for i in range(first[0] + 1, n):
        th = theta[i]
        if th is None:
            continue
        if direction == 1:
            if y[i] > ext_val:
                ext_idx, ext_val = i, y[i]
            elif ext_val - y[i] >= th:
                pivots.append((ext_idx, "peak"))
                direction = -1
                ext_idx, ext_val = i, y[i]
        else:
            if y[i] < ext_val:
                ext_idx, ext_val = i, y[i]
            elif y[i] - ext_val >= th:
                pivots.append((ext_idx, "trough"))
                direction = 1
                ext_idx, ext_val = i, y[i]
    return pivots


def causal_state(y: list[float], atr: list[float], delta: float, i: int) -> dict:
    """What was knowable about the path structure at bar *i* (no future information)."""
    pivots = zigzag_pivots_causal(y, atr, delta, upto=i)
    if not pivots:
        return {"pivots": [], "bars_since_pivot": None, "drawdown_from_extreme": None,
                "slope_since_pivot": None, "current_direction": None}
    last_idx, last_kind = pivots[-1]
    window = y[last_idx:i + 1]
    direction = 1 if last_kind == "trough" else -1
    extreme = max(window) if direction == 1 else min(window)
    return {
        "pivots": pivots,
        "bars_since_pivot": i - last_idx,
        "drawdown_from_extreme": (extreme - y[i]) if direction == 1 else (y[i] - extreme),
        "slope_since_pivot": (y[i] - y[last_idx]) / max(1, i - last_idx),
        "current_direction": "up" if direction == 1 else "down",
        "extreme_idx": window.index(extreme) + last_idx,
    }


def _slope(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2
    my = sum(vals) / n
    num = sum((i - mx) * (v - my) for i, v in enumerate(vals))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den else 0.0


def _features(y: list[float], i: int, j: int) -> dict:
    seg = y[i:j + 1]
    n = len(seg)
    amp = seg[-1] - seg[0]
    third = max(1, n // 3)
    v1 = _slope(seg[:third])
    v2 = _slope(seg[third:2 * third]) if n >= 3 * third else _slope(seg[third:])
    v3 = _slope(seg[2 * third:]) if n >= 2 * third + 1 else _slope(seg[-third:])
    path = sum(abs(seg[k + 1] - seg[k]) for k in range(n - 1))
    run_max, mdd = seg[0], 0.0
    for v in seg:
        run_max = max(run_max, v)
        mdd = max(mdd, run_max - v)
    rets = [seg[k + 1] - seg[k] for k in range(n - 1)]
    mean = sum(rets) / len(rets) if rets else 0.0
    var = sum((r - mean) ** 2 for r in rets) / len(rets) if rets else 0.0
    return {"amplitude": amp, "bars": n, "v_mean": amp / max(1, n - 1),
            "v1": v1, "v2": v2, "v3": v3, "delta_v": v3 - v1,
            "efficiency": abs(amp) / path if path > 0 else 0.0,
            "mdd": mdd, "vol": math.sqrt(var)}


def build_segments(bars: list[Bar], delta: float = 2.45, min_amp: float = 1.0,
                   min_bars: int = 6, atr_mode: str = "global",
                   atr_window: int = 20) -> list[Segment]:
    """Segment *bars* into alternating up/down intervals.

    ``theta = delta * ATR`` only.  ``min_amp`` and ``min_bars`` never move a pivot;
    they only flag segments, because removing a pivot would break the alternation and
    silently merge neighbours.  To merge, raise ``delta`` instead.
    """
    if len(bars) < 4:
        return []
    y = log_prices(bars)
    if atr_mode == "global":
        scale = atr_scale(bars, "global")
        pivots = zigzag_pivots(y, delta * scale)
        scale_used = scale
    elif atr_mode == "trailing":
        atr = trailing_atr_series(bars, atr_window)
        pivots = zigzag_pivots_causal(y, atr, delta)
        scale_used = atr_scale(bars, "trailing", atr_window)
    else:
        raise ValueError(atr_mode)
    if scale_used <= 0 or len(pivots) < 2:
        return []

    segments: list[Segment] = []
    counters = {"up": 0, "down": 0}
    for k in range(len(pivots) - 1):
        i, kind_i = pivots[k]
        j, kind_j = pivots[k + 1]
        if kind_i == "trough" and kind_j == "peak":
            kind = "up"
        elif kind_i == "peak" and kind_j == "trough":
            kind = "down"
        else:
            continue
        f = _features(y, i, j)
        counters[kind] += 1
        segments.append(Segment(
            kind=kind, start_idx=i, end_idx=j,
            start_ts=bars[i].ts.isoformat(sep=" "),
            end_ts=bars[j].ts.isoformat(sep=" "),
            calendar_days=(bars[j].ts.date() - bars[i].ts.date()).days,
            event_id=f"{kind.upper()}#{counters[kind]}",
            qualified=abs(f["amplitude"]) >= min_amp * scale_used,
            censored=(k == len(pivots) - 2),
            short=(f["bars"] < min_bars),
            meta={"atr": scale_used, "theta": delta * scale_used,
                  "start_price": bars[i].c, "end_price": bars[j].c}, **f))
    return segments


def analysis_segments(segments: list[Segment]) -> list[Segment]:
    """Segments usable as labels: long enough, big enough and not censored."""
    return [s for s in segments if not s.short and not s.censored and s.qualified]


def crossing_index(bars: list[Bar], start_idx: int, end_idx: int, target: float) -> int | None:
    """First bar at or after *start_idx* whose close is *target* above the start close."""
    if not bars or start_idx >= len(bars):
        return None
    base = bars[start_idx].c
    if base <= 0:
        return None
    threshold = base * (1.0 + target)
    for k in range(start_idx + 1, min(end_idx, len(bars) - 1) + 1):
        if bars[k].c >= threshold:
            return k
    return None


def qualifies(bars: list[Bar], seg: Segment, target: float = 0.08,
              mode: str = "amplitude", max_one_day: float = 0.12,
              retained_frac: float = 0.5, follow: int = 2) -> bool:
    """Does this segment count as a "target achieved" up-move?

    mode="amplitude" - the plain trough->peak amplitude reaches *target*.
    mode="window"    - additionally ports the previous study's window rule: no single
                       close-to-close gain above *max_one_day* up to the crossing, and
                       both of the next *follow* bars keep at least *retained_frac* of
                       the target gain versus the start close.  The calendar-day and
                       minimum-session constraints of that study are deliberately not
                       ported, because they were tuned on daily bars.
    """
    if seg.kind != "up":
        return False
    if math.exp(seg.amplitude) - 1 < target:
        return False
    if mode == "amplitude":
        return True
    if mode != "window":
        raise ValueError(mode)
    k = crossing_index(bars, seg.start_idx, seg.end_idx, target)
    if k is None:
        return False
    for j in range(seg.start_idx + 1, k + 1):
        if bars[j].c / bars[j - 1].c - 1 > max_one_day:
            return False
    base = bars[seg.start_idx].c
    for j in range(k + 1, k + 1 + follow):
        if j >= len(bars):
            return False
        if bars[j].c / base - 1 < target * retained_frac:
            return False
    return True
