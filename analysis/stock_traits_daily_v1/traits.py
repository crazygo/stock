"""five_traits_v2 price descriptors, implemented from v5_12_refs/five-traits-v2.md.

Not a trained model, no news/valuation, not a probability of future returns.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MODEL_VERSION = "five_traits_v2"
IMPL_VERSION = "stock_traits_daily_v1/traits.py@2"  # @2: corporate-action return breaks
PARAMS = {
    "model_version": MODEL_VERSION,
    "returns": "r_t = log(P_t / P_(t-1)) on consecutive official sessions; never across a true missing session",
    "G": {"window_returns": 120, "annualize": 252},
    "V": {"window_returns": 60, "annualize": "sqrt(252)", "ddof": 1},
    "M": {"window_returns": 60, "pairs": 59},
    "J": {"window_returns": 60, "top_k": 6},
    "S": {"changes": 20, "points": 21, "returns_needed": 140, "closes_needed": 141, "k": 10,
          "projection": "F=(tanh(G/.5), tanh((V-.35)/.25), M, tanh((J-.55)/.20))"},
    "bootstrap": {"replicates": 256, "block": 5, "max_returns": 140, "percentiles": [10, 90],
                  "seed_rule": "int.from_bytes(sha256('five_traits_v2|<security_id>|<asof YYYY-MM-DD>|256')[:8], 'little') -> numpy.default_rng",
                  "start_rule": "uniform integer in [0, n-5] inclusive; ceil(n/5) blocks; truncate to n"},
    "display": {"map_x": "asinh(V/.20)", "map_y": "asinh(G/.50)",
                "G": "50*(1+tanh(G/.5))", "V": "50*(1+tanh((V-.35)/.25))", "M": "50*(1+M)", "J": "100*J", "S": "100*S"},
    "numpy": np.__version__,
}
TRAITS = ["G", "V", "M", "J", "S"]
UNITS = {"G": "年化 log 增长 (1/yr)", "V": "年化波动 (1/sqrt(yr))", "M": "相关系数 (无单位)", "J": "能量占比 (0–1)", "S": "稳定度 (0–1)"}
CATEGORIES = {
    "G": ["下行", "中性", "上行"],
    "V": ["较低", "较高"],
    "M": ["反向", "接近零", "延续"],
    "J": ["较分散", "较集中"],
    "S": ["较易变", "较稳定"],
}
NEEDED_RETURNS = {"G": 120, "V": 60, "M": 60, "J": 60, "S": 140}


def category(trait: str, v: float | None) -> str | None:
    if v is None or not math.isfinite(v):
        return None
    if trait == "G":
        return "下行" if v < -0.10 else ("上行" if v > 0.10 else "中性")
    if trait == "V":
        return "较低" if v <= 0.35 else "较高"
    if trait == "M":
        return "反向" if v < -0.15 else ("延续" if v > 0.15 else "接近零")
    if trait == "J":
        return "较分散" if v <= 0.60 else "较集中"
    if trait == "S":
        return "较稳定" if v >= 0.75 else "较易变"
    raise KeyError(trait)


def display(trait: str, v: float | None) -> float | None:
    if v is None:
        return None
    return {"G": lambda x: 50 * (1 + math.tanh(x / .5)), "V": lambda x: 50 * (1 + math.tanh((x - .35) / .25)),
            "M": lambda x: 50 * (1 + x), "J": lambda x: 100 * x, "S": lambda x: 100 * x}[trait](v)


def map_xy(G: float | None, V: float | None) -> tuple[float, float] | None:
    if G is None or V is None:
        return None
    return math.asinh(V / .20), math.asinh(G / .50)


# ------------------------------------------------------------------ core (vectorised over windows)
def _g(w120: np.ndarray) -> np.ndarray:
    return w120.mean(axis=-1) * 252


def _v(w60: np.ndarray) -> np.ndarray:
    sd = w60.std(axis=-1, ddof=1)
    return np.where(sd > 0, sd * math.sqrt(252), np.nan)


def _m(w60: np.ndarray) -> np.ndarray:
    a, b = w60[..., :-1], w60[..., 1:]
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(axis=-1) / np.where(den > 0, den, 1), np.nan)


def _j(w60: np.ndarray) -> np.ndarray:
    x = w60 - w60.mean(axis=-1, keepdims=True)
    e = x * x
    tot = e.sum(axis=-1)
    top = -np.sort(-e, axis=-1)[..., :6].sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, top / np.where(tot > 0, tot, 1), np.nan)


def _f(G, V, M, J) -> np.ndarray:
    return np.stack([np.tanh(G / .5), np.tanh((V - .35) / .25), M, np.tanh((J - .55) / .20)], axis=-1)


def compute(r: np.ndarray) -> dict:
    """Five traits on a contiguous return vector (last element = as-of)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    out = {}
    out["G"] = (float(_g(r[-120:])), None) if n >= 120 else (None, "short_history")
    if n >= 60:
        w = r[-60:]
        for k, fn in (("V", _v), ("M", _m), ("J", _j)):
            val = float(fn(w))
            out[k] = (val, None) if math.isfinite(val) else (None, "zero_variance_or_zero_denominator")
    else:
        for k in ("V", "M", "J"):
            out[k] = (None, "short_history")
    if n >= 140:
        rr = r[-140:]
        w120 = sliding_window_view(rr, 120)[-21:]
        w60 = sliding_window_view(rr, 60)[-21:]
        F = _f(_g(w120), _v(w60), _m(w60), _j(w60))
        if not np.isfinite(F).all():
            out["S"] = (None, "daily_component_unknown")
        else:
            D = np.abs(np.diff(F, axis=0)).mean() / 2
            out["S"] = (float(math.exp(-10 * D)), None)
    else:
        out["S"] = (None, "short_history")
    return out


def s_reference(closes: np.ndarray) -> float:
    """Slow, literal S used by tests: per-day traits from daily closes, then the formula."""
    r = np.diff(np.log(np.asarray(closes, dtype=float)))
    F = []
    for end in range(len(r) - 20, len(r) + 1):
        w = r[:end]
        G = w[-120:].mean() * 252
        x = w[-60:]
        V = x.std(ddof=1) * math.sqrt(252)
        M = np.corrcoef(x[:-1], x[1:])[0, 1]
        e = (x - x.mean()) ** 2
        J = np.sort(e)[-6:].sum() / e.sum()
        F.append([math.tanh(G / .5), math.tanh((V - .35) / .25), M, math.tanh((J - .55) / .2)])
    F = np.array(F)
    return math.exp(-10 * np.abs(np.diff(F, axis=0)).mean() / 2)


def seed_for(security_id: str, asof: str, reps: int = 256) -> int:
    return int.from_bytes(hashlib.sha256(f"{MODEL_VERSION}|{security_id}|{asof}|{reps}".encode()).digest()[:8], "little")


def bootstrap(r: np.ndarray, security_id: str, asof: str, reps: int = 256, block: int = 5) -> dict:
    r = np.asarray(r, dtype=float)[-140:]
    n = len(r)
    res = {t: [] for t in TRAITS}
    if n < block:
        return {t: {"valid_replicates": 0} for t in TRAITS}
    rng = np.random.default_rng(seed_for(security_id, asof, reps))
    nb = math.ceil(n / block)
    for _ in range(reps):
        starts = rng.integers(0, n - block + 1, size=nb)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        vals = compute(sample)
        for t in TRAITS:
            v = vals[t][0]
            if v is not None and math.isfinite(v):
                res[t].append(v)
    out = {}
    for t in TRAITS:
        vs = np.array(res[t])
        k = len(CATEGORIES[t])
        if len(vs) == 0:
            out[t] = {"valid_replicates": 0}
            continue
        counts = {c: 0 for c in CATEGORIES[t]}
        for v in vs:
            counts[category(t, float(v))] += 1
        out[t] = {"valid_replicates": int(len(vs)), "p10": float(np.percentile(vs, 10)), "p90": float(np.percentile(vs, 90)),
                  "category_frequency": {c: (counts[c] + 0.5) / (len(vs) + 0.5 * k) for c in CATEGORIES[t]},
                  "note": "重采样条件倾向，未校准，不是未来概率"}
    return out


def evaluate(series: list[tuple[str, float]], sessions: list[str], asof: str, security_id: str,
             reps: int = 256, breaks: list[str] | None = None) -> dict:
    """series: [(session_date, close)] ; sessions: official calendar sessions (sorted).

    Uses only the contiguous run of sessions ending at `asof`; never bridges a missing session.
    """
    closes = {d: c for d, c in series if d <= asof}
    if asof not in closes:
        return {"asof": asof, "error": "no_close_at_asof", "traits": {t: {"value": None, "status": "unknown",
                                                                           "reason": "no_close_at_asof"} for t in TRAITS}}
    cal = [d for d in sessions if d <= asof]
    brk = set(breaks or [])
    run = []
    stop = None
    for d in reversed(cal):
        if d in closes:
            run.append(d)
            if d in brk:  # return into d is not usable (suspected unadjusted corporate action)
                stop = f"suspected_corporate_action_break at {d}"
                break
        else:
            break
    run.reverse()
    earlier_exists = any(d < run[0] for d in closes)
    gap_session = None
    if earlier_exists and stop is None:
        idx = cal.index(run[0])
        gap_session = cal[idx - 1] if idx > 0 else None
    P = np.array([closes[d] for d in run], dtype=float)
    r = np.diff(np.log(P))
    vals = compute(r)
    boot = bootstrap(r, security_id, asof, reps) if reps else {t: {"valid_replicates": 0} for t in TRAITS}
    traits = {}
    for t in TRAITS:
        v, why = vals[t]
        if why == "short_history" and stop is not None:
            why = stop
        elif why == "short_history" and earlier_exists:
            why = f"missing_session_in_window (gap at {gap_session})"
        traits[t] = {"value": v, "status": "ok" if v is not None else "unknown", "reason": why, "unit": UNITS[t],
                     "category": category(t, v), "display_0_100": display(t, v),
                     "returns_available": int(len(r)), "returns_needed": NEEDED_RETURNS[t],
                     "uncertainty": boot[t] if v is not None else {"valid_replicates": boot[t].get("valid_replicates", 0)}}
    xy = map_xy(traits["G"]["value"], traits["V"]["value"])
    return {"asof": asof, "contiguous_closes": int(len(P)), "window_first_session": run[0],
            "traits": traits, "map": None if xy is None else {"x": xy[0], "y": xy[1]},
            "bootstrap_seed": seed_for(security_id, asof, reps)}
