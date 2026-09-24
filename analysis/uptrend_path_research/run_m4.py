"""M4: first round of out-of-sample rule validation (spec_v1.md revision A).

No stop loss.  The label stays "did the high touch the target within 3 trading days";
a rule is judged by the lift of its signal hit rate over the matched control inside the
same out-of-sample window, with a paired calendar-block bootstrap and Holm correction.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "analysis" / "growth_trigger"))

from gt_common import load_hourly, pinned_tickers
from observations import TARGETS, CausalWalker, regular_bars
from segments import Bar, build_segments

SKIP = {"SNOW", "SPCX", "TSM"}
KEYS = [f"{int(q*100)}%" for q in TARGETS]
PRIMARY = "5%"
HOLDOUT_FRACTION = 0.30
RV_BUCKETS = 3
ITERS = 2000
SEED = 20260924


# ---------------- frozen rules (spec_v1.md revision A4) ----------------
def B1(f):
    return (f.get("rel_20d") is not None and f["rel_20d"] > 0
            and f.get("px_vs_hi20") is not None and f["px_vs_hi20"] >= -0.01
            and f.get("vol_ratio") is not None and f["vol_ratio"] >= 1.2)


def B2(f):
    return (f.get("vol_ratio") is not None and f["vol_ratio"] >= 2.0
            and f.get("sess_ret") is not None and f["sess_ret"] > 0
            and f.get("px_vs_hi20") is not None and f["px_vs_hi20"] >= -0.03)


def B3(f):
    return (f.get("ret_20") is not None and f["ret_20"] > 0.10
            and f.get("qqq_20d") is not None and f["qqq_20d"] > 0
            and f.get("rel_20d") is not None and f["rel_20d"] > 0)


def B4(f):
    return (f.get("px_vs_lo20") is not None and f["px_vs_lo20"] <= 0.02
            and f.get("sess_ret") is not None and f["sess_ret"] > 0
            and f.get("vol_ratio") is not None and f["vol_ratio"] >= 1.5)


def _struct(f):
    """Structural state derived from the causal pivots."""
    return (f.get("current_direction"), f.get("n_confirmed_pivots"),
            f.get("bars_since_pivot"), f.get("drawdown_from_extreme"),
            f.get("slope_since_pivot"), f.get("atr"))


def S1(f, hist):
    """Confirmed down -> trough -> rebound -> pullback that has not broken the prior low."""
    piv = hist["pivots"]
    if len(piv) < 3 or f.get("current_direction") != "down":
        return False
    kinds = [k for _, k in piv[-3:]]
    if kinds != ["peak", "trough", "peak"]:
        return False
    trough_price = hist["prices"][piv[-2][0]]
    return f["close"] > trough_price


def S2(f, hist):
    """Inside a confirmed up segment and the recent slope beats the segment mean."""
    if f.get("current_direction") != "up" or not hist["pivots"]:
        return False
    li = hist["pivots"][-1][0]
    mean_slope = f.get("slope_since_pivot")
    return (mean_slope is not None and f.get("slope_4") is not None
            and f["slope_4"] > mean_slope)


def S3(f, hist):
    """Inside a confirmed up segment and the drawdown from the extreme is still within theta."""
    if f.get("current_direction") != "up":
        return False
    dd, atr = f.get("drawdown_from_extreme"), f.get("atr")
    return dd is not None and atr is not None and dd <= 2.45 * atr


RULES = {"B1": (B1, False), "B2": (B2, False), "B3": (B3, False), "B4": (B4, False),
         "S1": (S1, True), "S2": (S2, True), "S3": (S3, True)}


def build_features():
    """Causal feature rows for every regular-session bar of every stock."""
    q = regular_bars([Bar(load_hourly("QQQ").ts[i], 0, 0, 0, load_hourly("QQQ").c[i])
                      for i in range(len(load_hourly("QQQ").ts))])
    qcloses = [b.c for b in q]
    qqq_ret20 = {}
    for i, b in enumerate(q):
        d = b.ts.strftime("%Y-%m-%d")
        qqq_ret20[d] = qcloses[i] / qcloses[i - 20] - 1 if i >= 20 else None
    feats, hist = {}, {}
    for t in pinned_tickers():
        if t in SKIP:
            continue
        s = load_hourly(t)
        bars = [Bar(s.ts[i], s.o[i], s.h[i], s.l[i], s.l[i] and s.c[i], s.v[i])
                for i in range(len(s.ts))]
        bars = [Bar(s.ts[i], s.o[i], s.h[i], s.l[i], s.c[i], s.v[i])
                for i in range(len(s.ts))]
        reg = regular_bars(bars)
        w = CausalWalker()
        for b in reg:
            f = w.push(b)
            f["ticker"] = t
            f["session_date"] = b.ts.strftime("%Y-%m-%d")
            q20 = qqq_ret20.get(f["session_date"])
            f["qqq_20d"] = q20
            f["rel_20d"] = (f["ret_20"] - q20
                            if f["ret_20"] is not None and q20 is not None else None)
            feats[(t, b.ts.isoformat(sep=" "))] = f
            hist[(t, b.ts.isoformat(sep=" "))] = {"pivots": list(w.pivots),
                                                  "prices": list(w.p)}
    return feats, hist


def load_observations():
    out = []
    for line in (HERE / "observations.jsonl").read_text().splitlines():
        out.append(json.loads(line))
    return [o for o in out if o["judged"]]


def split_sessions(obs):
    dates = sorted({o["session_date"] for o in obs})
    cut = int(len(dates) * (1 - HOLDOUT_FRACTION))
    return set(dates[:cut]), set(dates[cut:])


def assign(obs, dev_dates, hold_dates):
    """Keep an observation only if its signal date and its label end date are on the same side."""
    dev, hold = [], []
    for o in obs:
        sd = o["session_date"]
        ed = o["horizon_end"][:10]
        if sd in dev_dates and ed in dev_dates:
            dev.append(o)
        elif sd in hold_dates and ed in hold_dates:
            hold.append(o)
        # else: the label observation window crosses the split -> purged
    return dev, hold


def rv_bucket(f, edges):
    rv = f.get("rv")
    if rv is None:
        return None
    for i, e in enumerate(edges):
        if rv <= e:
            return i
    return len(edges)


def paired_block_bootstrap(sig, ctrl, key, iters=ITERS, seed=SEED):
    """Bootstrap the difference in hit rates by resampling whole calendar blocks."""
    rng = random.Random(seed)
    by_date = defaultdict(lambda: {"s": [], "c": []})
    for o in sig:
        by_date[o["session_date"]]["s"].append(o)
    for o in ctrl:
        by_date[o["session_date"]]["c"].append(o)
    dates = sorted(by_date)
    if not dates:
        return None
    diffs, p_le0 = [], 0
    for _ in range(iters):
        pick = [dates[rng.randrange(len(dates))] for _ in range(len(dates))]
        s = [o for d in pick for o in by_date[d]["s"]]
        c = [o for d in pick for o in by_date[d]["c"]]
        if not s or not c:
            continue
        d = (sum(o["labels"][key] for o in s) / len(s)
             - sum(o["labels"][key] for o in c) / len(c))
        diffs.append(d)
        if d <= 0:
            p_le0 += 1
    if not diffs:
        return None
    diffs.sort()
    return {"delta": round(diffs[iters // 2], 6),
            "ci95": [round(diffs[int(0.025 * iters)], 6), round(diffs[int(0.975 * iters)], 6)],
            "p_one_sided": round(p_le0 / len(diffs), 5)}


def holm(pvals):
    """Holm-Bonferroni over a dict name -> p-value.  Returns name -> adjusted p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    n = len(items)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (n - i) * p))
        out[name] = round(running, 5)
    return out


def main() -> None:
    feats, hist = build_features()
    obs = load_observations()
    for o in obs:
        k = (o["ticker"], o["signal_ts"])
        o["f"] = feats.get(k, {})
    obs = [o for o in obs if o["f"].get("close") is not None]
    dev_dates, hold_dates = split_sessions(obs)
    dev, hold = assign(obs, dev_dates, hold_dates)
    print(f"observations with features: {len(obs)}")
    print(f"dev sessions {len(dev_dates)} -> {len(dev)} obs | "
          f"holdout sessions {len(hold_dates)} -> {len(hold)} obs")
    print(f"purged (label window crosses the split): {len(obs) - len(dev) - len(hold)}")

    # volatility terciles from the DEVELOPMENT data only
    rvs = sorted(o["f"]["rv"] for o in dev if o["f"].get("rv") is not None)
    edges = [rvs[int(len(rvs) / 3)], rvs[int(2 * len(rvs) / 3)]]

    def strat(o):
        return (o["ticker"], rv_bucket(o["f"], edges))

    results, pvals = {}, {}
    for name, (fn, needs_hist) in RULES.items():
        for scope, pool in (("dev", dev), ("holdout", hold)):
            sig = []
            for o in pool:
                f = o["f"]
                try:
                    hit = fn(f, hist.get((o["ticker"], o["signal_ts"]), {})) if needs_hist else fn(f)
                except Exception:
                    hit = False
                if hit:
                    sig.append(o)
            if not sig:
                results.setdefault(name, {})[scope] = {"signals": 0}
                continue
            for key in KEYS:
                hr = sum(o["labels"][key] for o in sig) / len(sig)
                base = sum(o["labels"][key] for o in pool) / len(pool)
                close_hr = sum(1 for o in sig if o["mfe_close"] is not None
                               and o["mfe_close"] >= TARGETS[KEYS.index(key)]) / len(sig)
                # matched control: within each (ticker, rv bucket) stratum
                strata = defaultdict(lambda: [0, 0])
                for o in pool:
                    st = strat(o)
                    if st[1] is None:
                        continue
                    strata[st][0] += 1
                    strata[st][1] += o["labels"][key]
                matched_num = matched_den = 0.0
                for o in sig:
                    st = strat(o)
                    if st[1] is None or st not in strata or strata[st][0] == 0:
                        continue
                    matched_num += strata[st][1] / strata[st][0]
                    matched_den += 1
                matched = matched_num / matched_den if matched_den else None
                boot = paired_block_bootstrap(sig, pool, key)
                row = {
                    "signals": len(sig),
                    "coverage": round(len(sig) / len(pool), 5),
                    "covered_dates": len({o["session_date"] for o in sig}),
                    "covered_tickers": len({o["ticker"] for o in sig}),
                    "hit_rate": round(hr, 5), "baseline": round(base, 5),
                    "delta": round(hr - base, 5),
                    "close_hit_rate": round(close_hr, 5),
                    "matched_control": None if matched is None else round(matched, 5),
                    "delta_vs_matched": (None if matched is None else round(hr - matched, 5)),
                    "boot": boot,
                    "top_ticker_share": round(
                        max(defaultdict(int, {t: sum(1 for o in sig if o["ticker"] == t)
                                              for t in {o["ticker"] for o in sig}}).values())
                        / len(sig), 4),
                    "mae_median": round(statistics.median(
                        [o["mae"] for o in sig if o["mae"] is not None]), 5),
                }
                results.setdefault(name, {}).setdefault(scope, {})[key] = row
                if scope == "holdout":
                    pvals[f"{name}|{key}"] = (boot["p_one_sided"] if boot else 1.0)

    adj = holm(pvals)
    for nk, p in adj.items():
        name, key = nk.split("|")
        results[name]["holdout"][key]["holm_adjusted_p"] = p

    out = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spec": "spec_v1.md revision A",
        "label": "high touch within 3 trading days, NO stop loss",
        "split": {"dev_sessions": len(dev_dates), "holdout_sessions": len(hold_dates),
                  "holdout_first": min(hold_dates) if hold_dates else None,
                  "holdout_last": max(hold_dates) if hold_dates else None,
                  "purged": len(obs) - len(dev) - len(hold)},
        "rv_tercile_edges": edges,
        "rules": results,
    }
    (HERE / "m4_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    print(f"\n=== HOLDOUT ({min(hold_dates)} .. {max(hold_dates)}, {len(hold)} obs) ===")
    print(f"{'rule':4s} {'tgt':>4} {'sig':>5} {'cov':>6} {'hit':>7} {'base':>7} {'delta':>7} "
          f"{'CI95':>17} {'p':>7} {'holm':>7} {'matchedΔ':>8} {'topTkr':>7}")
    for name in RULES:
        for key in KEYS:
            r = results.get(name, {}).get("holdout", {}).get(key)
            if not r or not r.get("signals"):
                print(f"{name:4s} {key:>4} {'—':>5}")
                continue
            b = r["boot"]
            print(f"{name:4s} {key:>4} {r['signals']:5d} {r['coverage']*100:5.2f}% "
                  f"{r['hit_rate']*100:6.2f}% {r['baseline']*100:6.2f}% {r['delta']*100:+6.2f}pp "
                  f"[{b['ci95'][0]*100:+5.2f},{b['ci95'][1]*100:+5.2f}] "
                  f"{b['p_one_sided']:7.4f} {r.get('holm_adjusted_p'):>7} "
                  f"{(str(round(r['delta_vs_matched']*100,2))+'pp') if r['delta_vs_matched'] is not None else '—':>8} "
                  f"{r['top_ticker_share']*100:6.1f}%")


if __name__ == "__main__":
    main()
