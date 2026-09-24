"""Evaluate the frozen candidate rules on the frozen sample.

Train = even day-of-month, eval = odd day-of-month inside 2026-02..2026-09.
Eval events overlapping a same-stock train event within 14 calendar days are dropped.
Thresholds come from spec.md v1 and are not modified here.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gt_common import pinned_tickers

EVAL_START, EVAL_END = "2026-02-01", "2026-09-30"
MIN_SIGNALS, MIN_RECALL, LIFT = 15, 0.25, 1.3


def load_points():
    pts = []
    with (HERE / "points.jsonl").open() as fh:
        for line in fh:
            pts.append(json.loads(line))
    return pts


def featurize(p, breadth):
    last = p["run_close"] if p["run_close"] is not None else p["prev_close"]
    f = {}
    f["ret_1d"] = (p["run_close"] / p["prev_close"] - 1
                   if p["run_close"] is not None and p["prev_close"] else None)
    f["ret_5d"] = (last / p["c5"] - 1) if last and p["c5"] else None
    f["ret_20d"] = (last / p["c20"] - 1) if last and p["c20"] else None
    f["px_vs_hi20"] = (last / p["hi20"] - 1) if last and p["hi20"] else None
    f["px_vs_lo20"] = (last / p["lo20"] - 1) if last and p["lo20"] else None
    f["vol_ratio"] = (p["run_vol"] / p["vol20_mean"]) if p["run_vol"] and p["vol20_mean"] else None
    f["premkt_ret"] = (p["entry"] / p["prev_close"] - 1) if p["prev_close"] else None
    if p["run_high"] and p["run_low"] and p["run_high"] > p["run_low"]:
        f["intraday_pos"] = (p["entry"] - p["run_low"]) / (p["run_high"] - p["run_low"])
    else:
        f["intraday_pos"] = None
    f["rel_20d"] = (f["ret_20d"] - (p["q_now"] / p["q20"] - 1)
                    if f["ret_20d"] is not None and p["q_now"] and p["q20"] else None)
    f["qqq_20d"] = (p["q_now"] / p["q20"] - 1) if p["q_now"] and p["q20"] else None
    f["breadth_5d"] = breadth.get(p["date"])
    f["days_since_filing"] = ((date.fromisoformat(p["date"]) -
                               date.fromisoformat(p["latest_filed"])).days
                              if p["latest_filed"] else None)
    f["revenue"], f["ocf"], f["ni"] = p["revenue"], p["ocf"], p["ni"]
    return f


def build_breadth(points):
    """Fraction of the 21 pinned stocks with a positive 5-session return, per date."""
    by_ticker_date = defaultdict(dict)
    for p in points:
        by_ticker_date[p["ticker"]][p["date"]] = p
    dates = sorted({p["date"] for p in points})
    out = {}
    for d in dates:
        pos = tot = 0
        for t in pinned_tickers():
            rec = by_ticker_date[t].get(d)
            if not rec or rec["c5"] is None or rec["prev_close"] is None:
                continue
            tot += 1
            if rec["prev_close"] / rec["c5"] - 1 > 0:
                pos += 1
        out[d] = pos / tot if tot else None
    return out


RULES = {
    "R1": lambda f: (f["rel_20d"] is not None and f["rel_20d"] > 0
                     and f["px_vs_hi20"] is not None and f["px_vs_hi20"] >= -0.01
                     and f["vol_ratio"] is not None and f["vol_ratio"] >= 1.2),
    "R2": lambda f: (f["vol_ratio"] is not None and f["vol_ratio"] >= 2.0
                     and f["ret_1d"] is not None and f["ret_1d"] > 0
                     and f["px_vs_hi20"] is not None and f["px_vs_hi20"] >= -0.03),
    "R3": lambda f: (f["premkt_ret"] is not None and f["premkt_ret"] >= 0.02
                     and f["rel_20d"] is not None and f["rel_20d"] > 0
                     and f["ret_20d"] is not None and f["ret_20d"] > 0),
    "R4": lambda f: (f["ret_20d"] is not None and f["ret_20d"] > 0.10
                     and f["qqq_20d"] is not None and f["qqq_20d"] > 0
                     and f["rel_20d"] is not None and f["rel_20d"] > 0),
    "R5": lambda f: (f["px_vs_lo20"] is not None and f["px_vs_lo20"] <= 0.02
                     and f["ret_1d"] is not None and f["ret_1d"] > 0
                     and f["vol_ratio"] is not None and f["vol_ratio"] >= 1.5),
}


RULES["R6"] = lambda f: RULES["R1"](f) or RULES["R3"](f)


def market_state(f):
    q = f["qqq_20d"]
    if q is None:
        return "unknown"
    return "up" if q > 0.03 else ("down" if q < -0.03 else "flat")


def dedup_day(points):
    """At most one buy signal per (ticker, date): keep the earliest."""
    best = {}
    for p in points:
        k = (p["ticker"], p["date"])
        if k not in best or p["ts"] < best[k]["ts"]:
            best[k] = p
    return sorted(best.values(), key=lambda p: p["ts"])


def dedup_event(points, gap_days=14):
    """At most one buy signal per ticker within any 14-calendar-day span."""
    kept = []
    last = {}
    for p in sorted(points, key=lambda x: x["ts"]):
        prev = last.get(p["ticker"])
        if prev is None or (date.fromisoformat(p["date"]) - prev).days > gap_days:
            kept.append(p)
            last[p["ticker"]] = date.fromisoformat(p["date"])
    return kept


def metrics(signals, eligible, events, base_rate):
    n = len(signals)
    hits = sum(1 for s in signals if s["ext_hit"])
    prec = hits / n if n else None
    fp_per_100 = (n - hits) / len(eligible) * 100 if eligible else None
    # recall: an event is captured if a signal lands in [start-3d, start]
    captured = 0
    for e in events:
        s0 = date.fromisoformat(e["start_date"])
        lo, hi = s0 - timedelta(days=3), s0
        if any(date.fromisoformat(sig["date"]) >= lo and date.fromisoformat(sig["date"]) <= hi
               and sig["ticker"] == e["ticker"] for sig in signals):
            captured += 1
    recall = captured / len(events) if events else None
    leads = [(date.fromisoformat(e["start_date"]) - date.fromisoformat(s["date"])).days
             for e in events for s in signals
             if s["ticker"] == e["ticker"]
             and 0 <= (date.fromisoformat(e["start_date"]) - date.fromisoformat(s["date"])).days <= 3]
    return {"signals": n, "hits": hits, "precision": prec,
            "recall": recall, "captured": captured, "events": len(events),
            "fp_per_100_eligible": fp_per_100,
            "trigger_freq": n / len(eligible) if eligible else None,
            "lead_days": leads,
            "lift_vs_base": (prec / base_rate) if (prec and base_rate) else None}


def bootstrap_lift(signals, base_rate, iters=2000, seed=20260923):
    rng = random.Random(seed)
    n = len(signals)
    hits = [1 if s["ext_hit"] else 0 for s in signals]
    if n == 0:
        return None, None
    vals = []
    for _ in range(iters):
        sample = [hits[rng.randrange(n)] for _ in range(n)]
        vals.append(sum(sample) / n)
    vals.sort()
    lo = vals[int(0.025 * iters)]
    hi = vals[int(0.975 * iters)]
    return lo, hi


def main() -> None:
    points = load_points()
    breadth = build_breadth(points)
    for p in points:
        p["f"] = featurize(p, breadth)
        p["split"] = "eval" if int(p["date"][8:10]) % 2 == 1 else "train"
        p["in_eval_window"] = EVAL_START <= p["date"] <= EVAL_END

    events = json.loads((HERE / "events.json").read_text())
    # eval events: odd start date inside the study window, no same-stock train event within 14 days
    train_events = [e for e in events if e["parity"] == 0]
    train_days = defaultdict(list)
    for e in train_events:
        d = date.fromisoformat(e["start_date"])
        train_days[e["ticker"]].append(d)
    eval_events, dropped = [], []
    for e in events:
        if e["parity"] != 1 or not (EVAL_START <= e["start_date"] <= EVAL_END):
            continue
        s = date.fromisoformat(e["start_date"])
        if any(abs((s - t).days) <= 14 for t in train_days[e["ticker"]]):
            dropped.append(e)
        else:
            eval_events.append(e)
    eval_points = [p for p in points if p["split"] == "eval" and p["in_eval_window"] and "ext_hit" in p]
    train_points = [p for p in points if p["split"] == "train" and "ext_hit" in p]
    base = sum(p["ext_hit"] for p in eval_points) / len(eval_points)

    print(f"eval points={len(eval_points)} train points={len(train_points)} "
          f"base_rate_eval={base:.4f}")
    print(f"eval events={len(eval_events)} (dropped for overlap={len(dropped)})")
    by_ticker = defaultdict(int)
    for e in eval_events:
        by_ticker[e["ticker"]] += 1
    print("eval events by ticker:", dict(sorted(by_ticker.items(), key=lambda x: -x[1])))

    report = {"base_rate_eval": base, "eval_points": len(eval_points),
              "eval_events": len(eval_events), "dropped_events": len(dropped),
              "eval_events_by_ticker": dict(sorted(by_ticker.items(), key=lambda x: -x[1])),
              "rules": {}}
    r0 = metrics(eval_points, eval_points, eval_events, base)
    r0["bootstrap_precision_ci"] = bootstrap_lift(eval_points, base)
    report["rules"]["R0_always"] = {"eval_day": r0, "eval_point": r0}
    signals_all = []
    for name, fn in RULES.items():
        for scope, pool in (("train", train_points), ("eval", eval_points)):
            raw = [p for p in pool if fn(p["f"])]
            for mode, sig in (("point", raw),
                              ("day", dedup_day(raw)),
                              ("event", dedup_event(raw))):
                m = metrics(sig, pool, eval_events if scope == "eval" else
                            [e for e in events if e["parity"] == 0], base)
                m["bootstrap_precision_ci"] = bootstrap_lift(sig, base)
                report["rules"].setdefault(name, {})[f"{scope}_{mode}"] = m
                if scope == "eval" and mode == "day":
                    signals_all += [dict(ticker=s["ticker"], ts=s["ts"], date=s["date"],
                                         sess=s["sess"], entry=s["entry"], hit=s["ext_hit"],
                                         max_return=s["ext_max_return"], rule=name,
                                         **{k: v for k, v in s["f"].items()
                                            if k in ("ret_1d", "ret_5d", "ret_20d",
                                                     "px_vs_hi20", "px_vs_lo20",
                                                     "vol_ratio", "premkt_ret",
                                                     "rel_20d", "qqq_20d", "breadth_5d")})
                                     for s in sig]
    with (HERE / "signals.jsonl").open("w") as fh:
        for s in signals_all:
            fh.write(json.dumps(s) + "\n")
    (HERE / "eval_report.json").write_text(json.dumps(report, indent=1))

    for name in ["R0_always"] + list(RULES):
        r = report["rules"][name]["eval_day"]
        ci = r["bootstrap_precision_ci"]
        print(f"{name:12s} signals={r['signals']:4d} prec={r['precision'] if r['precision'] is None else round(r['precision'],4)} "
              f"recall={r['recall'] if r['recall'] is None else round(r['recall'],3)} "
              f"fp/100={r['fp_per_100_eligible'] if r['fp_per_100_eligible'] is None else round(r['fp_per_100_eligible'],2)} "
              f"lift={r['lift_vs_base'] if r['lift_vs_base'] is None else round(r['lift_vs_base'],3)} "
              f"CI={None if ci[0] is None else (round(ci[0],3), round(ci[1],3))}")
    (HERE / "eval_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
