"""Build the frozen label snapshot, prediction-point sample and descriptive stats.

Writes labels_8pct_v1.json, points.jsonl, events.json, sample_stats.json.
No candidate rules are evaluated here.
"""

from __future__ import annotations

import bisect as _b
import json
import statistics
import sys
import types
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gt_common import (PREDICT_LABELS, ROOT, daily_index, label_of, load_daily_market,
                       load_hourly, pinned_tickers, regular_session_closes, touch_outcome)

LABEL_PARAMS = dict(target_return=0.08, min_trading_close_intervals=3,
                    max_trading_close_intervals=None,
                    calendar_days_inclusive=[3, 14], max_single_session_close_gain=0.12,
                    follow_through_sessions=2, min_gain_retained_vs_start=0.04)
EVAL_START, EVAL_END = "2026-02-01", "2026-09-30"


def build_labels() -> dict:
    src = (ROOT / "identify_growth_windows.py").read_text()
    mod = types.ModuleType("igw8")
    mod.__dict__["__name__"] = "igw8"
    exec(compile(src.replace("if not 3 < calendar_days < 14:",
                             "if not 3 <= calendar_days <= 14:"), "igw8", "exec"),
         mod.__dict__)
    mod.TARGET_RETURN = LABEL_PARAMS["target_return"]
    mod.MIN_SESSIONS = LABEL_PARAMS["min_trading_close_intervals"]
    mod.MAX_SESSIONS = 10 ** 6
    mod.MIN_RETAINED_GAIN = LABEL_PARAMS["min_gain_retained_vs_start"]

    market = load_daily_market()
    manifest = json.loads((ROOT / "market_data/us/manifest.json").read_text())
    stocks, all_windows = [], []
    for ticker in pinned_tickers():
        dated = [({**b, "d": market.days[i].isoformat()} if b else None)
                 for i, b in enumerate(market.bars[ticker])]
        eligible = {i for i in range(len(market.days))
                    if market.eligible(ticker, i, require_next_bar=False)}
        windows = mod.find_growth_windows(dated, lambda i: i in eligible)
        for w in windows:
            w["quality"] = market.quality(ticker, w["start_index"])
        if windows:
            stocks.append({"ticker": ticker, "windows": windows})
        all_windows += [dict(ticker=ticker, **w) for w in windows]
    return {
        "label": "historical_growth_window_8pct_v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "params": LABEL_PARAMS,
        "data_versions": {
            "daily_archive": {"dir": "market_data/us", "sessions": len(market.days),
                              "first": market.days[0].isoformat(),
                              "last": market.days[-1].isoformat(),
                              "manifest_days": len(manifest["archives"])},
            "hourly_archive": {"dir": "market_data/us_60m", "source": "Futu OpenD 60m",
                               "session": "ALL", "extended_time": True,
                               "requested": ["2026-01-01", "2026-09-23"]},
            "sec_facts": "data/ai_sec_annual_facts.json",
        },
        "stock_count": len(stocks),
        "window_count": len(all_windows),
        "stocks": stocks,
    }


def main() -> None:
    market = load_daily_market()
    labels = build_labels()
    (HERE / "labels_8pct_v1.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1))

    day_pos = {d.isoformat(): i for i, d in enumerate(market.days)}
    tickers = pinned_tickers()
    series = {t: load_hourly(t) for t in tickers}
    qqq = load_hourly("QQQ")
    reg = {t: regular_session_closes(series[t]) for t in tickers}
    reg_days = {t: sorted(reg[t]) for t in tickers}
    qreg = regular_session_closes(qqq)
    qdays = sorted(qreg)

    points, unable = [], []
    for t in tickers:
        s = series[t]
        if not s.ts:
            continue
        days = reg_days[t]
        for i, ts in enumerate(s.ts):
            d = ts.strftime("%Y-%m-%d")
            if label_of(ts.strftime("%Y-%m-%d %H:%M:%S")) not in PREDICT_LABELS:
                continue
            j = day_pos.get(d)
            if j is None or not market.eligible(t, j, require_next_bar=False):
                continue
            # running regular-session state, strictly up to and including bar i
            run_hi = run_lo = run_vol = None
            run_close = None
            k = i
            while k >= 0 and s.ts[k].strftime("%Y-%m-%d") == d:
                if s.sess[k] == "reg":
                    run_close = s.c[k]
                    run_hi = s.h[k] if run_hi is None else max(run_hi, s.h[k])
                    run_lo = s.l[k] if run_lo is None else min(run_lo, s.l[k])
                    run_vol = s.v[k] if run_vol is None else run_vol + s.v[k]
                k -= 1
            idx = _b.bisect_right(days, d) - 1
            prev_close = reg[t][days[idx - 1]]["c"] if idx >= 1 else None
            c5 = reg[t][days[idx - 5]]["c"] if idx >= 5 else None
            c20 = reg[t][days[idx - 20]]["c"] if idx >= 20 else None
            lo_i = max(0, idx - 19)
            hi20 = max((reg[t][x]["h"] for x in days[lo_i:idx + 1]), default=None)
            lo20 = min((reg[t][x]["l"] for x in days[lo_i:idx + 1]), default=None)
            vol20 = [reg[t][x]["v"] for x in days[lo_i:idx]]   # completed sessions only
            qk = _b.bisect_right(qdays, d) - 1
            q_prev = qreg[qdays[qk - 1]]["c"] if qk >= 1 else None
            q_now = qreg[qdays[qk]]["c"] if qk >= 0 else None
            q5 = qreg[qdays[qk - 5]]["c"] if qk >= 5 else None
            q20 = qreg[qdays[qk - 20]]["c"] if qk >= 20 else None
            quality = market.quality(t, j)
            rec = {
                "ticker": t, "ts": ts.isoformat(sep=" "), "date": d,
                "label": label_of(ts.strftime("%Y-%m-%d %H:%M:%S")), "sess": s.sess[i],
                "entry": s.c[i], "prev_close": prev_close,
                "run_close": run_close, "run_high": run_hi, "run_low": run_lo,
                "run_vol": run_vol, "c5": c5, "c20": c20, "hi20": hi20, "lo20": lo20,
                "vol20_mean": statistics.mean(vol20) if vol20 else None,
                "q_prev": q_prev, "q_now": q_now, "q5": q5, "q20": q20,
                "revenue": quality["revenue"] if quality else None,
                "ocf": quality["operating_cashflow"] if quality else None,
                "ni": quality["net_income"] if quality else None,
                "latest_filed": quality["latest_filed"] if quality else None,
            }
            out = touch_outcome(s, i, basis="high", use_extended=True)
            out_close = touch_outcome(s, i, basis="close", use_extended=False)
            if out is None or out_close is None:
                unable.append({"ticker": t, "ts": rec["ts"], "reason": "horizon"})
            else:
                rec.update({f"ext_{k}": v for k, v in out.items()})
                rec.update({f"cls_{k}": v for k, v in out_close.items()})
            points.append(rec)

    # stock-days that were eligible on the daily data but have no hourly bar at all
    # (e.g. SPCX hourly history starts 2026-06-09): recorded, never counted as negatives
    no_hourly = []
    for t in tickers:
        have = {ts.strftime("%Y-%m-%d") for ts in series[t].ts}
        for d, i in day_pos.items():
            if d in have or not (EVAL_START <= d <= EVAL_END):
                continue
            if market.eligible(t, i, require_next_bar=False):
                no_hourly.append({"ticker": t, "date": d, "reason": "no_hourly_data"})
    unable.extend(no_hourly)

    with (HERE / "points.jsonl").open("w") as fh:
        for rec in points:
            fh.write(json.dumps(rec) + "\n")
    (HERE / "unable.json").write_text(json.dumps(unable))
    unable_summary = {}
    for u in unable:
        unable_summary[u["reason"]] = unable_summary.get(u["reason"], 0) + 1

    events = []
    for stock in labels["stocks"]:
        for w in stock["windows"]:
            events.append({"ticker": stock["ticker"], **w})
    for e in events:
        e["parity"] = int(e["start_date"][8:10]) % 2
    (HERE / "events.json").write_text(json.dumps(events))

    judged = [p for p in points if "ext_hit" in p]

    def rate(sub):
        return {"n": len(sub),
                "hit_rate": (sum(p["ext_hit"] for p in sub) / len(sub)) if sub else None}

    stats = {
        "generated_at": labels["generated_at"],
        "prediction_points_total": len(points),
        "prediction_points_judged": len(judged),
        "prediction_points_unable": len(unable),
        "unable_reasons": unable_summary,
        "base_rate_all": rate(judged),
        "base_rate_all_close_only": rate([p for p in judged if p.get("cls_hit") is not None]),
        "base_rate_by_session": {k: rate([p for p in judged if p["sess"] == k])
                                 for k in ("pre", "reg", "post")},
        "base_rate_by_ticker": {t: rate([p for p in judged if p["ticker"] == t])
                                for t in tickers},
        "events_total": len(events),
        "events_feb_sep": sum(1 for e in events if EVAL_START <= e["start_date"] <= EVAL_END),
        "events_by_ticker": {t: sum(1 for e in events if e["ticker"] == t) for t in tickers},
    }
    (HERE / "sample_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items()
                      if k not in ("base_rate_by_ticker", "base_rate_by_session",
                                   "events_by_ticker")}, indent=1))
    print("by session:", {k: v["hit_rate"] for k, v in stats["base_rate_by_session"].items()})


if __name__ == "__main__":
    main()
