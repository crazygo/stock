"""Assemble the interactive dashboard.html for the growth-trigger study."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_candidates as RC
from gt_common import load_hourly, pinned_tickers, regular_session_closes

RULE_DESC = {
    "R0_always": "基准：每个合格点都买入",
    "R1": "相对强弱>0 且近 20 日高点且量比≥1.2",
    "R2": "量比≥2 且当日上涨且距 20 日高 ≥-3%",
    "R3": "盘前涨幅≥2% 且 20 日相对强弱>0 且 20 日涨幅>0",
    "R4": "20 日涨幅>10% 且 QQQ 20 日>0 且相对强弱>0",
    "R5": "距 20 日低点 ≤2% 且当日上涨且量比≥1.5",
    "R6": "R1 或 R3",
}
CONCLUSION = ("6 条候选规则无一同时满足「≥15 个去重信号、召回率 ≥25%、精准率 ≥1.3×基准且 "
              "bootstrap 95% 下界高于基准」。R3/R4 有统计显著但幅度有限的正边缘（×1.21 / ×1.25），"
              "且集中在 4 只半导体股与 2026-04—05；R3 的事件级去重后边缘基本消失。详见 report.md。")


def main() -> None:
    pts = RC.load_points()
    breadth = RC.build_breadth(pts)
    for p in pts:
        p["f"] = RC.featurize(p, breadth)
        p["split"] = "eval" if int(p["date"][8:10]) % 2 == 1 else "train"
        p["in_win"] = RC.EVAL_START <= p["date"] <= RC.EVAL_END
    ev = [p for p in pts if p["split"] == "eval" and p["in_win"] and "ext_hit" in p]
    events = json.loads((HERE / "events.json").read_text())
    labels = json.loads((HERE / "labels_8pct_v1.json").read_text())

    stocks = {}
    for t in pinned_tickers():
        reg = regular_session_closes(load_hourly(t))
        stocks[t] = {
            "candles": [{"d": d, "o": reg[d]["o"], "h": reg[d]["h"], "l": reg[d]["l"],
                         "c": reg[d]["c"]} for d in sorted(reg)],
            "windows": [{"start_date": w["start_date"], "end_date": w["end_date"],
                         "return": w["return"]} for w in
                        next((s["windows"] for s in labels["stocks"] if s["ticker"] == t), [])],
        }

    sig_fields = ("ret_1d", "ret_5d", "ret_20d", "px_vs_hi20", "px_vs_lo20",
                  "vol_ratio", "premkt_ret", "rel_20d", "qqq_20d")
    signals = []
    for line in (HERE / "signals.jsonl").read_text().splitlines():
        s = json.loads(line)
        signals.append({k: s.get(k) for k in
                        ("ticker", "ts", "date", "sess", "entry", "hit", "max_return")
                        + sig_fields} | {"rule": s["rule"]})

    from datetime import date as _date
    tr_days = {}
    for e in events:
        if e["parity"] == 0:
            tr_days.setdefault(e["ticker"], []).append(e["start_date"])
    eval_ev = [e for e in events if e["parity"] == 1
               and RC.EVAL_START <= e["start_date"] <= RC.EVAL_END
               and not any(abs((_date.fromisoformat(e["start_date"])
                                - _date.fromisoformat(d)).days) <= 14
                           for d in tr_days.get(e["ticker"], []))]

    metrics, pool = {}, {}
    for name, fn in RC.RULES.items():
        pool[name] = {}
        metrics[name] = {}
        for scope, points in (("train", [p for p in pts if p["split"] == "train" and "ext_hit" in p]),
                              ("eval", ev)):
            raw = [p for p in points if fn(p["f"])]
            for mode, sig in (("point", raw), ("day", RC.dedup_day(raw)),
                              ("event", RC.dedup_event(raw))):
                pool[name][mode] = len(points)
                if scope != "eval":
                    continue
                if not sig:
                    metrics[name][mode] = None
                    continue
                prec = sum(s["ext_hit"] for s in sig) / len(sig)
                metrics[name][mode] = {
                    "signals": len(sig), "precision": prec,
                    "recall": RC.metrics(sig, points, eval_ev, 0)["recall"],
                    "lift": prec / (sum(p["ext_hit"] for p in ev) / len(ev)),
                    "ci": RC.bootstrap_lift(sig, sum(p["ext_hit"] for p in ev) / len(ev)),
                }

    # recall per stock (day-level)
    # per-stock recall / precision, per rule, day-level dedup (the primary mode)
    from datetime import date as _d, timedelta as _td
    recall_by_rule = {}
    for name, fn in RC.RULES.items():
        sig = RC.dedup_day([p for p in ev if fn(p["f"])])
        rows = []
        for t in pinned_tickers():
            evs = [e for e in eval_ev if e["ticker"] == t]
            mine = [x for x in sig if x["ticker"] == t]
            cap = 0
            for e in evs:
                s0 = _d.fromisoformat(e["start_date"])
                if any(_d.fromisoformat(x["date"]) in [s0 - _td(days=k) for k in (0, 1, 2, 3)]
                       for x in mine):
                    cap += 1
            rows.append({"t": t, "ev": len(evs), "cap": cap,
                         "r": (cap / len(evs)) if evs else None,
                         "n": len(mine),
                         "p": (sum(1 for x in mine if x["ext_hit"]) / len(mine)) if mine else None})
        recall_by_rule[name] = rows

    payload = {
        "generated_at": labels["generated_at"],
        "stock_count": sum(1 for t in stocks if stocks[t]["candles"]),
        "eval_points": len(ev),
        "eval_events": len(eval_ev),
        "base": sum(p["ext_hit"] for p in ev) / len(ev),
        "rules": list(RC.RULES),
        "ruleDesc": RULE_DESC,
        "conclusion": CONCLUSION,
        "stocks": stocks,
        "signals": signals,
        "metrics": metrics,
        "pool": pool,
        "recall": recall_by_rule,
    }
    html_text = (HERE / "dashboard_template.html").read_text(encoding="utf-8")
    html_text = html_text.replace("/*__DATA__*/", json.dumps(payload, ensure_ascii=False))
    out = HERE / "dashboard.html"
    out.write_text(html_text, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1024:.0f} KB), "
          f"{len(signals)} signals, {len(stocks)} stocks")


if __name__ == "__main__":
    main()
