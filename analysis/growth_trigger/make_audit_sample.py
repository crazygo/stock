"""Draw a human-audit sample: positives, negatives and boundary / rejected cases.

Writes human_audit_sample.json so a reviewer can eyeball labels and outcomes.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_candidates as RC
from gt_common import load_daily_market, load_hourly, pinned_tickers, regular_session_closes

LABEL = dict(target=0.08, min_sessions=3, max_one_day_gain=0.12,
             cal=(3, 14), follow=2, retained=0.04)


def scan_with_reasons(market, ticker):
    """Re-scan the stock and record why each candidate was accepted or rejected."""
    days = market.days
    bars = market.bars[ticker]
    eligible = {i for i in range(len(days))
                if market.eligible(ticker, i, require_next_bar=False)}
    out = []
    last_start = len(bars) - LABEL["min_sessions"] - LABEL["follow"] - 1
    for start in range(0, max(0, last_start + 1)):
        if start not in eligible:
            continue
        sb = bars[start]
        if not sb or float(sb["c"]) <= 0:
            continue
        s0 = float(sb["c"])
        crossing, jump, worst_jump = None, False, 0.0
        for end in range(start + 1, min(start + 40, len(bars) - LABEL["follow"])):
            prev, cur = bars[end - 1], bars[end]
            if not prev or not cur or float(prev["c"]) <= 0:
                break
            j = float(cur["c"]) / float(prev["c"]) - 1
            worst_jump = max(worst_jump, j)
            if j > LABEL["max_one_day_gain"]:
                jump = True
            if float(cur["c"]) / s0 - 1 >= LABEL["target"]:
                crossing = end
                break
        if crossing is None:
            continue
        cal = (date.fromisoformat(days[crossing].isoformat()) -
               date.fromisoformat(days[start].isoformat())).days
        sess = crossing - start
        follow = bars[crossing + 1:crossing + 1 + LABEL["follow"]]
        retained = [float(b["c"]) / s0 - 1 for b in follow if b]
        ok_cal = LABEL["cal"][0] <= cal <= LABEL["cal"][1]
        ok_sess = sess >= LABEL["min_sessions"]
        ok_jump = not jump
        ok_follow = (len(follow) == LABEL["follow"]
                     and all(r >= LABEL["retained"] for r in retained))
        rec = dict(ticker=ticker, start_date=days[start].isoformat(),
                   end_date=days[crossing].isoformat(), sessions=sess, calendar_days=cal,
                   start_close=s0, end_close=float(bars[crossing]["c"]),
                   return_=float(bars[crossing]["c"]) / s0 - 1,
                   worst_single_day_jump=worst_jump,
                   retained_gains=[round(r, 4) for r in retained],
                   checks=dict(calendar=ok_cal, sessions=ok_sess, no_big_jump=ok_jump,
                              follow_through=ok_follow),
                   accepted=all([ok_cal, ok_sess, ok_jump, ok_follow]))
        out.append(rec)
    return out


def main() -> None:
    market = load_daily_market()
    events = json.loads((HERE / "events.json").read_text())
    pts = RC.load_points()
    breadth = RC.build_breadth(pts)
    for p in pts:
        p["f"] = RC.featurize(p, breadth)
        p["split"] = "eval" if int(p["date"][8:10]) % 2 == 1 else "train"
        p["in_win"] = RC.EVAL_START <= p["date"] <= RC.EVAL_END
    ev = [p for p in pts if p["split"] == "eval" and p["in_win"] and "ext_hit" in p]

    rng = random.Random(20260923)
    # positives: eval events
    eval_ev = [e for e in events if e["parity"] == 1
               and RC.EVAL_START <= e["start_date"] <= RC.EVAL_END]
    positives = rng.sample(eval_ev, min(5, len(eval_ev)))
    # negatives: eligible eval points, no event nearby, and no +8% realised
    neg_pool = []
    for p in ev:
        near = any(e["ticker"] == p["ticker"] and abs(
            (date.fromisoformat(e["start_date"]) - date.fromisoformat(p["date"])).days) <= 14
            for e in events)
        if not near and not p["ext_hit"]:
            neg_pool.append(p)
    negatives = rng.sample(neg_pool, min(5, len(neg_pool)))

    # boundary + rejected cases from a full re-scan
    scan = []
    for t in pinned_tickers():
        scan += scan_with_reasons(market, t)
    accepted = [s for s in scan if s["accepted"]]
    rejected = [s for s in scan if not s["accepted"]]
    boundary = sorted(accepted, key=lambda s: (
        abs(s["calendar_days"] - 3) + abs(s["calendar_days"] - 14)
        + abs(s["sessions"] - 3) + abs(s["worst_single_day_jump"] - 0.12)
        + abs(min(s["retained_gains"]) - 0.04)))[:8]
    rejected_sample = rng.sample(rejected, min(8, len(rejected)))

    payload = {
        "generated_at": json.loads((HERE / "labels_8pct_v1.json").read_text())["generated_at"],
        "how_to_review": [
            "正例：确认区间起点/终点、涨幅、以及起点当时是否真的满足财报与流动性门槛。",
            "负例：确认该预测点当时合格、且之后 3–14 个自然日确实没有出现 +8% 触达。",
            "边界例：确认 3/14 自然日端点、3 个交易日间隔、单日 12% 上限、保留 4% 这几条判定没有放水。",
            "被拒例：确认拒绝理由成立（不是把本该收录的区间漏掉）。",
        ],
        "positives": [{k: v for k, v in e.items() if k != "quality"} for e in positives],
        "negatives": [{"ticker": p["ticker"], "ts": p["ts"], "sess": p["sess"],
                       "entry": p["entry"], "max_return": round(p["ext_max_return"], 4),
                       "closest_window": None} for p in negatives],
        "boundary_accepted": boundary,
        "rejected_candidates": rejected_sample,
        "counts": {"accepted_total": len(accepted), "rejected_total": len(rejected),
                   "sampled": len(positives) + len(negatives) + len(boundary)
                   + len(rejected_sample)},
    }
    reasons = {}
    for r in rejected:
        for k, ok in r["checks"].items():
            if not ok:
                reasons[k] = reasons.get(k, 0) + 1
    payload["rejection_reason_counts"] = reasons
    payload["note"] = ("accepted_total 是满足全部条件的候选区间数；最终标签在剔除相互重叠的"
                       "候选后保留涨幅最大的一段，得到 157 段。rejected_total 是找到了 +8% "
                       "收盘穿越但不满足至少一条条件的候选。")
    (HERE / "human_audit_sample.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1))
    print(json.dumps(payload["counts"], indent=1))
    print("rejection reasons:", reasons)
    for b in boundary:
        print("boundary:", b["ticker"], b["start_date"], "->", b["end_date"],
              "cal", b["calendar_days"], "sess", b["sessions"],
              "worst_jump", round(b["worst_single_day_jump"], 3),
              "retained", b["retained_gains"])
    for r in rejected_sample[:5]:
        print("rejected:", r["ticker"], r["start_date"], "->", r["end_date"],
              "cal", r["calendar_days"], "sess", r["sessions"],
              "jump", round(r["worst_single_day_jump"], 3), r["checks"])


if __name__ == "__main__":
    main()
