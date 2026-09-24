"""Q1 (revised): five-target baselines with honest uncertainty (spec_v1.md M3.2).

Corrections applied after review:
  * the binomial CI is reported only as an "independent-sample reference"; the formal
    interval is a calendar-block bootstrap because observations inside one session and
    across stocks in the same session are strongly dependent;
  * an episode reconciliation table is emitted (total / attributed / unattributed);
  * "ever touched a down barrier" is reported separately from "touched it first";
  * no claim is made about why an earlier study reported a different base rate.
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
from observations import DOWN_BARRIERS, TARGETS, build_observations
from segments import Bar, build_segments

SKIP = {"SNOW", "SPCX", "TSM"}
OBS_OUT = HERE / "observations.jsonl"
KEYS = [f"{int(q*100)}%" for q in TARGETS]


def all_bars(ticker: str) -> list[Bar]:
    s = load_hourly(ticker)
    return [Bar(s.ts[i], s.o[i], s.h[i], s.l[i], s.c[i], s.v[i]) for i in range(len(s.ts))]


def build_all() -> list:
    obs = []
    for t in pinned_tickers():
        if t in SKIP:
            continue
        bars = all_bars(t)
        segs = build_segments(bars)          # post-hoc label segmentation, episode ids only
        obs += build_observations(t, bars, segs)
    return obs


def dump(obs):
    with OBS_OUT.open("w") as fh:
        for o in obs:
            fh.write(json.dumps({
                "ticker": o.ticker, "signal_ts": o.signal_ts, "entry_ts": o.entry_ts,
                "entry_price": o.entry_price, "session_date": o.session_date,
                "horizon_end": o.horizon_end, "mfe": o.mfe, "mae": o.mae,
                "mfe_close": o.mfe_close, "mae_close": o.mae_close,
                "labels": o.labels, "tau": o.tau, "tau_down": o.tau_down,
                "tau_ts": o.tau_ts, "tau_down_ts": o.tau_down_ts,
                "bars_to_mfe": o.bars_to_mfe, "episode_id": o.episode_id,
                "judged": o.judged, "reason": o.reason,
            }) + "\n")


def block_bootstrap(judged, key, blocks, iters=2000, seed=20260924):
    """Resample whole calendar blocks (all stocks sharing a date) with replacement."""
    rng = random.Random(seed)
    by_date = defaultdict(list)
    for o in judged:
        by_date[o.session_date].append(o)
    dates = sorted(by_date)
    vals = []
    for _ in range(iters):
        pick = [dates[rng.randrange(len(dates))] for _ in range(len(dates))]
        sample = [o for d in pick for o in by_date[d]]
        vals.append(sum(o.labels[key] for o in sample) / len(sample))
    vals.sort()
    return (vals[int(0.025 * iters)], vals[int(0.975 * iters)])


def multi_day_blocks(judged, key, block_days, iters=2000, seed=20260924):
    """Sensitivity: resample blocks of *block_days* consecutive trading days."""
    rng = random.Random(seed)
    by_date = defaultdict(list)
    for o in judged:
        by_date[o.session_date].append(o)
    dates = sorted(by_date)
    chunks = [dates[i:i + block_days] for i in range(0, len(dates), block_days)]
    vals = []
    for _ in range(iters):
        pick = [c[rng.randrange(len(chunks))] for c in
                [chunks] * len(chunks)]
        sample = [o for c in pick for d in c for o in by_date[d]]
        vals.append(sum(o.labels[key] for o in sample) / len(sample))
    vals.sort()
    return (vals[int(0.025 * iters)], vals[int(0.975 * iters)])


def wilson(hits, n, z=1.96):
    if n == 0:
        return [None, None]
    p = hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - m) / d, 6), round((c + m) / d, 6)]


def dist(vals):
    if not vals:
        return {}
    s = sorted(vals)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return {"min": round(s[0], 4), "p25": round(q(0.25), 4), "p50": round(q(0.50), 4),
            "p75": round(q(0.75), 4), "p90": round(q(0.90), 4),
            "p99": round(q(0.99), 4), "max": round(s[-1], 4)}


def main() -> None:
    obs = build_all()
    dump(obs)
    judged = [o for o in obs if o.judged]
    unjudged = [o for o in obs if not o.judged]
    n = len(judged)

    res = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "spec": "spec_v1.md",
        "primary_label": "Y_high(q) = 1[max High_u / Entry - 1 >= q], u in (t, t+3 trading days]",
        "sensitivity_label": "Y_close(q) = 1[max Close_u / Entry - 1 >= q]",
        "entry_rule": "open of the next tradable regular-session bar (baseline simulated entry, not a fill guarantee)",
        "sample": {
            "observations_total": len(obs), "observations_judged": n,
            "observations_unjudged": len(unjudged),
            "unjudged_reasons": {r: sum(1 for o in unjudged if o.reason == r)
                                 for r in {o.reason for o in unjudged}},
            "sample_is_fully_observed": True,
            "sample_note": ("every judged observation has its full 3-trading-day horizon "
                            "inside the archive; observations whose horizon would run past "
                            "the last session are dropped, not kept as negatives"),
            "stocks": sorted({o.ticker for o in obs}),
            "sessions": len({o.session_date for o in obs}),
            "first_session": min(o.session_date for o in obs),
            "last_session": max(o.session_date for o in obs),
        },
        "baselines_high": {}, "baselines_close": {},
        "episode_reconciliation": {}, "barrier_order": {},
        "down_barrier_ever_touched": {},
    }

    for k in KEYS:
        hits = sum(o.labels[k] for o in judged)
        res["baselines_high"][k] = {
            "n": n, "hits": hits, "rate": round(hits / n, 6),
            "binom_ci_independent_reference": wilson(hits, n),
            "calendar_block_ci": [round(x, 6) for x in block_bootstrap(judged, k, None)],
            "five_day_block_ci": [round(x, 6) for x in multi_day_blocks(judged, k, 5)],
            "bars_to_target_median": statistics.median(
                [o.tau[k] for o in judged if o.tau.get(k) is not None]) if any(
                o.tau.get(k) is not None for o in judged) else None,
        }
        ch = sum(1 for o in judged if o.mfe_close is not None and o.mfe_close >= TARGETS[KEYS.index(k)])
        res["baselines_close"][k] = {"n": n, "hits": ch, "rate": round(ch / n, 6),
                                     "binom_ci_independent_reference": wilson(ch, n),
                                     "calendar_block_ci": [round(x, 6) for x in
                                                           block_bootstrap(judged, k, None)]
                                     if False else None}
    res["monotonic_high"] = all(
        res["baselines_high"][KEYS[i]]["rate"] >= res["baselines_high"][KEYS[i + 1]]["rate"]
        for i in range(len(KEYS) - 1))
    res["mfe_distribution"] = dist([o.mfe for o in judged if o.mfe is not None])
    res["mae_distribution"] = dist([o.mae for o in judged if o.mae is not None])
    res["high_vs_close_gap_pp"] = {
        k: round((res["baselines_high"][k]["rate"] - res["baselines_close"][k]["rate"]) * 100, 3)
        for k in KEYS}

    # ---- episode reconciliation ----
    attributed = [o for o in judged if o.episode_id]
    by_ep = defaultdict(list)
    for o in attributed:
        by_ep[(o.ticker, o.episode_id)].append(o)
    sizes = [len(v) for v in by_ep.values()]
    ep_rate = {}
    for k in KEYS:
        hit_eps = sum(1 for v in by_ep.values() if any(o.labels[k] for o in v))
        ep_rate[k] = {"episodes": len(by_ep), "episodes_with_a_hit": hit_eps,
                      "share_of_episodes_with_a_successful_entry": round(hit_eps / len(by_ep), 6)}
    res["episode_reconciliation"] = {
        "judged_observations": n,
        "attributed_to_an_episode": len(attributed),
        "unattributed": n - len(attributed),
        "episodes": len(by_ep),
        "observations_per_episode_mean": round(len(attributed) / len(by_ep), 2) if by_ep else None,
        "observations_per_episode_median": statistics.median(sizes) if sizes else None,
        "observations_per_episode_max": max(sizes) if sizes else None,
        "episode_id_includes_ticker": True,
        "by_target": ep_rate,
        "warning": ("share_of_episodes_with_a_successful_entry gives every episode many "
                    "chances, so it is NOT a precision and must not replace the "
                    "observation-level rate or feed a live confidence number"),
    }

    # ---- barrier ordering (this is the only supportable risk statement) ----
    for k in KEYS:
        for dk in [f"{int(d*100)}%" for d in DOWN_BARRIERS]:
            up_first = same_bar = down_first = neither = 0
            for o in judged:
                tq, td = o.tau.get(k), o.tau_down.get(dk)
                if tq is None and td is None:
                    neither += 1
                elif tq is not None and td is None:
                    up_first += 1
                elif td is not None and tq is None:
                    down_first += 1
                elif tq < td:
                    up_first += 1
                elif td < tq:
                    down_first += 1
                else:
                    same_bar += 1
            res["barrier_order"][f"up_{k}_vs_down_{dk}"] = {
                "up_barrier_first": round(up_first / n, 6),
                "down_barrier_first": round(down_first / n, 6),
                "same_bar_ambiguous": round(same_bar / n, 6),
                "neither_within_3d": round(neither / n, 6)}
    for dk in [f"{int(d*100)}%" for d in DOWN_BARRIERS]:
        res["down_barrier_ever_touched"][dk] = round(
            sum(1 for o in judged if o.tau_down.get(dk) is not None) / n, 6)

    (HERE / "baselines.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))

    print(f"observations {len(obs)} | judged {n} | unjudged {len(unjudged)} "
          f"{res['sample']['unjudged_reasons']}")
    print(f"sessions {res['sample']['sessions']} ({res['sample']['first_session']} .. "
          f"{res['sample']['last_session']}), stocks {len(res['sample']['stocks'])}")
    print("\n五档基准率（主标签 = 最高价触及）")
    print(f"{'目标':>5} {'命中率':>8} {'二项CI(仅参照)':>18} {'日历块bootstrap CI':>20} {'5日块CI':>18} {'收盘价口径':>9}")
    for k in KEYS:
        b = res["baselines_high"][k]
        print(f"+{k:>4} {b['rate']*100:7.2f}%  [{b['binom_ci_independent_reference'][0]*100:.2f},"
              f"{b['binom_ci_independent_reference'][1]*100:.2f}]  "
              f"[{b['calendar_block_ci'][0]*100:.2f},{b['calendar_block_ci'][1]*100:.2f}]  "
              f"[{b['five_day_block_ci'][0]*100:.2f},{b['five_day_block_ci'][1]*100:.2f}]  "
              f"{res['baselines_close'][k]['rate']*100:7.2f}%")
    print(f"\n单调性: {res['monotonic_high']}")
    print(f"High vs Close 差(pp): {res['high_vs_close_gap_pp']}")
    er = res["episode_reconciliation"]
    print(f"\nepisode 对账: judged={er['judged_observations']} 归属={er['attributed_to_an_episode']} "
          f"未归属={er['unattributed']} episodes={er['episodes']} "
          f"均长={er['observations_per_episode_mean']} 中位={er['observations_per_episode_median']} "
          f"最长={er['observations_per_episode_max']}")
    print(f"episode 内存在成功入口的比例（不是精准率）: "
          f"{ {k: v['share_of_episodes_with_a_successful_entry'] for k,v in er['by_target'].items()} }")
    print(f"\n曾经触及下障碍（不是'先跌'）: {res['down_barrier_ever_touched']}")
    print("障碍先后顺序（up+8% vs down-5%）:",
          res["barrier_order"]["up_8%_vs_down_5%"])


if __name__ == "__main__":
    main()
