"""Cross-check the November portfolio ledger against independent entry/exit paths."""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import controls


def key(trade):
    return trade["symbol"], trade["entry"]


def outcomes(trades):
    trades = list(trades)
    closed = [t for t in trades if t["exit_reason"] != "open"]
    return dict(entries=len(trades), closed=len(closed), open=len(trades)-len(closed),
                wins=sum(t["net"] > 0 for t in closed),
                losses=sum(t["net"] < 0 for t in closed),
                flat=sum(t["net"] == 0 for t in closed),
                targets=sum(t["exit_reason"] == "target" for t in closed),
                compound_net_pct=100*(math.prod(1+t["net"] for t in trades)-1))


def intersection(original, candidate):
    old = {key(t): t for t in original}
    new = {key(t): t for t in candidate}
    assert len(old) == len(original) and len(new) == len(candidate)
    for k in old.keys() & new.keys():
        assert abs(old[k]["net"] - new[k]["net"]) < 1e-10, k
    return dict(shared=outcomes(t for k, t in old.items() if k in new),
                original_only=outcomes(t for k, t in old.items() if k not in new),
                candidate_only=outcomes(t for k, t in new.items() if k not in old))


def matched_losses(trades, eligible, keys, factors, reasons):
    """Expected mature/loss counts for the same symbol, day, session and buy count."""
    pool = defaultdict(list)
    for i in eligible:
        pool[keys[i]].append(i)
    strata = Counter((r["symbol"], r["entry_day"], r["entry_session"]) for r in trades)
    expected_closed = expected_losses = 0.
    for (symbol, day, session), count in strata.items():
        j = controls.SYMBOLS.index(symbol)
        candidates = pool[day,session]
        closed = reasons[j,candidates] != "open"
        expected_closed += count * float(np.mean(closed))
        expected_losses += count * float(np.mean(closed & (factors[j,candidates] < 1.0)))
    return dict(expected_closed=expected_closed, expected_losses=expected_losses,
                loss_rate_ratio_of_expected_counts=(expected_losses/expected_closed if expected_closed else None))


def verify_entries(trades, clock, keys, by_time, price, factors, ready, reasons):
    earliest = 0
    prior = set()
    for row in trades:
        j = controls.SYMBOLS.index(row["symbol"])
        i = by_time[pd.Timestamp(row["entry"])]
        assert i >= earliest and i not in prior and i >= 1, row
        assert (row["entry_day"], row["entry_session"]) == keys[i], row
        assert pd.Timestamp(row["signal_end"]) == clock[i]-controls.BAR, row
        assert pd.Timestamp(row["decision"]) == clock[i]-controls.BAR+pd.Timedelta(seconds=1), row
        assert abs(row["entry_price"]-price[j][i][0]) < 1e-10, row
        assert reasons[j,i] == row["exit_reason"], row
        assert abs(factors[j,i] - (1+row["net"])) < 1e-10, row
        if row["exit_reason"] == "open":
            assert abs(price[j][-1][3]-row["mark_price"]) < 1e-10, row
            assert pd.Timestamp(row["mark_at"]) == clock[-1]+controls.BAR
        else:
            exit_i = by_time[pd.Timestamp(row["exit_bar_start"])]
            assert ready[j,i] == exit_i + (1 if row["exit_at_open"] else 2)
            assert keys[exit_i] == (row["exit_day"], row["exit_session"])
            assert pd.Timestamp(row["exit_bar_end"]) == clock[exit_i] + controls.BAR
            exit_price = factors[j,i]*row["entry_price"]*(1+controls.FEE)/(1-controls.FEE)
            assert abs(exit_price-row["exit_price"]) < 1e-9, row
        earliest = ready[j,i]
        prior.add(i)


def check(ledger):
    source = json.loads(ledger.read_text())
    clock, keys, price = controls.tape()
    by_time = {stamp: i for i, stamp in enumerate(clock)}
    paths = [[controls.trade(bars, i) for i in range(len(clock))] for bars in price]
    factors = np.array([[result[0] for result in side] for side in paths])
    ready = np.array([[result[1] for result in side] for side in paths])
    reasons = np.array([[result[2] for result in side] for side in paths])
    actual = source["trades"]
    original = actual[source["original"]]
    archived = json.loads((HERE.parent/"ledger.json").read_text())["trades"]["5"]
    assert [key(t) for t in archived] == [key(t) for t in original]
    checked = {}
    for name, trades in actual.items():
        verify_entries(trades, clock, keys, by_time, price, factors, ready, reasons)
        checked[name] = dict(**outcomes(trades), versus_original=intersection(original, trades))

    selected = source["selected"]
    trades = actual.get(selected, [])
    out = dict(selected=selected, verified=checked,
               overlap_definition="Identical symbol and five-minute entry time; original-only also includes differences from portfolio occupancy.",
               random_seed=controls.SEED+5, paths=controls.PATHS, cash_net_pct=0.0,
               random_pool="All delayed-entry slots eligible for the original five-bar rule's warmup, without a momentum requirement.")
    if trades:
        # Each eligible random entry obeys the same week-reset five-bar warmup and delay.
        eligible = controls.eligible_entries(clock, 5)
        rng = np.random.default_rng(out["random_seed"])
        matched = controls.matched_reference(trades, eligible, keys, clock, factors, reasons, rng)
        matched.update(matched_losses(trades, eligible, keys, factors, reasons))
        mix = sum(t["symbol"] == "SOXL" for t in trades) / len(trades)
        max_n = controls.expected_n(1., mix, np.searchsorted(eligible, ready[:,eligible]))
        one = dict(requested_trades=len(trades), always_enter_expected_trades=max_n)
        if len(trades) > max_n + 1e-6:
            one["status"] = "target frequency exceeds fit range for the random entry control"
        else:
            one.update(controls.portfolio_reference(trades, eligible, factors, ready, rng,
                                                    matched["actual_compound_net_pct"]))
            one["frequency_matched"] = abs(one["expected_trades"] - len(trades)) < 1e-4
        out.update(matched_individual=matched, random_single_holding=one)
    else:
        out["random_control_status"] = "no candidate entries"
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=HERE/"ledger.json")
    parser.add_argument("--output", type=Path, default=HERE/"check.json")
    args = parser.parse_args()
    args.output.write_text(json.dumps(check(args.ledger), indent=2, allow_nan=False)+"\n")
