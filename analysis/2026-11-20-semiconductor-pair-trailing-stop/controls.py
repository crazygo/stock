"""Random entries on the same SOXL/SOXS 5m tape as the November 2026 replay."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SYMBOLS = ("SOXL", "SOXS")
SEED = 20261120
PATHS = 2000
FEE = .001
EPS = 1e-10
BAR = pd.Timedelta(minutes=5)
ET = "America/New_York"


def tape():
    calendar = ROOT / "market_data/calendars/nasdaq_sessions_2026_v2.json"
    sessions = json.loads(calendar.read_text())["sessions"]
    sessions = [r for r in sessions if "2026-11-02" <= r["session_date"] <= "2026-11-19"]
    clock, keys = [], []
    for row in sessions:
        day = row["session_date"]
        midnight = pd.Timestamp(day, tz=ET)
        for start in pd.date_range(midnight - pd.Timedelta(hours=4),
                                   midnight + pd.Timedelta(hours=20),
                                   inclusive="left", freq=BAR):
            if start.hour >= 20 or start.hour < 4:
                session = "overnight"
            elif start < pd.Timestamp(row["open_at_et"]):
                session = "pre_market"
            elif start < pd.Timestamp(row["close_at_et"]):
                session = "regular"
            else:
                session = "post_market"
            clock.append(start)
            keys.append((day, session))
    assert len(sessions) == 14 and len(clock) == 4032
    prices = []
    for symbol in SYMBOLS:
        raw = pd.read_parquet(ROOT / f"market_data/us_5m/{symbol}/2026.parquet")
        raw.index = pd.to_datetime(raw.start_at, utc=True)
        rows = raw.loc[pd.DatetimeIndex(clock).tz_convert("UTC")]
        assert len(rows) == len(clock) and rows.volume.gt(0).all()
        assert rows.price_basis.eq("NONE").all()
        prices.append(rows[["open", "high", "low", "close"]].to_numpy().tolist())
    return clock, keys, prices


def trade(bars, first):
    """Return marked/closed wealth factor, next admissible fill index and outcome."""
    entry = bars[first][0]
    peak, target = entry, 1.08 * entry
    for i in range(first, len(bars)):
        op, hi, lo, close = bars[i]
        # A gap exit at the open frees the slot for the decision 1s later.
        if op <= .97 * peak + EPS or op >= target - EPS:
            raw = op
            reason = "target" if op >= target - EPS else "stop"
            ready = i + 1
            break
        peak = max(peak, op)
        if lo <= .97 * peak + EPS:
            raw, reason = .97 * peak, "stop"
        elif hi >= target - EPS:
            raw, reason = target, "target"
        else:
            peak = max(peak, hi)
            if close > .97 * peak + EPS:
                continue
            raw, reason = .97 * peak, "stop"
        ready = i + 2
        break
    else:
        raw, reason, ready = bars[-1][3], "open", len(bars)
    sell_factor = 1 if reason == "open" else 1 - FEE
    return raw * sell_factor / (entry * (1 + FEE)), ready, reason


def describe(values):
    arr = np.asarray(values)
    return dict(mean=float(arr.mean()), p05=float(np.quantile(arr, .05)),
                median=float(np.median(arr)), p95=float(np.quantile(arr, .95)))


def eligible_entries(clock, n):
    # At the planned entry, the most recent available signal ends 5 min earlier.
    run_start = 0
    result = []
    for i, now in enumerate(clock):
        if i > 0 and now - clock[i-1] != BAR:
            run_start = i
        if i - run_start >= n + 1:
            result.append(i)
    return np.array(result)


def matched_reference(trades, eligible, keys, clock, factors, reasons, rng):
    by_time = {ts: i for i, ts in enumerate(clock)}
    strata = defaultdict(list)
    for i in eligible:
        strata[keys[i]].append(i)
    groups = Counter()
    actual = []
    for row in trades:
        j = SYMBOLS.index(row["symbol"])
        i = by_time[pd.Timestamp(row["entry"])]
        key = row["entry_day"], row["entry_session"]
        assert i in strata[key] and row["exit_reason"] == reasons[j, i], row
        assert abs(factors[j, i] - (1 + row["net"])) < 1e-10, row
        groups[j, key] += 1
        actual.append(factors[j, i])
    returns = np.zeros(PATHS)
    expected = 0.
    for (j, key), count in sorted(groups.items()):
        pool = factors[j, strata[key]] - 1
        # Sampling without replacement inside each day x symbol x session stratum.
        draws = rng.random((PATHS, len(pool)))
        idx = np.argpartition(draws, count - 1, axis=1)[:, :count]
        returns += pool[idx].sum(axis=1)
        expected += count * pool.mean()
    returns *= 100 / len(actual)
    return dict(n=len(actual), strata=len(groups),
                actual_equal_weight_net_pct=100*(np.mean(actual)-1),
                actual_compound_net_pct=100*(np.prod(actual)-1),
                exact_random_mean_net_pct=100*expected/len(actual),
                random_equal_weight_net_pct=describe(returns),
                actual_percentile=float((returns <= 100*(np.mean(actual)-1)).mean()*100))


def expected_n(prob, mix, next_index):
    count = np.zeros(next_index.shape[1]+1)
    for i in reversed(range(len(count)-1)):
        when_filled = 1 + mix*count[next_index[0, i]] + (1-mix)*count[next_index[1, i]]
        count[i] = prob * when_filled + (1-prob) * count[i+1]
    return float(count[0])


def portfolio_reference(trades, eligible, factors, ready, rng, actual_return):
    target_n = len(trades)
    mix = sum(t["symbol"] == SYMBOLS[0] for t in trades)/target_n
    next_index = np.searchsorted(eligible, ready[:, eligible])
    # Match expected frequency on this historical tape. These controls describe
    # this sample, not a live-calibrated strategy or an independent holdout.
    lower, upper = 0., 1.
    for _ in range(32):
        mid = (lower + upper)/2
        if expected_n(mid, mix, next_index) >= target_n:
            upper = mid
        else:
            lower = mid
    prob = upper
    returns, counts = [], []
    for _ in range(PATHS):
        wealth, n, i = 1., 0, 0
        while i < len(eligible):
            i += rng.geometric(prob) - 1
            if i >= len(eligible):
                break
            j = 0 if rng.random() < mix else 1
            wealth *= factors[j, eligible[i]]
            n += 1
            i = next_index[j, i]
        returns.append(100*(wealth-1))
        counts.append(n)
    return dict(target_trades=target_n, soxl_probability=mix, entry_probability=prob,
                expected_trades=expected_n(prob, mix, next_index), trade_counts=describe(counts),
                compound_net_pct=describe(returns), actual_compound_net_pct=actual_return,
                actual_percentile=float((np.asarray(returns) <= actual_return).mean()*100))


def run(ledger):
    clock, keys, prices = tape()
    actual = json.loads(ledger.read_text())["trades"]
    results = [[trade(bars, i) for i in range(len(clock))] for bars in prices]
    factors = np.array([[r[0] for r in row] for row in results])
    ready = np.array([[r[1] for r in row] for row in results])
    reasons = np.array([[r[2] for r in row] for row in results])
    summary = dict(seed=SEED, paths=PATHS, fee_each_side=FEE, cash_return_pct=0,
                   outcomes=dict())
    for n in (3, 5, 10):
        eligible = eligible_entries(clock, n)
        trades = actual[str(n)]
        rng = np.random.default_rng(SEED + n)
        matched = matched_reference(trades, eligible, keys, clock, factors, reasons, rng)
        summary["outcomes"][str(n)] = dict(individually_matched=matched,
            one_position=portfolio_reference(trades, eligible, factors, ready, rng,
                                             matched["actual_compound_net_pct"]))
    output = ledger.with_name("controls.json")
    output.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=HERE / "ledger.json")
    run(parser.parse_args().ledger)
