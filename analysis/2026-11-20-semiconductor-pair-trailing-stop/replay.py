"""Replay the fixed 14-day SOXL/SOXS bracket strategy using 5-minute OHLCV."""
import gzip
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ET = ZoneInfo("America/New_York")
STEP = timedelta(minutes=5)
SYMBOLS = ("SOXL", "SOXS")
WINDOWS = (3, 5, 10)
COST = .001
EPS = 1e-10


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def date_grid():
    calendar = ROOT / "market_data/calendars/nasdaq_sessions_2026_v2.json"
    sessions = json.loads(calendar.read_text())["sessions"]
    dates = [s["session_date"] for s in sessions if s["session_date"] <= "2026-11-19"][-14:]
    times, labels = [], []
    for day in dates:
        start = datetime.fromisoformat(day).replace(tzinfo=ET) - timedelta(hours=4)
        end = start + timedelta(days=1)
        while start < end:
            times.append(start)
            labels.append(day)
            start += STEP
    return times, labels


def session(t):
    if t.hour < 4 or t.hour >= 20:
        return "overnight"
    if (t.hour, t.minute) < (9, 30):
        return "pre_market"
    return "regular" if t.hour < 16 else "post_market"


def load_bars(times):
    tape, inputs = {}, {}
    grid = pd.DatetimeIndex(times).tz_convert("UTC")
    for symbol in SYMBOLS:
        path = ROOT / f"market_data/us_5m/{symbol}/2026.parquet"
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.start_at, utc=True)
        rows = frame.loc[frame.index.isin(grid)].reindex(grid)
        prices = rows[["open", "high", "low", "close"]]
        assert np.isfinite(prices.to_numpy()).all() and (prices > 0).all().all()
        assert rows.volume.gt(0).all() and rows.price_basis.eq("NONE").all()
        assert (prices.low <= prices[["open", "close"]].min(axis=1)).all()
        assert (prices.high >= prices[["open", "close"]].max(axis=1)).all()
        assert (pd.to_datetime(rows.end_at, utc=True) == grid + STEP).all()
        assert (pd.to_datetime(rows.available_at, utc=True) == grid + STEP + timedelta(seconds=1)).all()
        ca = pd.read_parquet(ROOT / f"market_data/corporate_actions/{symbol}.parquet")
        first_day = (times[0] + timedelta(hours=4)).date().isoformat()
        last_day = (times[-1] + timedelta(hours=4)).date().isoformat()
        assert not ca.ex_div_date.astype(str).between(first_day, last_day).any()
        content = rows[["symbol", "end_at", "available_at", "open", "high", "low", "close", "volume", "price_basis"]]
        inputs[symbol] = dict(path=str(path.relative_to(ROOT)), rows=len(rows),
                              used_rows_sha256=hashlib.sha256(content.to_csv(index=False).encode()).hexdigest())
        tape[symbol] = list(prices.itertuples(index=False, name=None))
    return tape, inputs


def signal(window):
    first, last = window[0][0], window[-1][3]
    peak = worst = 0.
    blocked = False
    for _, high, low, _ in window:
        peak = max(peak, high)
        worst = max(worst, 1 - low / peak)
        blocked |= low <= peak * .97 + EPS
    gain = last / first - 1
    return dict(gain=gain, drawdown=worst, rising=last > first, passed=last > first and not blocked)


def signals(times, tape, n):
    out, start = [], 0
    for i, clock in enumerate(times):
        if i and times[i - 1] + STEP != clock:
            start = i
        out.append({s: signal(tape[s][i-n:i]) for s in SYMBOLS} if i - start >= n else {})
    return out


@dataclass
class Holding:
    trade: dict
    peak: float
    target: float
    units: float
    gross_units: float


def process_price(pos, price, at_open=False):
    # Continuous segments cross at the threshold. A gap can fill beyond it.
    stop = pos.peak * .97
    if price <= stop + EPS:
        return "stop", price if at_open else stop
    if price >= pos.target - EPS:
        return "target", price if at_open else pos.target
    pos.peak = max(pos.peak, price)
    return None


def replay(times, days, tape, features, path, instant=False, keep_checks=False):
    cash = gross_cash = 1.
    pos = pending = None
    trades, daily, checks = [], {}, []
    counts, gate, flat_reasons = Counter(), Counter(), Counter()
    last_equity = running_high = 1.
    worst_dd = 0.

    def buy(order, i):
        symbol = order["symbol"]
        price = tape[symbol][i][0]
        trade = dict(**order, entry=times[i].isoformat(), entry_day=days[i],
                     entry_session=session(times[i]), entry_price=price)
        trades.append(trade)
        return Holding(trade, price, price * 1.08, cash / (price * (1+COST)), gross_cash / price)

    def sell(holding, event, i, at_open=False):
        reason, price = event
        tr = holding.trade
        tr.update(exit_reason=reason, exit_bar_start=times[i].isoformat(), exit_bar_end=(times[i]+STEP).isoformat(),
                  exit_day=days[i], exit_session=session(times[i]), exit_at_open=at_open, exit_price=price,
                  gross=price / tr["entry_price"] - 1,
                  net=price / tr["entry_price"] * (1-COST) / (1+COST) - 1)
        return holding.units * price * (1-COST), holding.gross_units * price

    for i, clock in enumerate(times):
        if pending is not None:
            assert pos is None
            pos = buy(pending, i)
            pending = None
        if pos is not None:
            event = process_price(pos, tape[pos.trade["symbol"]][i][0], at_open=True)
            if event:
                cash, gross_cash = sell(pos, event, i, at_open=True)
                pos = None

        f = features[i]
        choices = sorted((s for s in f if f[s]["passed"]), key=lambda s: (-f[s]["gain"], s))
        closed = not instant and (i + 1 == len(times) or times[i+1] != clock + STEP)
        why = "pass" if choices else "drawdown" if any(v["rising"] for v in f.values()) else "no_rise"
        if f and not closed:
            gate[why] += 1
        if not f:
            status = "warmup"
        elif closed:
            status = "no_fill_slot"
        elif pos:
            status = "holding"
        elif not choices:
            status = "filter_blocked"
            flat_reasons[why] += 1
        else:
            status = "buy"
            symbol = choices[0]
            order = dict(symbol=symbol, signal_end=clock.isoformat(), decision=(clock if instant else clock + timedelta(seconds=1)).isoformat(),
                         window_return=f[symbol]["gain"], window_drawdown=f[symbol]["drawdown"])
            if instant:
                pos = buy(order, i)
            else:
                pending = order

        counts[status] += 1
        if keep_checks:
            checks.append(dict(cutoff=clock.isoformat(), status=status, features=f,
                               picked=choices[0] if status=="buy" else None))

        if pos is not None:
            _, high, low, close = tape[pos.trade["symbol"]][i]
            sequence = (low, high, close) if path == "OLHC" else (high, low, close)
            for next_price in sequence:
                event = process_price(pos, next_price)
                if event:
                    cash, gross_cash = sell(pos, event, i)
                    pos = None
                    break
        if pos:
            mark = tape[pos.trade["symbol"]][i][3]
            equity, gross_equity = pos.units * mark, pos.gross_units * mark
        else:
            equity, gross_equity = cash, gross_cash
        running_high = max(running_high, equity)
        worst_dd = max(worst_dd, 1 - equity/running_high)
        if i+1 == len(times) or days[i+1] != days[i]:
            daily[days[i]] = dict(end_equity=equity, day_return=equity/last_equity-1)
            last_equity = equity

    assert pending is None
    if pos is not None:
        tr = pos.trade
        mark = tape[tr["symbol"]][-1][3]
        tr.update(exit_reason="open", mark_at=(times[-1]+STEP).isoformat(), mark_price=mark,
                  gross=mark/tr["entry_price"]-1, net=mark/tr["entry_price"]/(1+COST)-1)
    done = [t for t in trades if t["exit_reason"] != "open"]
    def breakdown(group):
        return dict(buys=len(group), target=sum(t["exit_reason"]=="target" for t in group),
                    stop=sum(t["exit_reason"]=="stop" for t in group),
                    net_wins=sum(t["exit_reason"]!="open" and t["net"]>0 for t in group),
                    gross_wins=sum(t["exit_reason"]!="open" and t["gross"]>0 for t in group))
    for day, row in daily.items():
        row.update(entries=sum(t["entry_day"] == day for t in trades),
                   exits=sum(t.get("exit_day") == day for t in done))
    assert len(trades)==counts["buy"] and sum(counts.values())==len(times)
    assert math.isclose(math.prod(1+t["net"] for t in trades), equity, rel_tol=1e-11)
    assert math.isclose(math.prod(1+t["gross"] for t in trades), gross_equity, rel_tol=1e-11)
    return dict(**breakdown(trades), closed=len(done), open=len(trades)-len(done), checks=dict(counts),
                gate=dict(gate), flat_reasons=dict(flat_reasons), gross_return=gross_equity-1, net_return=equity-1,
                cost_5bp_return=gross_equity*(1-.0005)**len(done)/(1+.0005)**len(trades)-1,
                drawdown_at_5m_closes=worst_dd, by_symbol={s:breakdown([t for t in trades if t["symbol"]==s]) for s in SYMBOLS},
                daily=daily), trades, checks


def main():
    times, days = date_grid()
    tape, inputs = load_bars(times)
    cases, ledger, decisions = {}, {}, {}
    for n in WINDOWS:
        features = signals(times, tape, n)
        for label, path, instant in [("main","OLHC",False), ("high_first","OHLC",False), ("instant_reference","OLHC",True)]:
            result, trades, checks = replay(times, days, tape, features, path, instant, label=="main")
            cases.setdefault(label, {})[str(n)] = result
            if label=="main":
                ledger[str(n)], decisions[str(n)] = trades, checks
    dump("results.json", dict(dates=list(dict.fromkeys(days)), inputs=inputs, cases=cases, fee_per_side=COST))
    dump("ledger.json", dict(trades=ledger))
    with gzip.open(HERE / "decisions.json.gz","wt",encoding="utf-8") as f:
        json.dump(decisions, f, separators=(",",":"), allow_nan=False)
    for label, variants in cases.items():
        print(label)
        for n, stat in variants.items():
            print(n, {k:v for k,v in stat.items() if k not in {"daily","by_symbol"}})


if __name__ == "__main__":
    main()
