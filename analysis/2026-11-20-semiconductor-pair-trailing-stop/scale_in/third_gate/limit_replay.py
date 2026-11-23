"""One-bar buy limit at the close available when the paper third stage was decided.

The original, cash-constrained paper account must actually execute stage three.
At that bar's open the real simulation can place one order, expiring that bar.
Opening improvement is credited. Intrabar touches fill exactly at the limit.
Only the price path following the actual fill can activate its bracket.
"""
from collections import Counter
from datetime import timedelta
from math import fsum, isclose, isfinite
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import EPS, STEP, LiveLot, process_price


def vertices(bar, path):
    _, high, low, close = bar
    if path == "OLHC":
        return (("L", low), ("H", high), ("C", close))
    return (("H", high), ("L", low), ("C", close))


def locate_fill(opening, vertices_after_open, limit):
    """Return the fill, its leg, and the remaining vertices (including leg end)."""
    if opening <= limit:
        return opening, "O", vertices_after_open
    previous = "O"
    for k, (name, px) in enumerate(vertices_after_open):
        if px <= limit:
            return limit, previous + "-" + name, vertices_after_open[k:]
        previous = name
    return None


def run(ctx, path, lots, notional=600, fee=2.50, initial=1000):
    if path not in ("OLHC", "OHLC"):
        raise ValueError("Expected OLHC or OHLC")
    if not all(isfinite(x) for x in (notional, fee, initial)) or min(notional, initial) <= 0 or fee < 0:
        raise ValueError("Expected a positive trade amount and initial cash and a nonnegative fee")
    data = ctx.data
    tape, times, days = data["tape"], data["times"], data["days"]
    if not times or len(times) != len(days) or any(len(x) != len(times) for x in tape.values()):
        raise ValueError("Expected a shared nonempty grid")
    orders = {}
    for lot in lots:
        if lot["stage"] != 3:
            continue
        entry, decision = lot["entry_i"], lot["decision_i"]
        if not (0 < decision < entry < len(times) and entry == decision + 1
                and times[decision - 1] + STEP == times[decision] and times[decision] + STEP == times[entry]):
            raise ValueError("Stage-three decision needs a contiguous completed bar and entry bar")
        if entry in orders:
            raise ValueError("Overlapping third-stage signals at the same open")
        orders[entry] = lot["symbol"], lot["cycle"], decision

    cash = equity = equity_peak = float(initial)
    held = None
    maximum_drawdown = 0.
    events, trades, daily = [], [], {}

    def cross(i, px, at_open):
        nonlocal cash, held
        signal = process_price(held, px, at_open=at_open)
        if signal is None:
            return
        why, price = signal
        trade = held.record
        gross = trade["qty"] * (price - trade["entry_price"])
        trade.update(exit_i=i, end=times[i].isoformat(), exit_bar_end=(times[i]+STEP).isoformat(),
                     exit_at_open=at_open, exit_price=price, reason=why,
                     gross=gross, fees=2*fee, net=gross-2*fee)
        cash += trade["qty"] * price - fee
        held = None

    for i, now in enumerate(times):
        remaining = ()
        if held is not None:
            bar = tape[held.record["symbol"]][i]
            cross(i, bar[0], True)
            if held is not None:
                remaining = vertices(bar, path)

        if i in orders:
            symbol, cycle, d = orders[i]
            limit = tape[symbol][d - 1][3]
            bar = tape[symbol][i]
            event = dict(symbol=symbol, paper_cycle=cycle, decision_i=d,
                         decision=(times[d]+timedelta(seconds=1)).isoformat(),
                         reference_bar_i=d-1, limit_price=limit, entry_i=i,
                         start=now.isoformat(), expires=(now+STEP).isoformat(),
                         session_date=days[i], original_open=bar[0], cash_at_open=cash,
                         trade_index=None)
            events.append(event)
            if held is not None:
                event["status"] = "occupied"
            elif cash + EPS < notional + fee:
                event["status"] = "cash_blocked"
            else:
                match = locate_fill(bar[0], vertices(bar, path), limit)
                if match is None:
                    event["status"] = "unfilled"
                else:
                    price, leg, remaining = match
                    event.update(status="filled", trade_index=len(trades), fill_price=price, fill_leg=leg)
                    trade = dict(entry_i=i, start=now.isoformat(), entry_bar_end=(now+STEP).isoformat(),
                                 symbol=symbol, paper_cycle=cycle, entry_price=price, fill_leg=leg,
                                 limit_price=limit, original_open=bar[0], qty=notional/price,
                                 exit_i=None, reason="open", fees=fee)
                    trades.append(trade)
                    cash -= notional + fee
                    held = LiveLot(trade, price, price*1.08)
        for _, price in remaining:
            cross(i, price, False)
            if held is None:
                break
        equity = cash
        if held is not None:
            t = held.record
            equity += t["qty"]*tape[t["symbol"]][i][3]
        equity_peak = max(equity_peak, equity)
        maximum_drawdown = max(maximum_drawdown, 1-equity/equity_peak)
        if i + 1 == len(times) or days[i+1] != days[i]:
            daily[days[i]] = dict(end_equity=equity, open=int(held is not None))

    if held is not None:
        t = held.record
        mark = tape[t["symbol"]][-1][3]
        gross = t["qty"]*(mark-t["entry_price"])
        t.update(mark_price=mark, mark_at=(times[-1]+STEP).isoformat(), gross=gross, net=gross-fee)

    tally = Counter(e["status"] for e in events)
    closed = [t for t in trades if t["exit_i"] is not None]
    gross = fsum(t["gross"] for t in trades)
    fees = fee * (len(trades) + len(closed))
    assert sum(tally.values()) == len(orders) and tally["filled"] == len(trades)
    assert isclose(equity-initial, gross-fees, abs_tol=1e-8)
    assert isclose(gross-fees, fsum(t["net"] for t in trades), abs_tol=1e-8)
    return dict(triggers=len(orders), buys=len(trades), sells=len(closed), open=int(held is not None),
                wins=sum(t["net"] > EPS for t in closed), losses=sum(t["net"] < -EPS for t in closed),
                targets=sum(t["reason"] == "target" for t in closed), cash_blocked=tally["cash_blocked"],
                occupied=tally["occupied"], unfilled=tally["unfilled"], gross=gross, fees=fees,
                net=gross-fees, end_equity=equity, max_drawdown=maximum_drawdown,
                daily=daily, trades=trades, events=events)
