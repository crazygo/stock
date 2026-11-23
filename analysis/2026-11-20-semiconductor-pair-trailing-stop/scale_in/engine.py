"""A cash ledger for staged, same-symbol additions with per-lot bracket exits."""
from dataclasses import dataclass
from datetime import timedelta
from math import floor, fsum, isclose, isfinite
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import EPS, STEP, process_price


@dataclass
class LiveLot:
    record: dict
    peak: float
    target: float


def breakdown(rows):
    closed = [r for r in rows if r["exit_i"] is not None]
    wins = sum(r["net_pnl"] > EPS for r in closed)
    losses = sum(r["net_pnl"] < -EPS for r in closed)
    return {
        "buys": len(rows), "sells": len(closed), "open": len(rows)-len(closed),
        "wins": wins, "losses": losses, "flat": len(closed)-wins-losses,
        "win_rate": wins/len(closed) if closed else None,
        "targets": sum(r["reason"] == "target" for r in closed),
        "stops": sum(r["reason"] == "stop" for r in closed),
        "capital_bought": fsum(r["value"] for r in rows),
        "gross_pnl": fsum(r["gross_pnl"] for r in rows),
        "fees": fsum(r["fees"] for r in rows),
        "net_pnl": fsum(r["net_pnl"] for r in rows),
    }


def cycle_breakdown(groups):
    closed = [g for g in groups if all(t["exit_i"] is not None for t in g)]
    realized = [fsum(t["net_pnl"] for t in g) for g in closed]
    wins, losses = sum(x > EPS for x in realized), sum(x < -EPS for x in realized)
    return dict(count=len(groups), closed=len(closed), open=len(groups)-len(closed),
                wins=wins, losses=losses, flat=len(closed)-wins-losses,
                win_rate=wins/len(closed) if closed else None)


def run(tape, times, days, seed, confirm, amounts, wait_bars, cutoffs, path, fee,
        initial=1000, whole=False):
    """Buy at next contiguous open. The passed seed and confirm use decision indices.

    Fee is USD per executed order; amounts are securities notionals exclusive of fee.
    Additions require the seed lot to survive both the decision and the fill open.
    PnL includes marked open lots; win/loss counts include closed lots only.
    """
    if not 1 <= len(amounts) <= 3 or any(not isfinite(x) or x <= 0 for x in amounts):
        raise ValueError("Supply one to three positive purchase notionals")
    if path not in ("OLHC", "OHLC") or fee < 0 or initial <= 0:
        raise ValueError("Invalid price path, fee or initial cash")
    if not len(times) or len(days) != len(times) or any(len(v) != len(times) for v in tape.values()):
        raise ValueError("The OHLC tape, day labels and times must share a nonempty grid")
    delays = [wait_bars] * (len(amounts)-1) if isinstance(wait_bars, int) else list(wait_bars)
    if len(delays) < len(amounts)-1 or any(not isinstance(n,int) or n < 0 for n in delays):
        raise ValueError("Expected a nonnegative bar count for every addition")

    cash = prior_day_equity = peak_equity = float(initial)
    max_drawdown = 0.
    max_deployed = 0.
    cash_blocked = qty_blocked = canceled_additions = 0
    missed_seeds = day_buys = day_sells = 0
    lots, live, groups, daily = [], [], [], {}
    active = []
    pending = None

    def cross(index, price, opening):
        nonlocal cash, day_sells
        for held in live[:]:
            event = process_price(held, price, at_open=opening)
            if event:
                lot = held.record
                why, fill = event
                qty = lot["qty"]
                cash += qty * fill - fee
                gross = qty * (fill-lot["entry_price"])
                lot.update(exit_i=index, exit_bar_start=times[index].isoformat(),
                           exit_bar_end=(times[index]+STEP).isoformat(), exit_day=days[index],
                           exit_at_open=opening, exit_price=fill, reason=why,
                           fees=2*fee, gross_pnl=gross, net_pnl=gross-2*fee)
                live.remove(held)
                day_sells += 1

    for i, clock in enumerate(times):
        if live:
            symbol = active[0]["symbol"]
            cross(i, tape[symbol][i][0], opening=True)
        if not live:
            active = []

        if pending is not None:
            symbol, decision_i, stage = pending
            pending = None
            assert i == decision_i+1 and times[decision_i]+STEP == clock
            if stage > 1 and (not active or active[0]["exit_i"] is not None):
                canceled_additions += 1
            else:
                assert (stage == 1 and not live) or (len(active)+1 == stage and active[0]["symbol"] == symbol)
                price = tape[symbol][i][0]
                amount = amounts[stage-1]
                qty = floor(amount/price) if whole else amount/price
                value = qty * price
                if qty <= 0:
                    qty_blocked += 1
                elif cash+EPS < value+fee:
                    cash_blocked += 1
                else:
                    cash -= value+fee
                    if stage == 1:
                        active = []
                        groups.append(active)
                    lot = dict(stage=stage, cycle=len(groups), symbol=symbol,
                               decision_i=decision_i, decision=(times[decision_i]+timedelta(seconds=1)).isoformat(),
                               entry_i=i, entry=clock.isoformat(), entry_day=days[i], entry_price=price,
                               qty=qty, nominal=amount, value=value, exit_i=None, exit_price=None,
                               reason="open", fees=fee)
                    lots.append(lot)
                    active.append(lot)
                    live.append(LiveLot(lot, price, 1.08*price))
                    day_buys += 1
                    max_deployed = max(max_deployed, sum(h.record["value"] for h in live))

        next_open = i+1 < len(times) and clock+STEP == times[i+1]
        if next_open:
            if not live:
                if i in seed:
                    pending = (seed[i], i, 1)
            else:
                missed_seeds += int(i in seed)
                first, last = active[0], active[-1]
                stage = len(active)+1
                if (stage <= len(amounts) and first["exit_i"] is None
                        and i-last["entry_i"] >= delays[stage-2]
                        and confirm(first["symbol"], first["decision_i"], last["decision_i"], i, stage, cutoffs)):
                    pending = (first["symbol"], i, stage)

        if live:
            symbol = active[0]["symbol"]
            _, high, low, close = tape[symbol][i]
            for price in ((low, high, close) if path == "OLHC" else (high, low, close)):
                cross(i, price, opening=False)
            equity = cash + fsum(h.record["qty"] * close for h in live)
        else:
            equity = cash
        peak_equity = max(peak_equity, equity)
        max_drawdown = max(max_drawdown, 1-equity/peak_equity)
        if i+1 == len(times) or days[i+1] != days[i]:
            daily[days[i]] = dict(equity=equity, net_pnl=equity-prior_day_equity, cash=cash,
                                  buys=day_buys, sells=day_sells, open=len(live))
            prior_day_equity, day_buys, day_sells = equity, 0, 0

    assert pending is None
    for h in live:
        lot = h.record
        mark = tape[lot["symbol"]][-1][3]
        gross = lot["qty"] * (mark-lot["entry_price"])
        lot.update(mark_price=mark, mark_at=(times[-1]+STEP).isoformat(),
                   gross_pnl=gross, net_pnl=gross-fee)
    summary = breakdown(lots)
    summary.update(initial=initial, ending_cash=cash, ending_equity=equity,
                   max_drawdown=max_drawdown, max_deployed=max_deployed,
                   cash_blocked=cash_blocked, zero_qty_blocked=qty_blocked,
                   canceled_additions=canceled_additions, occupied_seed_signals=missed_seeds,
                   cycles=cycle_breakdown(groups),
                   by_stage={str(n):breakdown([t for t in lots if t["stage"] == n]) for n in (1,2,3)})
    assert isclose(summary["net_pnl"], equity-initial, abs_tol=1e-7)
    assert isclose(summary["fees"], (summary["sells"]+len(lots))*fee, abs_tol=1e-7)
    assert isclose(fsum(d["net_pnl"] for d in daily.values()), equity-initial, abs_tol=1e-7)
    assert all(len(g) <= len(amounts) and len({t["symbol"] for t in g}) == 1 for g in groups)
    return {"summary":summary, "lots":lots, "daily":daily}
