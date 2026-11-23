"""Independently reconstruct the third-stage decision, limit fill, exit, cash and P&L."""
import argparse
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from audit import close_enough, confirm_fn, verify_run
from paper_probe import VIRTUAL, FEE, INITIAL
from study_scale import Context, LEVELS, MODEL, PATHS, score

USD = Decimal("600")
F = Decimal(str(FEE))
START = Decimal(str(INITIAL))
D = lambda n: Decimal(str(n))
NAMES = ("baseline", "breakout", "progress", "both", "limit")


def fill_and_exit(bars, i, path, limit=None):
    """Exact decimal point traversal. The fill leg can begin mid-segment."""
    op, hi, lo, close = bars[i]
    traversal = (op, lo, hi, close) if path == "OLHC" else (op, hi, lo, close)
    if limit is None or op <= limit:
        buy = op
        remaining = traversal[1:]
    else:
        if lo > limit:
            return None
        buy = limit
        remaining = traversal[traversal.index(lo):]
    high_water = buy
    target = buy * D("1.08")
    for j in range(i, len(bars)):
        if j == i:
            vertices = remaining
        else:
            op, hi, lo, close = bars[j]
            vertices = (op, lo, hi, close) if path == "OLHC" else (op, hi, lo, close)
        for k, px in enumerate(vertices):
            at_open = j != i and k == 0
            stop = high_water * D("0.97")
            if px <= stop:
                return buy, j, px if at_open else stop, "stop", at_open
            if px >= target:
                return buy, j, px if at_open else target, "target", at_open
            high_water = max(px, high_water)
    return buy, None, bars[-1][3], "open", False


def synthetic_boundaries():
    x = [[D(x) for x in row] for row in [(103,110,99,103),(102,103,95,99)]]
    # In OHLC, 110 occurred before the 100 limit was touched. On the low-first path
    # the same high is after entry and crosses the new lot's target at 108.
    assert fill_and_exit(x, 0, "OHLC", D(100)) == (D(100), 1, D("99.91"), "stop", False)
    assert fill_and_exit(x, 0, "OLHC", D(100)) == (D(100), 0, D(108), "target", False)
    assert fill_and_exit(x, 0, "OLHC", D(98)) is None
    assert fill_and_exit(x, 0, "OLHC", D(106))[0] == D(103)
    assert (D("121") - D("120")) * USD == 2*F*D("120")


def source_events(ctx, path):
    d = ctx.data
    bars = {s:[tuple(map(D, row)) for row in side] for s, side in d["tape"].items()}
    v = ctx.execute(VIRTUAL, path, FEE)
    audit = verify_run(d, bars, path, v["lots"], v["summary"], VIRTUAL.amounts,
                       VIRTUAL.wait, FEE, False, INITIAL, ctx.seed[path, "q50"],
                       confirm_fn(d, MODEL), (LEVELS["q75"], LEVELS["q90"]))
    second, events = {}, []
    for t in v["lots"]:
        cycle = t["cycle"]
        if t["stage"] == 2:
            second[cycle] = t
        if t["stage"] != 3:
            continue
        idx, i, symbol = t["decision_i"], t["entry_i"], t["symbol"]
        previous = second[cycle]
        assert previous["entry_i"] <= idx-1 and previous["symbol"] == symbol
        close = bars[symbol][idx-1][3]
        earlier_high = bars[symbol][idx-2][1]
        price2 = bars[symbol][previous["entry_i"]][0]
        events.append(dict(entry_i=i, decision_i=idx, cycle=cycle, symbol=symbol,
                           close=close, prior_high=earlier_high, second_entry=price2,
                           breakout=close > earlier_high, progress=(close-price2)*USD >= 2*F*price2))
    return bars, events, audit["verified_lots"]


def ledger(bars, days, events, path, rule):
    allowed = [e for e in events if rule in ("baseline","limit")
               or (e["breakout"] if rule=="breakout" else e["progress"] if rule=="progress"
                   else e["breakout"] and e["progress"])]
    by_entry = {e["entry_i"]:e for e in allowed}
    n = len(days)
    cash = top = START
    max_dd = D(0)
    active = None
    rows, daily, status = [], {}, {}
    def close_position(r):
        return r["qty"]*r["exit_price"]-F
    for i in range(n):
        if active and active["exit_i"] == i and active["exit_at_open"]:
            cash += close_position(active)
            active = None
        if i in by_entry:
            ev = by_entry[i]
            if active:
                status[i] = "occupied"
            elif cash < USD + F:
                status[i] = "cash_blocked"
            else:
                hit = fill_and_exit(bars[ev["symbol"]], i, path, ev["close"] if rule=="limit" else None)
                if hit is None:
                    status[i] = "unfilled"
                else:
                    status[i] = "filled"
                    buy, exit_i, price, reason, at_open = hit
                    qty = USD/buy
                    gross = qty*(price-buy)
                    active = dict(symbol=ev["symbol"], entry_i=i, entry_price=buy, qty=qty,
                                  paper_cycle=ev["cycle"], exit_i=exit_i, exit_price=price,
                                  exit_at_open=at_open, reason=reason,
                                  gross=gross, net=gross-F*(1+int(exit_i is not None)))
                    cash -= USD+F
                    rows.append(active)
        if active and active["exit_i"] == i:
            cash += close_position(active)
            active = None
        equity = cash + (active["qty"] * bars[active["symbol"]][i][3] if active else 0)
        top = max(top,equity)
        max_dd = max(max_dd, 1-equity/top)
        daily[days[i]] = dict(end_equity=equity, open=int(active is not None))
    closed = [t for t in rows if t["exit_i"] is not None]
    net = sum((t["net"] for t in rows),D(0))
    close_enough(cash + (active["qty"]*active["exit_price"] if active else 0), START+net)
    skips = Counter(status.values())
    summary = dict(triggers=len(allowed), buys=len(rows), sells=len(closed), open=len(rows)-len(closed),
                   wins=sum(t["net"]>0 for t in closed), losses=sum(t["net"]<0 for t in closed),
                   targets=sum(t["reason"]=="target" for t in closed), gross=sum(t["gross"] for t in rows),
                   net=net, fees=(len(rows)+len(closed))*F, end_equity=START+net,
                   max_drawdown=max_dd, cash_blocked=skips["cash_blocked"])
    return summary, rows, daily, status


def compare(summary, rows, daily, account):
    for k,v in summary.items():
        close_enough(account[k], v)
    assert set(daily) == set(account["daily"])
    for day, rec in daily.items():
        for k, v in rec.items():
            close_enough(account["daily"][day][k],v)
    for ours, theirs in zip(rows,account["trades"],strict=True):
        for k in ("symbol","entry_i","exit_i","reason","paper_cycle"):
            assert ours[k] == theirs[k], (k,ours[k],theirs[k])
        for k in ("entry_price", "qty","gross","net"):
            close_enough(ours[k],theirs[k])
        px = theirs["mark_price"] if ours["exit_i"] is None else theirs["exit_price"]
        close_enough(ours["exit_price"],px)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE/"results.json")
    parser.add_argument("--archived-only", action="store_true")
    opt = parser.parse_args()
    synthetic_boundaries()
    old = json.loads((HERE.parent/"paper_probe.json").read_text())["blocks"]
    results = {} if opt.archived_only else json.loads(opt.results.read_text())["cases"]
    coverage = json.loads((HERE/"coverage.json").read_text())["blocks"]
    registered = {b["name"]:(b["start"],b["end"]) for group in coverage.values() for b in group}
    old_by_dates = {tuple(score.study.BLOCKS[k]):k for k in old}
    score.study.BLOCKS.update(registered)
    names = old if opt.archived_only else results
    if not opt.archived_only:
        assert set(names) == set(registered), (names.keys(), registered.keys())
    for name in names:
        ctx = Context(name)
        for path in PATHS:
            bars, events, paper_count = source_events(ctx, path)
            cases = results.get(name,{}).get(path)
            if cases:
                for ours,theirs in zip(events,cases["events"],strict=True):
                    for k in ("symbol","decision_i","entry_i","cycle","breakout","progress"):
                        assert ours[k] == theirs[k], (name,path,k,ours[k],theirs[k])
                    for k in ("close","prior_high","second_entry"):
                        close_enough(ours[k],theirs[k])
            counts = []
            for rule in NAMES if cases else ("baseline",):
                summary, rows, daily, status = ledger(bars, ctx.data["days"], events, path, rule)
                archive = old_by_dates.get(tuple(score.study.BLOCKS[name]))
                if archive is not None and rule=="baseline":
                    compare(summary,rows,daily,old[archive][path]["real"]["stage3_600"])
                if cases:
                    actual = cases["accounts"][rule]
                    compare(summary,rows,daily,actual)
                    if rule == "limit":
                        assert {e["entry_i"]:e["status"] for e in actual["events"]} == status
                        indexed = {e["entry_i"]:e for e in events}
                        for event in actual["events"]:
                            close_enough(event["limit_price"], indexed[event["entry_i"]]["close"])
                            assert event["reference_bar_i"] == event["entry_i"]-2
                counts.append((rule,len(rows),float(summary["net"])))
            print(name,path,"paper",paper_count,"third signals",len(events),counts)


if __name__ == "__main__":
    main()
