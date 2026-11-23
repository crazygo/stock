"""Independent price-path and cash-account checks for fixed-fee pyramiding."""
from collections import defaultdict
from decimal import Decimal
from math import exp, floor, isclose
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "entry_filters"))
import shadow_score as shadow

TRAIL = Decimal("0.97")
TARGET = Decimal("1.08")


def close_enough(a, b):
    assert isclose(float(a), float(b), rel_tol=2e-10, abs_tol=2e-8), (a, b)


def outcome(bars, entry, path):
    """Trace one lot with exact decimal trigger levels, without consulting other lots."""
    peak = bars[entry][0]
    target = peak * TARGET
    for idx in range(entry, len(bars)):
        op, hi, lo, close = bars[idx]
        points = (op, lo, hi, close) if path == "OLHC" else (op, hi, lo, close)
        for segment, price in enumerate(points):
            if price <= peak * TRAIL:
                return idx, float(price if segment == 0 else peak * TRAIL), "stop", segment == 0
            if price >= target:
                return idx, float(price if segment == 0 else target), "target", segment == 0
            peak = max(peak, price)
    return None, float(bars[-1][3]), "open", False


def old_reference(data):
    """Value original q50 signals at constant notional with unrestricted capital."""
    saved = json.loads((HERE.parent/"entry_filters/shadow_freeze.json").read_text())
    gate = saved["thresholds"][saved["selected"]]
    reference, seed = {}, {}
    for path in ("OLHC", "OHLC"):
        _, rows, features = shadow.original(data, path)
        scores = shadow.portable_scores(features, saved["model"])
        kept = [t for t, s in zip(rows, scores) if s is not None and s >= gate]
        seed[path] = {key:{data["index"][t["signal_end"]]:t["symbol"] for t,s in zip(rows,scores)
                           if s is not None and s >= limit}
                      for key,limit in saved["thresholds"].items()}
        closed = sum(t["exit_reason"] != "open" for t in kept)
        gross_sum = sum(t["gross"] for t in kept)
        ref = dict(buys=len(kept), sells=closed, sum_gross_returns=gross_sum)
        for fee in (2.5, 1.25):
            fees = fee*(len(kept)+closed)
            ref[str(fee)] = dict(fees=fees, break_even_notional=fees/gross_sum if gross_sum>0 else None,
                                 net_pnl={str(a):a*gross_sum-fees for a in (100,200,300,600)})
        reference[path] = ref
    return reference,seed


def verify_run(data, bars, path, lots, summary, amounts, wait_bars, fee, whole, initial,
               seed, confirm=None, cutoffs=None):
    times = data["times"]
    cash = peak_equity = initial
    active = {}
    opening, closing, intra = defaultdict(list), defaultdict(list), defaultdict(list)
    cycles, exits = defaultdict(list), {}
    gross = fees = net = dd = 0.
    examples = {}
    for j, lot in enumerate(lots):
        symbol, e, d, stage = lot["symbol"], lot["entry_i"], lot["decision_i"], lot["stage"]
        assert e==d+1 and times[e]-times[d] == shadow.base.STEP
        assert 1<=stage<=3 and len(amounts)>=stage
        notional = amounts[stage-1]
        px = float(bars[symbol][e][0])
        close_enough(px,lot["entry_price"])
        qty = floor(notional/px) if whole else notional/px
        close_enough(lot["qty"],qty)
        close_enough(lot["nominal"],notional)
        close_enough(lot["value"],qty*px)
        assert qty>0
        ex, price, reason, gap = outcome(bars[symbol],e,path)
        assert (ex, reason) == (lot["exit_i"], lot["reason"]), (lot, ex, price, reason)
        # An unclosed position is marked but incurs no hypothetical sell fee.
        if ex is not None:
            close_enough(lot["exit_price"], price)
        lot_gross = qty*(price-px)
        lot_fee = fee*(1+int(ex is not None))
        close_enough(lot_fee, lot["fees"])
        close_enough(lot_gross, lot["gross_pnl"])
        close_enough(lot_gross-lot_fee, lot["net_pnl"])
        gross += lot_gross
        fees += lot_fee
        net += lot_gross-lot_fee
        group = cycles[lot["cycle"]]
        assert stage == len(group)+1
        if stage == 1:
            assert seed[d] == symbol
        else:
            seed_lot, prev = lots[group[0]], lots[group[-1]]
            assert seed_lot["symbol"] == symbol and d-prev["entry_i"] >= wait_bars[stage-2]
            first_exit, _, first_gap = exits[group[0]]
            assert first_exit is None or first_exit>e or (first_exit==e and not first_gap)
            if confirm:
                assert confirm(symbol,seed_lot["decision_i"],prev["decision_i"],d,stage,cutoffs)
        group.append(j)
        exits[j] = (ex,price,gap)
        opening[e].append(j)
        if ex is not None:
            (closing if gap else intra)[ex].append(j)

    daily = {}
    pending = None
    partial_exits = 0
    max_simultaneous = max_deployed = 0
    for idx, clock in enumerate(times):
        def sell(j, phase):
            nonlocal cash, partial_exits
            lot = active.pop(j)
            _, price, _ = exits[j]
            cash += lot["qty"]*price-fee
            survivors = [v for key,v in active.items() if exits[key][0] is None or exits[key][0]>idx]
            if survivors:
                partial_exits += 1
                examples.setdefault("partial_exit", dict(symbol=lot["symbol"],stage=lot["stage"],
                                    clock=clock.isoformat(),phase=phase, price=price,
                                    remaining_stages=[t["stage"] for t in survivors]))

        for j in closing[idx]:
            sell(j,"open")
        if pending is None:
            assert not opening[idx]
        else:
            stage, symbol, first_id = pending
            blocked = stage > 1 and first_id not in active
            op = float(bars[symbol][idx][0])
            qty = floor(amounts[stage-1]/op) if whole else amounts[stage-1]/op
            can_buy = not blocked and qty > 0 and cash+1e-10 >= qty*op+fee
            assert len(opening[idx]) == int(can_buy), (clock,pending,can_buy,opening[idx])
            if can_buy:
                assert (lots[opening[idx][0]]["stage"],lots[opening[idx][0]]["symbol"]) == (stage,symbol)
        pending = None
        for j in opening[idx]:
            lot = lots[j]
            assert all(prev["symbol"]==lot["symbol"] and prev["cycle"]==lot["cycle"]
                       for prev in active.values())
            assert (not active) if lot["stage"]==1 else bool(active)
            cash -= lot["qty"]*lot["entry_price"] + fee
            assert cash >= -2e-8, (clock,cash,lot)
            active[j] = lot
            max_simultaneous = max(max_simultaneous,len(active))
            max_deployed = max(max_deployed,sum(l["value"] for l in active.values()))
            assert len(active)<=3
        if idx+1 < len(times) and times[idx+1]-times[idx] == shadow.base.STEP:
            if not active and idx in seed:
                pending = (1,seed[idx],None)
            elif active:
                cycle_id = next(iter(active.values()))["cycle"]
                bought = [j for j in cycles[cycle_id] if lots[j]["entry_i"]<=idx]
                first_id,last_id = bought[0],bought[-1]
                stage = len(bought)+1
                if (first_id in active and stage<=len(amounts)
                        and idx-lots[last_id]["entry_i"] >= wait_bars[stage-2]
                        and confirm(lots[first_id]["symbol"],lots[first_id]["decision_i"],
                                    lots[last_id]["decision_i"],idx,stage,cutoffs)):
                    pending = (stage,lots[first_id]["symbol"],first_id)
        for j in intra[idx]:
            sell(j,"intrabar")
        equity = cash+sum(v["qty"] * float(bars[v["symbol"]][idx][3]) for v in active.values())
        peak_equity = max(peak_equity,equity)
        dd = max(dd,1-equity/peak_equity)
        daily[data["days"][idx]]=equity
    close_enough(equity,initial+net)
    close_enough(gross-fees,net)
    for key, val in dict(gross_pnl=gross, fees=fees, net_pnl=net, max_drawdown=dd, max_deployed=max_deployed, ending_equity=equity, ending_cash=cash).items():
        close_enough(summary[key],val)
    for k,v in dict(buys=len(lots),sells=len(lots)-len(active),open=len(active)).items():
        assert summary[k]==v
    completed = [[lots[j] for j in group] for group in cycles.values() if all(lots[j]["exit_i"] is not None for j in group)]
    totals = [sum(l["net_pnl"] for l in group) for group in completed]
    reconstructed = dict(count=len(cycles), closed=len(completed), open=len(cycles)-len(completed),
                         wins=sum(x>1e-10 for x in totals), losses=sum(x < -1e-10 for x in totals))
    for k,v in reconstructed.items():
        assert summary["cycles"][k] == v
    stages = {str(stage):sum(l["stage"]==stage for l in lots) for stage in (1,2,3)}
    return dict(verified_lots=len(lots), stages=stages, gross=gross, fees=fees, net=net,
                max_endbar_drawdown=dd, max_simultaneous=max_simultaneous,
                partial_exits_with_other_lots_held_after_bar=partial_exits, ending_cash=cash, open=len(active),
                ending_equity=equity,daily_equity=daily, examples=examples)


def confirm_fn(data, frozen):
    def trend(bars):
        high = bars[0][1]
        for _, hi, lo, _ in bars:
            high = max(high,hi)
            if lo <= .97*high+1e-10:
                return False
        return bars[-1][3] > bars[0][0]

    def confirm(sym, first, last, i, stage, cuts):
        start = first-5
        if start<0 or data["times"][i] - data["times"][start] != (i-start)*shadow.base.STEP:
            return False
        tape = data["tape"][sym]
        if not (trend(tape[start:i]) and trend(tape[i-5:i]) and tape[i-1][3] > tape[last-1][3]):
            return False
        x = shadow.feature_at(data["tape"], data["times"], i, sym)
        model = frozen["model"]
        z = sum((value-m)/sd*w for value,m,sd,w in zip(x,model["mean"],model["scale"],model["weights"]))
        score = 1/(1+exp(-z-model["intercept"]))
        return score >= cuts[stage-2]
    return confirm


def audit():
    data = shadow.load("november")
    bars = {s:[tuple(Decimal(str(x)) for x in row) for row in side] for s,side in data["tape"].items()}
    frozen = json.loads((HERE.parent/"entry_filters/shadow_freeze.json").read_text())
    reference,seed = old_reference(data)
    ledger = json.loads((HERE/"ledger.json").read_text())
    assert ledger["block"]=="november"
    check = {}
    for name, case in ledger["runs"].items():
        cfg = case["config"]
        first = frozen["thresholds"][cfg["seed"]]
        cuts = [first,first] if cfg["mode"]=="hold" else [max(first,frozen["thresholds"][q]) for q in ("q75","q90")]
        path = case["path"]
        v = verify_run(data,bars,path,case["lots"],case["summary"],cfg["amounts"],cfg["wait"],
                       case["fee"],case["whole"],ledger["initial"],seed[path][cfg["seed"]],confirm_fn(data,frozen),cuts)
        for day, eq in v["daily_equity"].items():
            close_enough(eq,case["daily"][day]["equity"])
        check[name] = v
    report = dict(verified_tape_rows=len(data["times"]), actual_inputs=data["sources"],
                  starts=reference, checked_runs=check,
                  isolated_lot_economics={str(fee):{
                      str(a):dict(break_even_move=2*fee/a, net_at_target=a*.08-2*fee,
                                  net_at_initial_stop=-a*.03-2*fee)
                      for a in (100,200,300,600)} for fee in (2.50,1.25)})
    (HERE/"audit.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("Verified", len(check),"scenario ledgers",sum(c["verified_lots"] for c in check.values()),"lot records")


if __name__=="__main__":
    audit()
