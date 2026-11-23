"""Reprice the verified third-stage fills, retaining fixed per-fill fees and checking cash."""
import json
from math import ceil, isclose
from pathlib import Path

HERE = Path(__file__).resolve().parent
FEE = 2.50
OLD = 600.
SIZE = 1000.
START = 1400.  # Same $400 cash buffer above the order's notional as the $600 study.
RULES = {"all": 0, "original_gate": 2*FEE/OLD, "scaled_gate": 2*FEE/SIZE}


def read(name):
    return json.loads((HERE/name).read_text())


def recent_rows(case):
    account = case["accounts"]["baseline"]
    assert account["cash_blocked"] == 0
    rows = []
    for t, e in zip(account["trades"], case["events"], strict=True):
        assert t["entry_i"] == e["entry_i"]
        end = t["mark_price"] if t["reason"] == "open" else t["exit_price"]
        move = end / t["entry_price"] - 1
        assert isclose(OLD*move, t["gross"], abs_tol=1e-8)
        rows.append((t["start"], t["symbol"], e["close"]/e["second_entry"]-1, move, t["reason"]))
    return rows


def earlier_rows(block):
    assert block["cash_blocked"]["baseline"] == 0 and len(block["opportunities"]) == block["paper_thirds"]
    rows = []
    for t in block["opportunities"]:
        fees = FEE if t["reason"] == "open" else 2*FEE
        rows.append((t["entry"], t["symbol"], t["prev_close"]/t["paper_second_fill"]-1,
                     (t["net"]+fees)/OLD, t["reason"]))
    return rows


def account(rows, rule, amount, initial, cap_to_cash=False):
    cash, min_required, cumulative_net = initial, 0., 0.
    trades = []
    blocked = 0
    for entry, symbol, progress, move, reason in rows:
        if rule != "all" and progress < RULES[rule]-1e-12:
            continue
        # Original stage-three lots never overlap. An open lot must end the block.
        assert not trades or trades[-1]["reason"] != "open"
        min_required = max(min_required, amount+FEE-cumulative_net)
        cumulative_net += amount*move-FEE*(1+int(reason != "open"))
        size = min(amount, cash-FEE) if cap_to_cash else amount
        if size <= 0 or cash+1e-9 < size+FEE:
            blocked += 1
            continue
        fees = FEE*(1+int(reason != "open"))
        gross = size*move
        net = gross-fees
        cash += net if reason != "open" else -(size+FEE)
        trades.append(dict(entry=entry,symbol=symbol,reason=reason,notional=size,gross=gross,fees=fees,net=net))
    net = sum(t["net"] for t in trades)
    marked = sum(t["notional"]+t["gross"] for t in trades if t["reason"] == "open")
    assert isclose(cash+marked-initial,net,abs_tol=1e-8)
    closed = [t for t in trades if t["reason"] != "open"]
    return dict(buys=len(trades), wins=sum(t["net"] > 1e-9 for t in closed),
                losses=sum(t["net"] < -1e-9 for t in closed), open=len(trades)-len(closed),
                fees=sum(t["fees"] for t in trades), gross=sum(t["gross"] for t in trades), net=net,
                cash_blocked=blocked, min_initial_full_size=ceil(min_required*100-1e-8)/100,
                trades=trades)


def main():
    later = read("results.json")
    blocks = read("coverage.json")["blocks"]
    earlier = read("older_transfer.json")
    out = {}
    for path in ("OLHC","OHLC"):
        groups = {period:[recent_rows(later["cases"][b["name"]][path]) for b in group]
                  for period,group in blocks.items()}
        groups["march_july"] = [earlier_rows(b) for b in earlier["results"][path]["blocks"].values()]
        out[path] = {}
        for period, group in groups.items():
            cases = {}
            for rule in RULES:
                for capital, amount, cap in ((START,SIZE,False),(OLD+400,OLD,False),
                                             (SIZE,SIZE,False),(SIZE,SIZE,True)):
                    if amount == OLD and rule == "scaled_gate":
                        continue
                    if period != "november" and capital == SIZE and amount == SIZE:
                        continue
                    results = [account(rows,rule,amount,capital,cap) for rows in group]
                    totals = {k:sum(r[k] for r in results) for k in
                              ("buys","open","wins","losses","cash_blocked","fees","gross","net")}
                    if capital == START:
                        assert totals["cash_blocked"] == 0
                    if amount == OLD:
                        key = "progress" if rule == "original_gate" else "baseline"
                        source = (earlier["results"][path]["overall"][key] if period == "march_july"
                                  else later["summary"][period][key][path])
                        assert isclose(totals["net"],source["net"],abs_tol=1e-8)
                    totals["max_required_initial_per_block"] = max(r["min_initial_full_size"] for r in results)
                    if period == "november":
                        totals["trades"] = results[0]["trades"]
                    mode = "cash_cap" if cap else "fixed"
                    cases[f"{rule}_{amount:g}_{capital:g}_{mode}"] = totals
            out[path][period] = cases
    report = dict(fee_per_fill=FEE, initial_cash_main=START, order_notional=SIZE,
                  gain_thresholds=RULES, results=out)
    (HERE/"size_check.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
