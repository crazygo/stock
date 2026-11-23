"""Replay four preregistered changes to the third-confirmation strategy."""
import argparse
from collections import defaultdict
import json
from math import isclose
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paper_probe import FEE, VIRTUAL, late_account
from study_scale import Context, PATHS, base, score
import limit_replay

NOTIONAL = 600.
RULES = ("baseline", "breakout", "progress", "both", "limit")
FIELDS = ("buys", "sells", "open", "wins", "losses", "targets", "gross", "fees", "net")
EPS = 1e-9


def read(name):
    return json.loads((HERE / name).read_text())


def write(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def observe(ctx, lots):
    second = {}
    events = []
    for lot in lots:
        cycle = lot["cycle"]
        if lot["stage"] == 2:
            second[cycle] = lot
        if lot["stage"] != 3:
            continue
        i, s = lot["decision_i"], lot["symbol"]
        prev = second[cycle]
        assert prev["symbol"] == s and prev["entry_i"] < i < lot["entry_i"] == i+1
        close = ctx.data["tape"][s][i-1][3]
        prior_high = ctx.data["tape"][s][i-2][1]
        px = prev["entry_price"]
        events.append(dict(entry_i=i+1, decision_i=i, cycle=cycle, symbol=s, decision=lot["decision"],
                           date=ctx.data["days"][i+1], session=base.session(ctx.data["times"][i+1]),
                           close=close, prior_high=prior_high, second_entry=px,
                           breakout=close > prior_high, progress=(close-px)*NOTIONAL >= (2*FEE)*px - EPS))
    return events


def run_block(block):
    name = block["name"]
    score.study.BLOCKS[name] = block["start"], block["end"]
    ctx = Context(name)
    result = {}
    for path in PATHS:
        lots = ctx.execute(VIRTUAL, path, FEE)["lots"]
        events = observe(ctx, lots)
        accounts = {}
        for rule in RULES[:-1]:
            chosen = {e["entry_i"] for e in events if rule == "baseline" or
                      (e["breakout"] and e["progress"] if rule == "both" else e[rule])}
            accepted = [t for t in lots if t["stage"] == 3 and t["entry_i"] in chosen]
            accounts[rule] = late_account(dict(ctx.data, path=path), accepted, 3, NOTIONAL)
        accounts["limit"] = limit_replay.run(ctx, path, lots)
        result[path] = dict(events=events, accounts=accounts)
    return result


def closed_outcome(t):
    if t["reason"] == "open":
        return "open"
    return "win" if t["net"] > EPS else "loss" if t["net"] < -EPS else "flat"


def random_keep_expected(rows, kept, groups):
    """Expected P&L, drawing uniformly without replacement within each stratum."""
    pools = defaultdict(list)
    for k, t in rows.items():
        pools[groups[k]].append((k, t))
    expected, movable = 0., 0
    for pool in pools.values():
        n = sum(k in kept for k, _ in pool)
        expected += n/len(pool)*sum(t["net"] for _, t in pool)
        if 0 < n < len(pool):
            movable += n
    return expected, movable


def aggregate(cases, names, path, rule):
    table = {field:0 for field in FIELDS}
    table.update(orig_wins=0, orig_losses=0, avoided_loss_dollars=0., forgone_win_dollars=0.,
                 skipped_win=0, skipped_loss=0, skipped_open=0, skipped_flat=0, kept_win=0, kept_loss=0,
                 expected_random=0., expected_matched=0., matched_swappable=0, retained_pnl_change=0.,
                 removed_marked_net=0., base_net=0., base_buys=0, max_block_drawdown=0.)
    did_buy = set()
    daily_pnl, symbol_stats = {}, {}
    for name in names:
        run = cases[name][path]
        account = run["accounts"][rule]
        baseline = run["accounts"]["baseline"]
        events = {e["entry_i"]:e for e in run["events"]}
        rows = {t["entry_i"]:t for t in baseline["trades"]}
        retained = {t["entry_i"]:t for t in account["trades"]}
        assert retained.keys() <= rows.keys(), "one-to-one cash-funded paired opportunities required"
        for field in FIELDS:
            table[field] += account[field]
        table["base_buys"] += len(rows)
        table["base_net"] += baseline["net"]
        table["orig_wins"] += baseline["wins"]
        table["orig_losses"] += baseline["losses"]
        table["max_block_drawdown"] = max(table["max_block_drawdown"], account["max_drawdown"])
        exp, _ = random_keep_expected(rows, retained, dict.fromkeys(rows, name))
        table["expected_random"] += exp
        strata = {k:(e["symbol"], e["date"], e["session"]) for k, e in events.items()}
        exp, movable = random_keep_expected(rows, retained, strata)
        table["expected_matched"] += exp
        table["matched_swappable"] += movable
        for k,t in rows.items():
            status = closed_outcome(t)
            if k in retained:
                did_buy.add(events[k]["date"])
                if status in ("win", "loss"):
                    table[f"kept_{status}"] += 1
                table["retained_pnl_change"] += retained[k]["net"] - t["net"]
            else:
                table[f"skipped_{status}"] += 1
                table["removed_marked_net"] += t["net"]
                if status == "win":
                    table["forgone_win_dollars"] += t["net"]
                elif status == "loss":
                    table["avoided_loss_dollars"] -= t["net"]
        for t in retained.values():
            sym = symbol_stats.setdefault(t["symbol"], dict(buys=0, net=0.))
            sym["buys"] += 1
            sym["net"] += t["net"]
        previous = 1000.
        for date,d in account["daily"].items():
            assert date not in daily_pnl
            daily_pnl[date] = d["end_equity"]-previous
            previous = d["end_equity"]
    table["buy_days"] = len(did_buy)
    table["net_delta"] = table["net"]-table["base_net"]
    table["purchased_dollars"] = NOTIONAL*table["buys"]
    table["symbol_stats"] = symbol_stats
    table["worst_day"] = min(daily_pnl, key=daily_pnl.get)
    table["worst_day_net"] = daily_pnl[table["worst_day"]]
    assert isclose(sum(daily_pnl.values()), table["net"], abs_tol=1e-7)
    assert isclose(table["net_delta"], table["retained_pnl_change"] - table["removed_marked_net"], abs_tol=1e-7)
    return table


def qualifies(t):
    return (t["sells"] >= 5 and t["buy_days"] >= 2 and t["orig_wins"] > 0 and
            2*t["kept_win"] >= t["orig_wins"] and t["skipped_loss"] >= 1 and
            t["losses"] < t["orig_losses"] and t["net_delta"] > EPS)


def summary(cases, names):
    return {r:{p:aggregate(cases, names, p, r) for p in PATHS} for r in RULES}


def select(development):
    ranking = {}
    for rule in RULES[1:]:
        a = [development[rule][p] for p in PATHS]
        ranking[rule] = dict(passed=all(qualifies(row) for row in a),
                             worst_delta=min(row["net_delta"] for row in a),
                             best_delta=max(row["net_delta"] for row in a))
    eligible = [r for r in ranking if ranking[r]["passed"]]
    chosen = min(eligible, key=lambda r:(-ranking[r]["worst_delta"], -ranking[r]["best_delta"], r),
                 default=None)
    return dict(selected=chosen, ranking=ranking)


def main(phase):
    blocks = read("coverage.json")["blocks"]
    if phase == "develop":
        cases = {b["name"]:run_block(b) for b in blocks["development"]}
        overview = summary(cases, cases.keys())
        frozen = select(overview)
        if (HERE/"selection.json").exists() and frozen != read("selection.json"):
            raise ValueError("A changed selection requires versioning")
        write("development.json", dict(summary=overview, cases=cases))
        write("selection.json", frozen)
        print(frozen)
        return
    dev = read("development.json")
    frozen = read("selection.json")
    assert select(dev["summary"]) == frozen
    cases = dict(dev["cases"])
    overview = {"development":dev["summary"]}
    for phase in ("evaluation","november"):
        names = []
        for b in blocks[phase]:
            names.append(b["name"])
            cases[b["name"]] = run_block(b)
        overview[phase] = summary(cases, names)
    chosen = frozen["selected"]
    passed = bool(chosen) and all(qualifies(t) and t["net"] > EPS
            for period in ("evaluation", "november") for t in overview[period][chosen].values())
    write("results.json", dict(selected=chosen, passed=passed, summary=overview, cases=cases))
    print("selected", chosen, "passed", passed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["develop","evaluate"])
    main(parser.parse_args().phase)
