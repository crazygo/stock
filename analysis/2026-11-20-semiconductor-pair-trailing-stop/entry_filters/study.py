"""Predeclared entry gates. Select on development blocks before auditing later dates."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from early import early_masks

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import replay as base

BLOCKS = {
    "development_a": ("2026-09-28", "2026-10-09"),
    "development_b": ("2026-10-12", "2026-10-16"),
    "october_check": ("2026-10-26", "2026-10-30"),
    "november": ("2026-11-02", "2026-11-19"),
    "last_four": ("2026-11-16", "2026-11-19"),
}
RULES = ("original", "early_trend", "early_reclaim", "early_breakout", "quiet_breakout",
         "chart_recent", "chart_now", "chart_fresh", "chart_reclaim")
CHART = base.ROOT / "analysis/2026-11-17-candle-center-area-inspector"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_period(name):
    begin, end = BLOCKS[name]
    records = json.loads((base.ROOT / "market_data/calendars/nasdaq_sessions_2026_v2.json").read_text())["sessions"]
    times, days = [], []
    for row in records:
        day = row["session_date"]
        if begin <= day <= end:
            assert row["duration_minutes"] == 390
            start = datetime.fromisoformat(day).replace(tzinfo=base.ET) - timedelta(hours=4)
            times.extend(start + j * base.STEP for j in range(288))
            days.extend([day]*288)
    tape, sources = base.load_bars(times)
    return times, days, tape, sources


def chart_flags(times, tape):
    ends = [int((x + base.STEP).timestamp()) for x in times]
    data = json.dumps(dict(times=ends, closes={s:[b[3] for b in tape[s]] for s in base.SYMBOLS}), separators=(",",":"))
    checksum = hashlib.sha256(data.encode()).hexdigest()
    snapshot = HERE / "chart_marks.json"
    cache = json.loads(snapshot.read_text()) if snapshot.exists() else {}
    sources = [CHART / x for x in ("events.js", "timeline_shape.js")]
    if not all(p.exists() for p in sources):
        if checksum not in cache:
            raise ValueError("A new tape needs the chart source files to derive its confirmations")
        saved = {s:set(v) for s,v in cache[checksum]["bar_ends"].items()}
        return {s:[int(t in saved[s]) for t in ends] for s in base.SYMBOLS}
    out = subprocess.run(["node", str(HERE/"events_adapter.cjs")], input=data, text=True,
                         check=True, capture_output=True)
    flags = json.loads(out.stdout)
    assert all(len(flags[s]) == len(times) for s in base.SYMBOLS)
    record = dict(bar_ends={s:[t for t, flag in zip(ends, flags[s]) if flag] for s in base.SYMBOLS},
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    if checksum in cache and record != cache[checksum]:
        raise ValueError("Chart source or confirmation snapshot changed since this run")
    if checksum not in cache:
        cache[checksum] = record
        write(snapshot, cache)
    return flags


def confirmed_pullback(bars, flag, first, last):
    if last < first + 2 or bars[last][3] <= bars[last-1][1]:
        return False
    for event in range(last-2, max(first-1, last-13), -1):
        if not flag[event]:
            continue
        peak = bars[event][3]
        worst = 0.
        for j in range(event+1, last):
            close = bars[j][3]
            peak = max(peak, close)
            worst = max(worst, 1 - close/peak)
        if .005-1e-12 <= worst <= .02+1e-12:
            return True
    return False


def filters(times, tape, flags):
    masks = early_masks(tape, times)
    masks.update({r:{s:[False]*len(times) for s in base.SYMBOLS}
                  for r in RULES if r not in masks and r != "original"})
    start = 0
    for i, now in enumerate(times):
        if i and now-times[i-1] != base.STEP:
            start = i
        if i <= start:
            continue
        for s in base.SYMBOLS:
            flag, bars = flags[s], tape[s]
            masks["chart_recent"][s][i] = any(flag[max(start,i-12):i])
            masks["chart_now"][s][i] = bool(flag[i-1])
            masks["chart_fresh"][s][i] = bool(flag[i-1] and not any(flag[max(start,i-7):i-1]))
            masks["chart_reclaim"][s][i] = confirmed_pullback(bars,flag,start,i-1)
            if masks["early_breakout"][s][i]:
                masks["quiet_breakout"][s][i] = base.signal(bars[i-12:i])["drawdown"] < .015-1e-12
    return masks


def masked(features, mask):
    if mask is None:
        return features
    return [{s:dict(x, passed=bool(x["passed"] and mask[s][i])) for s,x in f.items()}
            for i, f in enumerate(features)]


def summarize(stat, trades):
    closed = [t for t in trades if t["exit_reason"] != "open"]
    loss = sum(t["net"] < 0 for t in closed)
    return dict(buys=stat["buys"], closed=len(closed), wins=stat["net_wins"], losses=loss,
                open=stat["open"], targets=stat["target"], loss_rate=loss/len(closed) if closed else None,
                gross=stat["gross_return"], net=stat["net_return"], max_dd=stat["drawdown_at_5m_closes"],
                coverage=len({t["entry_day"] for t in trades}), daily=stat["daily"],
                by_symbol=stat["by_symbol"], flat_blocked=stat["checks"].get("filter_blocked",0))


def counted(trades):
    return dict(loss=sum(t["exit_reason"]!="open" and t["net"]<0 for t in trades),
                win=sum(t["exit_reason"]!="open" and t["net"]>0 for t in trades),
                flat=sum(t["exit_reason"]!="open" and t["net"]==0 for t in trades),
                open=sum(t["exit_reason"]=="open" for t in trades))


def assess(block, rules, n=5):
    times, days, tape, sources = load_period(block)
    ref = base.signals(times, tape, n)
    flags = chart_flags(times, tape)
    masks = filters(times, tape, flags)
    indexes = {t.isoformat():i for i,t in enumerate(times)}
    output, ledgers = {}, {}
    for rule in rules:
        mask = masks.get(rule)
        feats = masked(ref,mask)
        output[rule] = {}
        for path in ("OLHC", "OHLC"):
            stat, trades, _ = base.replay(times, days, tape, feats, path)
            result = summarize(stat,trades)
            if rule == "original":
                ledgers[path] = trades
            else:
                # Direct gate at the old decision; later differences in occupancy are kept separate.
                rejected = [t for t in ledgers[path] if not mask[t["symbol"]][indexes[t["signal_end"]]]]
                result["directly_vetoed"] = counted(rejected)
            if path == "OLHC":
                ledgers[rule] = trades
            output[rule][path] = result
    for p in ("OLHC","OHLC"):
        del ledgers[p]
    return dict(rules=output, sources=sources, confirmations={s:sum(v) for s,v in flags.items()}), ledgers


def pick(development):
    table = {}
    for name in RULES[1:]:
        deltas, growth, qualified = [], [], True
        for block, min_trades, min_days in (("development_a",8,4),("development_b",4,3)):
            cases = development[block]["rules"]
            for path in ("OLHC","OHLC"):
                orig, result = cases["original"][path], cases[name][path]
                qualified &= (result["closed"] >= min_trades and result["coverage"] >= min_days
                              and result["buys"] <= orig["buys"]*.75)
                if result["loss_rate"] is not None:
                    deltas.append(result["loss_rate"] - orig["loss_rate"])
                growth.append(math.log1p(result["net"])/len(result["daily"]))
        table[name] = dict(qualified=qualified, worst_loss_rate_delta=max(deltas) if len(deltas)==4 else None,
                           worst_daily_log_growth=min(growth))
    eligible = [r for r in table if table[r]["qualified"]]
    selected = min(eligible,key=lambda r:(table[r]["worst_loss_rate_delta"],-table[r]["worst_daily_log_growth"],r)) if eligible else None
    return dict(selected=selected, ranking=table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["develop","evaluate"])
    phase = parser.parse_args().phase
    if phase == "develop":
        data = {b:assess(b,RULES)[0] for b in ("development_a","development_b")}
        result = pick(data)
        freeze = HERE/"freeze.json"
        if freeze.exists() and json.loads(freeze.read_text()) != result:
            raise ValueError("A changed selection requires a new study version")
        write(HERE/"development.json", data)
        write(freeze, result)
        print(json.dumps(result,indent=2))
        return
    selected = json.loads((HERE/"freeze.json").read_text())["selected"]
    shortlist = ["original",selected] if selected else ["original"]
    later = {b:assess(b,shortlist)[0] for b in ("october_check","last_four")}
    later["november"], trades = assess("november",RULES)
    later["other_windows"] = {str(n):assess("november",shortlist,n)[0] for n in (3,10)}
    write(HERE/"results.json",later)
    write(HERE/"ledger.json",dict(selected=selected,original="original",trades=trades))
    for name, v in later["november"]["rules"].items():
        r=v["OLHC"]
        print(name, r["buys"], r["losses"], r["wins"], r["net"], r.get("directly_vetoed"))


if __name__ == "__main__":
    main()
