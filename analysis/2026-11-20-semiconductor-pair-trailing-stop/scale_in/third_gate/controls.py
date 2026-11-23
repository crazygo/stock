"""Exact random omission of November opportunities plus a matched-timing placebo."""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
from random import Random
from statistics import mean, median
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from engine import LiveLot, process_price
from study_scale import base, score

AMOUNT, FEE = 600., 2.5
SAMPLES = 2000


def distribution(values, actual):
    ordered = sorted(values)
    return dict(samples=len(values), mean=mean(values), median=median(values),
                p05=ordered[int(.05*(len(values)-1))], p95=ordered[int(.95*(len(values)-1))],
                share_at_least_actual=sum(x >= actual-1e-8 for x in values)/len(values))


def replay_trade(tape, symbol, i, path):
    price = tape[symbol][i][0]
    lot = LiveLot({}, price, 1.08*price)
    for j in range(i, len(tape[symbol])):
        op, hi, lo, cl = tape[symbol][j]
        prices = (op, lo, hi, cl) if path == 'OLHC' else (op, hi, lo, cl)
        for k, p in enumerate(prices):
            outcome = process_price(lot, p, at_open=k == 0)
            if outcome:
                return j, k == 0, AMOUNT*(outcome[1]/price-1)-2*FEE
    return len(tape[symbol]), False, AMOUNT*(cl/price-1)-FEE


def analyze(block, runs, path):
    times, days, tape, _ = score.study.load_period(block)
    # A market order must have a preceding contiguous decision opportunity.
    grid = defaultdict(list)
    for i in range(1, len(times)):
        if times[i] == times[i-1]+base.STEP:
            grid[days[i],base.session(times[i])].append(i)
    cached = lru_cache(maxsize=None)(lambda s, i:replay_trade(tape,s,i,path))
    conclusions = {}
    for rule in ('baseline','progress'):
        account = runs[path]['accounts'][rule]
        groups = Counter((t['symbol'],days[t['entry_i']],base.session(times[t['entry_i']])) for t in account['trades'])
        rng = Random(f'semiconductor-third-gate-v1:{block}:{path}:{rule}')
        base_rows = runs[path]['accounts']['baseline']['trades']
        subsets = [sum(t['net'] for t in combo) for combo in combinations(base_rows, account['buys'])]
        need = sum(groups.values())
        draws, attempts = [], 0
        while len(draws) < SAMPLES and attempts < SAMPLES*100:
            attempts += 1
            orders = sorted((i, symbol) for (symbol, day, session), n in groups.items()
                            for i in rng.sample(grid[day,session], n))
            if len({i for i,_ in orders}) != need:
                continue
            end, at_open, net = -1, False, 0.
            for idx,symbol in orders:
                if idx < end or (idx == end and not at_open) or 1000.+net < AMOUNT+FEE:
                    break
                end, at_open, pnl = cached(symbol,idx)
                net += pnl
            else:
                draws.append(net)
        assert len(draws) == SAMPLES
        conclusions[rule] = dict(actual=account['net'], count=need, attempts=attempts,
                                 random_subset=distribution(subsets,account['net']),
                                 random_time=distribution(draws,account['net']))
    return conclusions


def main():
    runs = json.loads((HERE/'results.json').read_text())['cases']['november']
    results = {p:analyze('november',runs,p) for p in ('OLHC','OHLC')}
    (HERE/'controls.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__ == '__main__':
    main()
