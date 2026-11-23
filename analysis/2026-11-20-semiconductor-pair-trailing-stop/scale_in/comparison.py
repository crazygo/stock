"""Paired entry-price/fee attribution and matched single-lot random-entry checks."""
from collections import defaultdict
from pathlib import Path
import json
import math
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import controls

RANDOM_SEED = 20261120
REPLICATES = 5000


def gross_ratio_and_exit(bars, i):
    wealth, _, reason = controls.trade(bars, i)
    # The shared path scanner prices exits independently of its percentage cost.
    gross_ratio = wealth * (1 + controls.FEE)
    if reason != 'open':
        gross_ratio /= (1 - controls.FEE)
    return gross_ratio - 1, reason


def pnl_totals(rows):
    """Each row is (gross dollars, fees, open, gross notional entered)."""
    realized = [r for r in rows if not r[2]]
    net = sum(r[0] - r[1] for r in rows)
    return dict(buys=len(rows), closed=len(realized), open=len(rows)-len(realized),
                wins=sum(r[0]-r[1] > 1e-9 for r in realized),
                losses=sum(r[0]-r[1] < -1e-9 for r in realized),
                capital_deployed=sum(r[3] for r in rows),
                gross_pnl=sum(r[0] for r in rows), fees=sum(r[1] for r in rows), net_pnl=net)


def summary_for_stage(rows, fee):
    done = [r for r in rows if r['exit_i'] is not None]
    n = len(done)
    wins = sum(r['net_pnl'] > 1e-9 for r in done)
    gross_wins = sum(r['gross_pnl'] > 1e-9 for r in done)
    ratios = [r['gross_pnl']/(r['entry_price']*r['qty']) for r in done]
    common_wins = sum(200*x > 2*fee + 1e-9 for x in ratios)
    n_target = sum(r['reason']=='target' for r in done)
    return dict(buys=len(rows), closed=n, open=len(rows)-n, net_wins=wins, gross_wins=gross_wins,
                target=n_target, target_rate=n_target/n if n else None, net_win_rate=wins/n if n else None,
                gross_win_rate=gross_wins/n if n else None, common_200_net_wins=common_wins,
                common_200_net_win_rate=common_wins/n if n else None,
                mean_closed_gross_pct=100*np.mean(ratios) if n else None,
                gross_pnl=sum(r['gross_pnl'] for r in rows), net_pnl=sum(r['net_pnl'] for r in rows))


def stage_pairs(lots, fee):
    seeds = {r['cycle']:r for r in lots if r['stage'] == 1}
    out = {}
    for stage in (2,3):
        pairs = [(seeds[r['cycle']],r) for r in lots if r['stage']==stage]
        mature = [(s,r) for s,r in pairs if s['exit_i'] is not None and r['exit_i'] is not None]
        pairs_won = []
        for seed, add in mature:
            seed_ret = seed['gross_pnl'] / seed['value']
            add_ret = add['gross_pnl'] / add['value']
            pairs_won.append((200*seed_ret > 2*fee+1e-9, 200*add_ret > 2*fee+1e-9))
        out[str(stage)] = dict(reached=len(pairs), both_closed=len(mature),
                               seed_wins_at_200=sum(a for a,b in pairs_won),
                               add_wins_at_200=sum(b for a,b in pairs_won),
                               both_won_at_200=sum(a and b for a,b in pairs_won),
                               seed_only_won_at_200=sum(a and not b for a,b in pairs_won),
                               add_only_won_at_200=sum(b and not a for a,b in pairs_won),
                               neither_won_at_200=sum(not a and not b for a,b in pairs_won))
    return out


def attribution(lots, fee, cap, initial):
    """Hold cycle starts fixed; re-price all exposure on the observed seed's path."""
    grouped = defaultdict(list)
    for lot in lots:
        grouped[lot['cycle']].append(lot)
    actual_rows, sized_upfront, full_upfront, funded_full = [], [], [], []
    funded_cash = initial
    rounds = defaultdict(list)
    for group in grouped.values():
        first = group[0]
        assert first['stage'] == 1 and [t['stage'] for t in group] == list(range(1, len(group)+1))
        assert all(t['symbol'] == first['symbol'] for t in group)
        actual = [(r['gross_pnl'], (1+int(r['exit_i'] is not None))*fee, r['exit_i'] is None,
                   r['entry_price']*r['qty']) for r in group]
        deployed = sum(r[3] for r in actual)
        total = pnl_totals(actual)
        assert math.isclose(total['net_pnl'], sum(r['net_pnl'] for r in group), abs_tol=1e-7)
        first_ret = first['gross_pnl']/(first['entry_price']*first['qty'])
        end_open = first['exit_i'] is None
        first_fees = (1 + int(not end_open))*fee
        same = (first_ret*deployed, first_fees, end_open, deployed)
        full = (first_ret*cap, first_fees, end_open, cap)
        actual_rows.extend(actual)
        sized_upfront.append(same)
        full_upfront.append(full)
        if funded_cash + 1e-8 >= cap + fee:
            funded_full.append(full)
            funded_cash += full[0]-full[1]
        rounds['actual'].append((total['gross_pnl'], total['fees'], bool(total['open']), deployed))
        rounds['same_deployment'].append(same)
        rounds['full'].append(full)
    actual = pnl_totals(actual_rows)
    same = pnl_totals(sized_upfront)
    full = pnl_totals(full_upfront)
    round_detail = {}
    for name, rows in rounds.items():
        summary = pnl_totals(rows)
        summary['mean_deployed_per_entered_cycle'] = summary['capital_deployed']/len(rows) if rows else 0.
        round_detail[name] = summary
    result = dict(actual=actual, fixed_600_on_same_cycle_starts=full, same_total_at_seed=same,
                  same_starts_600_with_cash_limit=pnl_totals(funded_full),
                  capital_blocked_600=len(full_upfront)-len(funded_full),
                  rounds=round_detail, extra_fees_vs_same_total=actual['fees']-same['fees'],
                  gross_entry_timing_effect=actual['gross_pnl']-same['gross_pnl'],
                  exposure_difference_at_seed=same['gross_pnl']-full['gross_pnl'],
                  delta_net_vs_same_total=actual['net_pnl']-same['net_pnl'])
    assert math.isclose(result['delta_net_vs_same_total'],
                        result['gross_entry_timing_effect']-result['extra_fees_vs_same_total'], abs_tol=1e-8)
    assert math.isclose(actual['net_pnl']-full['net_pnl'], result['delta_net_vs_same_total'] +
                        result['exposure_difference_at_seed'], abs_tol=1e-8)
    return result


def random_comparison(lots, fee, grid, keys, all_exits):
    eligible = set(controls.eligible_entries(grid, 12))
    pools = defaultdict(list)
    for i in sorted(eligible):
        pools[keys[i]].append(i)
    grouped = defaultdict(list)
    for row in lots:
        i, s = row['entry_i'], row['symbol']
        assert i in eligible
        gross, reason = all_exits[s][i]
        notional = row['entry_price'] * row['qty']
        assert math.isclose(notional*gross, row['gross_pnl'], abs_tol=1e-7)
        assert (reason=='open') == (row['exit_i'] is None)
        grouped[s,keys[i]].append(row)
    rng = np.random.default_rng(RANDOM_SEED)
    random_pnl = np.zeros(REPLICATES)
    exact = defaultdict(lambda:defaultdict(float))
    for (sym, key), rows in sorted(grouped.items()):
        pool = pools[key]
        raw = np.array([all_exits[sym][i][0] for i in pool])
        reason = np.array([all_exits[sym][i][1] for i in pool])
        closed = reason != 'open'
        fees = fee*(1+closed.astype(float))
        assert len(pool) >= len(rows)
        # A random permutation gives distinct bars and a random assignment of
        # the actual lot notionals within each symbol x day x session group.
        draws = rng.random((REPLICATES, len(pool))).argsort(axis=1)[:,:len(rows)]
        for k, row in enumerate(rows):
            notional = row['entry_price'] * row['qty']
            net = notional*raw - fees
            random_pnl += net[draws[:,k]]
            for stage in ('all',str(row['stage'])):
                a = exact[stage]
                a['buys'] += 1
                a['expected_closed'] += np.mean(closed)
                a['expected_net_wins'] += np.mean(closed & (net>1e-9))
                a['expected_target'] += np.mean(reason=='target')
                a['expected_net_pnl'] += np.mean(net)
    for a in exact.values():
        n = a['expected_closed']
        a['expected_net_win_ratio'] = a['expected_net_wins']/n if n else None
        a['expected_target_ratio'] = a['expected_target']/n if n else None
    actual_net = sum(r['net_pnl'] for r in lots)
    return dict(exact=dict(exact), random_sum_of_lot_pnl=controls.describe(random_pnl),
                actual_sum_of_lot_pnl=actual_net,
                actual_percentile=100*float(np.mean(random_pnl <= actual_net+1e-9)),
                draws=REPLICATES, groups=len(grouped), seed=RANDOM_SEED)


def compare(lots, fee, grid, keys, all_exits, cap=600, initial=1000):
    return dict(attribution=attribution(lots, fee, cap, initial),
                actual_stage={str(k): summary_for_stage([r for r in lots if r['stage']==k], fee)
                              for k in (1,2,3)},
                seed_vs_add_within_same_reached_cycles=stage_pairs(lots,fee),
                random=random_comparison(lots, fee, grid, keys, all_exits))


def main():
    source = json.loads((HERE/'ledger.json').read_text())
    assert source['block']=='november'
    grid, keys, prices = controls.tape()
    all_exits = {s:[gross_ratio_and_exit(prices[j], i) for i in range(len(grid))]
                 for j,s in enumerate(controls.SYMBOLS)}
    out = dict(block=source['block'], policy='Independent per-lot brackets, OLHC, fixed fee per fill.',
               paired='Hold actually filled cycle starts fixed. Same-total-at-seed uses final deployed dollars, which were unknown when the first order was decided. The $600 attribution ignores cash; a separate $600 series skips unfunded orders.',
               matched='Random entry slots use the same symbol, exchange trading date, session and lot dollars. Draws without replacement within each group. Each lot is scored as a separate path; aggregate dollar P&L is not a single-capital simulation.',
               runs={})
    for label in ('primary_best3_OLHC','primary_example_OLHC'):
        run = source['runs'][label]
        assert run['path']=='OLHC' and not run['whole']
        comparison = compare(run['lots'],run['fee'],grid,keys,all_exits,initial=source['initial'])
        assert math.isclose(comparison['attribution']['actual']['net_pnl'],run['summary']['net_pnl'],abs_tol=1e-7)
        out['runs'][label] = dict(config=run['config'],fee_each_side=run['fee'],**comparison)
    (HERE/'comparison.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')


if __name__=='__main__':
    main()
