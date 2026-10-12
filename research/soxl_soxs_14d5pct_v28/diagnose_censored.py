"""Post-result attribution, never parameter selection or primary replacement."""
import argparse
from collections import Counter
from censored_learning import *


def diagnose(run):
    run=Path(run);records=[read(p) for p in sorted((run/'training').glob('*.json'))]
    records=[r for r in records if '2026-01-01'<=r['date'][:10]<='2026-04-30']
    clocks={};differences=[]
    for c in CLOCKS:
        rows=[r for r in records if r['clock']==c];ready={};actions={};reasons={};censors=[];terminal_counts=[];margins=[]
        for name in ['censored','terminal']:
            ready[name]=sum(r['estimates'][name]['ready'] for r in rows)
            reasons[name]=dict(Counter(r['estimates'][name].get('reason','warmup') for r in rows if not r['estimates'][name]['ready']))
        for route in ROUTES:actions[route]=dict(Counter(r['events'][route]['action'] for r in rows))
        for r in rows:
            a,b=r['estimates']['censored'],r['estimates']['terminal']
            if a['ready']:
                censors.extend(v['right_censored'] for v in a['risks'].values());terminal_counts.extend(v['known_terminals'] for v in a['risks'].values());margins.append(a['margin'])
            if a['ready'] and b['ready']:
                differences.append({'at':r['at'],'point_difference':(np.array(a['point'])-b['point']).tolist(),
                    'censored_action':r['events']['censored_band']['action'],'terminal_action':r['events']['terminal_only_band']['action']})
        clocks[c]={'ticks':len(rows),'ready':ready,'blocked_reasons':reasons,'proposed_actions':actions,
            'mean_censored_parents_per_etf':np.mean(censors) if censors else None,'max_censored_parents_per_etf':max(censors) if censors else None,
            'mean_known_terminals_per_etf':np.mean(terminal_counts) if terminal_counts else None,'mean_block_margin':np.mean(margins) if margins else None}
    trades={}
    for route in ROUTES:
        try:t=pd.read_csv(run/(route+'_trades.csv'))
        except pd.errors.EmptyDataError:t=pd.DataFrame()
        trades[route]={'count':len(t),'entry_clocks':dict(Counter(pd.to_datetime(t.decision_at,utc=True).dt.tz_convert(core.ET).dt.strftime('%H:%M:%S'))) if len(t) else {},
            'symbols':dict(Counter(t.symbol)) if len(t) else {},'exits':dict(Counter(t.reason)) if len(t) else {}}
    receipt={'post_result_diagnostic_only':True,'clock_summary':clocks,'actual_trades':trades,'comparable_clock_ticks':len(differences),
        'all_tick_action_differences':sum(r['events']['censored_band']['action']!=r['events']['terminal_only_band']['action'] for r in records),
        'changed_censored_vs_terminal_actions':sum(r['censored_action']!=r['terminal_action'] for r in differences),
        'max_absolute_price_logvalue_difference':max((max(abs(v) for v in r['point_difference']) for r in differences),default=None),
        'differences':differences,'results_sha256':sha(run/'results.json'),
        'interpretation':'Censor versus terminal-only is a matched local-weight estimator comparison. V27 additionally differs in window, residual regression and horizon support; its increment cannot isolate censoring. Same-date clocks/ETF outcomes are dependent.'}
    dump(run/'diagnostics.json',receipt);print({k:v for k,v in receipt.items() if k not in ['clock_summary','differences']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
