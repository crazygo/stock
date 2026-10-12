"""Describe detected change, effective memory and executable outcomes separately."""
import argparse
from collections import Counter
from regime_learning import *


def diagnose(run):
    run=Path(run);records=[read(p) for p in sorted((run/'training').glob('*.json'))]
    records=[r for r in records if '2026-01-01'<=r['date'][:10]<='2026-04-30']
    states=[s for s in read(run/'states.json') if '2026-01-01'<=s['date'][:10]<='2026-04-30']
    clocks={};differences=[]
    for c in CLOCKS:
        rows=[r for r in records if r['clock']==c];summary={}
        for mode in ['regime','hazard']:
            models=[r['estimates'][mode] for r in rows];with_weights=[m for m in models if 'weights' in m]
            recent=[sum(w for w,a in zip(m['weights'],m['ages']) if a<=20)/sum(m['weights']) for m in with_weights]
            original=[sum(w for w,a in zip(weights(len(m['members']),63),m['ages']) if a<=20)/len(m['members']) for m in with_weights]
            summary[mode]={'ready':sum(m['ready'] for m in models),'expected_mature_regime_mass_min':min((m['current_regime_mature_mass'] for m in with_weights),default=None),
                'insufficient_regime_mass_ticks':sum(m['current_regime_mature_mass']<10 for m in with_weights),
                'effective_dates_min':min((m['effective_dates'] for m in with_weights),default=None),
                'mean_recent20_weight':float(np.mean(recent)) if recent else None,'original_mean_recent20_weight':float(np.mean(original)) if original else None}
        summary['actions']={route:dict(Counter(r['events'][route]['action'] for r in rows)) for route in ROUTES}
        clocks[c]=summary
        for r in rows:
            differences.append({'at':r['at'],'regime':r['events']['regime_band']['action'],'prior':r['events']['hazard_only_band']['action'],
                'v27':r['events']['source_v27']['action'],'point':r['events']['regime_point']['action']})
    trades={}
    for route in ROUTES:
        try:t=pd.read_csv(run/(route+'_trades.csv'))
        except pd.errors.EmptyDataError:t=pd.DataFrame()
        trades[route]={'count':len(t),'entry_clocks':dict(Counter(pd.to_datetime(t.decision_at,utc=True).dt.tz_convert(core.ET).dt.strftime('%H:%M:%S'))) if len(t) else {},
            'symbols':dict(Counter(t.symbol)) if len(t) else {},'exits':dict(Counter(t.reason)) if len(t) else {}}
    receipt={'post_result_diagnostic_only':True,'state_dates':len(states),'clock_ticks':len(records),
        'expected_run_length':{'min':min(s['expected_run_length'] for s in states),'median':float(np.median([s['expected_run_length'] for s in states])),
            'max':max(s['expected_run_length'] for s in states)},'max_last_five_change_probability':max(s['last_five_change_probability'] for s in states),
        'changed_actions_versus_prior':sum(r['regime']!=r['prior'] for r in differences),
        'changed_actions_versus_v27':sum(r['regime']!=r['v27'] for r in differences),
        'clock_summary':clocks,'trades':trades,'action_differences':differences,'states':states,
        'results_sha256':sha(run/'results.json'),'filtering_state_changes_is_not_proof_of_profitable_forecasting':True}
    dump(run/'diagnostics.json',receipt)
    print({k:v for k,v in receipt.items() if k not in ['clock_summary','action_differences','states']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
