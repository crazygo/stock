"""Descriptive, post-outcome diagnostics; cannot select or promote a route."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from clock_training import core,CLOCKS
from run_clocks import tag,sha
from verify_continuation import read,labels,key


def diagnose(run):
    run=Path(run);reg=read(run/'registration.json');settings=reg['preregistration']['protocol'][reg['phase']]
    features=pd.read_parquet(run/'features.parquet');panel=pd.read_parquet(run/'panel.parquet')
    features=features[features.date.between(settings['evaluation_start'],settings['evaluation_end'])]
    cutoff=pd.Timestamp(read(run/'input_manifest.json')['observation_cutoff_exclusive']);base=labels(run/'baseline_labels.json')
    results=read(run/'results.json');out={'phase':reg['phase'],'post_outcome_descriptive_only':True,
        'diagnostics_sha256':sha(Path(__file__)),'results_sha256':sha(run/'results.json'),'clock_variants':{},'routes':{}}
    for c in CLOCKS:
        ids=list(features[features.clock==c].index);mature=[t for t in ids if pd.notna(panel.loc[t,'release']) and panel.loc[t,'release']<cutoff]
        for v in ['base','context']:
            records={t:read(run/'training'/(tag(t)+'.json'))['variants'][v] for t in ids}
            for mode in ['early','full','raw']:
                ready=[t for t in ids if records[t]['forecasts'][mode]['event']['ready']]
                scored=[t for t in mature if t in ready]
                eligible=[t for t in scored if records[t]['capital'][mode]['event']['action'] in core.ETFS]
                correct=0;values=[];errors=[];regrets=[]
                for t in scored:
                    e=records[t]['forecasts'][mode]['event'];s=e['top_symbol'];actual=base[key(t,s)]['net_return']
                    other=base[key(t,'SOXS' if s=='SOXL' else 'SOXL')]['net_return']
                    correct+=actual>=other;values.append(actual);errors.append((e['adjusted'][s]-np.log1p(actual))**2)
                    regrets.append(max(actual,other,0.)-actual)
                eligible_correct=sum(base[key(t,records[t]['capital'][mode]['event']['action'])]['net_return']>=
                    base[key(t,'SOXS' if records[t]['capital'][mode]['event']['action']=='SOXL' else 'SOXL')]['net_return'] for t in eligible)
                out['clock_variants'][c+'_'+v+'_'+mode]={'forecast_ticks':len(ids),'unique_dates':len(ids),'ready':len(ready),
                    'common_full_window_mature':len(scored),'unknown_among_ready':len(ready)-len(scored),
                    'better_etf_direction_correct':int(correct),'mean_selected_counterfactual_return':float(np.mean(values)) if values else None,
                    'selected_log_payoff_mse':float(np.mean(errors)) if errors else None,'mean_cash_inclusive_regret':float(np.mean(regrets)) if regrets else None,
                    'mature_risk_eligible':len(eligible),'eligible_better_etf_correct':int(eligible_correct),
                    'eligible_ticks_all':sum(records[t]['capital'][mode]['event']['action'] in core.ETFS for t in ids),
                    'capital_reasons':dict(Counter(records[t]['capital'][mode]['event']['reason'] for t in ids)),
                    'clock_ticks_are_not_independent_trades':True}
    for n,stats in results['routes'].items():
        ds=pd.read_csv(run/(n+'_decisions.csv'));bought=ds[ds.decision.str.startswith('buy_')]
        try:trades=pd.read_csv(run/(n+'_trades.csv'))
        except pd.errors.EmptyDataError:trades=pd.DataFrame()
        out['routes'][n]={'monthly':stats['monthly'],'primary_average':stats['primary_average'],'continuous_return':stats['continuous_return'],
            'completed_trades':stats['completed_trades'],'actual_entries':len(bought),'entry_clock_counts':dict(Counter(bought.clock)),
            'statuses':dict(Counter(ds.decision)),'exit_reasons':dict(Counter(trades.reason)) if len(trades) else {},
            'readiness_is_not_capital_acceptance':True}
    core.dump(run/'diagnostics.json',out)
    print({n:{'entries':v['actual_entries'],'clock':v['entry_clock_counts']} for n,v in out['routes'].items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
