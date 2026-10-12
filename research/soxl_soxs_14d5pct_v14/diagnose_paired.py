"""Registered paired direction diagnostics; never changes frozen actions."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from paired_entry import core


def interval(values):
    a=np.asarray(values,dtype=float);n=len(a)
    if not n:return None
    rng=np.random.default_rng(1414);starts=rng.integers(0,n,size=(5000,int(np.ceil(n/10))))
    idx=((starts[:,:,None]+np.arange(10))%n).reshape(5000,-1)[:,:n]
    means=a[idx].mean(axis=1)
    return {'mean':float(a.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),'dates':n,
            'block_dates':10,'draws':5000,'seed':1414,'descriptive_exposed_development':True}


def diagnose(run):
    run=Path(run);reg=json.loads((run/'registration.json').read_text());p=reg['preregistration']['protocol'];phase=p[reg['phase']]
    panel=pd.read_parquet(run/'panel.parquet');pred=json.loads((run/'predictions.json').read_text())
    dates=panel[(panel.date>=phase['evaluation_start'])&(panel.date<=phase['evaluation_end'])]
    cutoff=pd.Timestamp(phase['evaluation_end'],tz=core.ET)+pd.Timedelta(days=1)
    mature=dates[dates.release.notna()&(dates.release<cutoff)]
    out={'phase':reg['phase'],'total_dates':len(dates),'joint_mature_dates':len(mature),'unknown_dates':[str(d.date()) for d in dates[~dates.index.isin(mature.index)].date],
        'q_is_target_hit_probability':False,'overlapping_opportunities_are_not_capital_trades':True,'routes':{}}
    chosen={}
    for name,events in pred.items():
        mapping={pd.Timestamp(e['at']):e for e in events};rows=[]
        for at,r in mature.iterrows():
            e=mapping[at]
            if not e['ready']:continue
            top=e['top_symbol'];signed=1 if top=='SOXL' else -1;actual=float(r.C+signed*r.D)
            support=r.relative_5>0 and r.breadth_5>0 and r.volume_direction_diffusion>0
            weakness=r.relative_5<0 and r.breadth_5<0 and r.volume_direction_diffusion<0
            rows.append({'at':at,'top':top,'value':e['adjusted'][top],'actual':actual,
                'regret':abs(r.D)-signed*r.D,'raw_d_error':(e['raw']['D']-r.D)**2,
                'adjusted_d_error':(e['adjusted']['D']-r.D)**2,'action':e['action'],
                'regime':'support' if support else 'weakness' if weakness else 'mixed'})
        f=pd.DataFrame(rows);positive=f[f.action.ne('CASH')] if len(f) else f
        chosen[name]=dict(zip(f['at'],f.actual)) if len(f) else {}
        out['routes'][name]={'ready_dates':len(f),'top_SOXL_dates':int(f.top.eq('SOXL').sum()) if len(f) else 0,
            'raw_D_mse':float(f.raw_d_error.mean()) if len(f) else None,'adjusted_D_mse':float(f.adjusted_d_error.mean()) if len(f) else None,
            'mean_financial_regret':float(f.regret.mean()) if len(f) else None,
            'top_predicted_mean_log_return':float(f.value.mean()) if len(f) else None,
            'top_actual_mean_log_return':float(f.actual.mean()) if len(f) else None,
            'positive_action_dates':len(positive),'positive_actual_mean_log_return':float(positive.actual.mean()) if len(positive) else None,
            'regime_slices':{k:{'dates':len(g),'top_actual_mean_log_return':float(g.actual.mean()),
                 'mean_financial_regret':float(g.regret.mean()),'positive_actions':int(g.action.ne('CASH').sum())}
                for k,g in f.groupby('regime')} if len(f) else {}}
    common=sorted(set(chosen['regret_regime'])&set(chosen['regret_base']))
    out['regime_minus_base_chosen_log_return']=interval([chosen['regret_regime'][at]-chosen['regret_base'][at] for at in common])
    out['always_SOXL_opportunity_mean_log_return']=float((mature.C+mature.D).mean())
    out['always_SOXS_opportunity_mean_log_return']=float((mature.C-mature.D).mean())
    core.dump(run/'diagnosis.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);diagnose(p.parse_args().run)
