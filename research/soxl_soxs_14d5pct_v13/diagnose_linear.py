"""Matched probability and selected-value diagnostics, never a policy selector."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from linear_entry import core, MODELS
from verify_continuation import read, labels, key


def block_difference(values):
    x=np.asarray(values,dtype=float); n=len(x)
    if not n: return {'dates':0,'mean':None,'interval95':None}
    rng=np.random.default_rng(1313); starts=rng.integers(0,n,size=(5000,int(np.ceil(n/10))))
    indices=((starts[...,None]+np.arange(10))%n).reshape(5000,-1)[:,:n]
    intervals=np.quantile(x[indices].mean(axis=1),[.025,.975])
    return {'dates':n,'mean':float(x.mean()),'interval95':intervals.tolist(),'block_sessions':10,'replicates':5000,
            'positive_supported':bool(intervals[0]>0),'not_independent_forward_validation':True}


def average(values):
    return float(np.mean(values)) if len(values) else None


def diagnose(run):
    run=Path(run); reg=read(run/'registration.json'); p=reg['preregistration']['protocol']; settings=p[reg['phase']]
    base=labels(run/'baseline_labels.json'); histories=read(run/'predictions.json'); out={'phase':reg['phase'],'routes':{}}
    for name,events in histories.items():
        records=[]; tops=[]; unknown=[]; unready=[]; calibration=[]
        for e in events:
            if not settings['evaluation_start']<=e['date'][:10]<=settings['evaluation_end']: continue
            pair=[base[key(e['at'],s)] for s in core.ETFS]
            if not all(x['complete_window'] and x.get('label_available_at') is not None for x in pair): unknown.append(e['date']); continue
            if not e['ready']: unready.append(e['date']); continue
            model_name=name if name in MODELS else p['primary']
            cal=read(run/'training'/(e['date'][:10]+'.json'))['routes'][model_name]['calibration']; calibration.append(cal)
            for s,lab in zip(core.ETFS,pair):
                raw=e['raw'][s]; value=float(np.log1p(lab['net_return']))
                records.append({'date':e['date'],'symbol':s,'hit':int(lab['target_hit']),'p':raw['p'],
                    'baseline_p':raw['baseline_p'],'score':raw['score'],'actual':value})
                if s==e['top_symbol']:
                    tops.append({'date':e['date'],'symbol':s,'raw':raw['score'],'adjusted':e['adjusted'],'actual':value,
                        'action':e['action'],'hit':int(lab['target_hit']),
                        'within_support':e['within_calibration_score_range']})
        f=pd.DataFrame(records); metrics={}
        for s in core.ETFS:
            g=f[f.symbol.eq(s)] if len(f) else pd.DataFrame(); bins=[]
            for lo,hi in zip([0.,.2,.4,.6,.8],[.2,.4,.6,.8,1.000001]):
                part=g[(g.p>=lo)&(g.p<hi)] if len(g) else pd.DataFrame()
                bins.append({'lo':lo,'hi':min(hi,1.),'n':len(part),
                    'predicted':average(part.p) if len(part) else None,'observed':average(part.hit) if len(part) else None})
            metrics[s]={'dates':len(g),'hits':int(g.hit.sum()) if len(g) else 0,
                'mean_probability':average(g.p) if len(g) else None,'hit_fraction':average(g.hit) if len(g) else None,
                'brier':average((g.p-g.hit)**2) if len(g) else None,
                'historical_frequency_brier':average((g.baseline_p-g.hit)**2) if len(g) else None,
                'raw_log_value_mse':average((g.score-g.actual)**2) if len(g) else None,'reliability':bins}
        positive=[e for e in tops if e['action']!='CASH']
        if len(f):
            f['brier_gain']=(f.baseline_p-f.hit)**2-(f.p-f.hit)**2
            improvement=block_difference(f.groupby('date').brier_gain.mean().to_numpy())
        else: improvement=block_difference([])
        out['routes'][name]={'ready_mature_dates':len(tops),'joint_immature_dates':unknown,'unready_mature_dates':unready,
            'probability_is_classifier_prediction':MODELS[name if name in MODELS else p['primary']][0]=='logit',
            'per_symbol':metrics,'date_equal_probability_gain':improvement,
            'top':{'raw_mean':average([e['raw'] for e in tops]),'corrected_mean':average([e['adjusted'] for e in tops]),
                'actual_mean':average([e['actual'] for e in tops]),'raw_mse':average([(e['raw']-e['actual'])**2 for e in tops]),
                'corrected_mse':average([(e['adjusted']-e['actual'])**2 for e in tops]),
                'positive_candidate_dates':len(positive),'positive_predicted_mean':average([e['adjusted'] for e in positive]),
                'positive_actual_mean':average([e['actual'] for e in positive]),'positive_hit_count':sum(e['hit'] for e in positive),
                'negative_unconstrained_calibration_slope_dates':sum(c['unconstrained_coefficient']<0 for c in calibration),
                'outside_calibration_support_dates':sum(not e['within_support'] for e in tops),
                'positive_outside_support_dates':sum(not e['within_support'] for e in positive)},
            'counterfactual_dates_are_not_independent_actual_trades':True}
    core.dump(run/'diagnosis.json',out); print({n:v['top'] for n,v in out['routes'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True); diagnose(p.parse_args().run)
