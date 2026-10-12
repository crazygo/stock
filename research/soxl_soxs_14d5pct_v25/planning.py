"""Daily finite fitted-Q iteration over complete settled-cash transitions."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v24'))
from joint_loss import core,day_cutoff,restrict_market,development_gate,COLUMNS,WeightedScale,weights
from verify_continuation import key

ACTIONS=['CASH','SOXL','SOXS'];GAMMA=.99;ITERATIONS=20
ROUTES=['planning_calibrated','planning_raw','myopic_calibrated','myopic_raw',
        'same_action_regular_cash','source_full_cash','source_early','cash','double_cost']


def next_cycle(market,label):
    net=label.get('net_return',np.nan)
    if label.get('exit_at') is None or net is None or not np.isfinite(net):return None
    day=label['exit_at'].tz_convert(core.ET).tz_localize(None).normalize()
    later=market.sessions[market.sessions>day]
    if not len(later):return None
    free=later[0]
    for d in label.get('dividends',[]):free=max(free,d['pay'])
    candidates=market.sessions[market.sessions>=free]
    return candidates[0] if len(candidates) else None


def transition_bank(market,panel,labels):
    by_date={r.date:at for at,r in panel.iterrows()};rows=[]
    for at,r in panel.iterrows():
        if not r.feature_ready:continue
        sides=[labels[key(at,s)] for s in core.ETFS]
        later=market.sessions[market.sessions>r.date]
        nextdates=[later[0] if len(later) else None]+[next_cycle(market,x) for x in sides]
        record={'at':at,'date':r.date,'ready_at':None,'next_at':[],'next_date':nextdates,'duration':[],
                'reward':[0.]+[float(np.log1p(x.get('net_return') if x.get('net_return') is not None else np.nan)) for x in sides]}
        ready=pd.notna(r.terminal_release)
        for d in nextdates:
            nxt=by_date.get(d);valid=nxt is not None and panel.loc[nxt,'feature_ready']
            record['next_at'].append(nxt if valid else None)
            record['duration'].append(int((market.sessions>r.date).sum()-(market.sessions>d).sum()) if d is not None else None)
            ready=ready and valid
        if ready and np.isfinite(record['reward']).all():record['ready_at']=max(r.terminal_release,*record['next_at'])
        # A known contractual bound detects missing historical transitions, but
        # a normal not-yet-paid dividend is allowed to wait until its pay date.
        expiry=sides[0].get('expiry_date',r.date+pd.Timedelta(days=13));bound=market.sessions[market.sessions>expiry]
        deadline=bound[0] if len(bound) else None
        if deadline is not None:
            for symbol in core.ETFS:
                for event in market.actions[symbol]:
                    ex=pd.Timestamp(event['date'])
                    if event['type']=='CashDividends' and r.date<ex<=expiry:
                        pay=pd.Timestamp(event['payment_date'])
                        deadline=max(deadline,pay)
            future=market.sessions[market.sessions>=deadline]
            deadline=future[0] if len(future) else None
        record['expired_after']=deadline
        rows.append(record)
    return rows


def admitted(bank,cutoff):
    day=cutoff.tz_convert(core.ET).tz_localize(None).normalize();prior=[r for r in bank if r['date']<day and r['at']<cutoff]
    ids=[]
    for r in prior:
        if r['ready_at'] is None or r['ready_at']>=cutoff:break
        ids.append(r['at'])
    blocked=prior[len(ids)] if len(ids)<len(prior) else None
    expired=bool(blocked and blocked['expired_after'] is not None and blocked['expired_after']<day)
    return ids,{'prior_dates':len(prior),'admitted_dates':len(ids),'first_pending':blocked['date'] if blocked else None,
        'known_behind_pending':sum(r['ready_at'] is not None and r['ready_at']<cutoff for r in prior[len(ids)+1:]),
        'expired_unresolved':expired}


def fit(panel,bank,ids):
    records={r['at']:r for r in bank};n=len(ids);w=weights(n,126)
    info={'ready':n>=120,'members':ids,'dates':list(panel.loc[ids,'date']),'weights':w.tolist(),'columns':COLUMNS,'stages':[]}
    if n<120:return None,info
    X=np.ascontiguousarray(panel.loc[ids,COLUMNS].to_numpy(float));R=np.array([records[t]['reward'] for t in ids]);D=np.array([records[t]['duration'] for t in ids])
    N=np.array([[panel.loc[a,COLUMNS].to_numpy(float) for a in records[t]['next_at']] for t in ids])
    assert np.isfinite(X).all() and np.isfinite(N).all() and np.isfinite(R).all() and (D>=1).all()
    scaler=WeightedScale(X,w);Z=scaler.transform(X);ZN=scaler.transform(N.reshape(-1,len(COLUMNS)))
    V=np.zeros_like(R);models=[]
    for iteration in range(1,ITERATIONS+1):
        target=(GAMMA**D)*(R+V)
        model=Ridge(alpha=100.,solver='svd').fit(Z,target,sample_weight=w);models.append(model)
        info['stages'].append({'iteration':iteration,'coefficients':model.coef_.tolist(),'intercept':model.intercept_.tolist(),
            'target_min':float(target.min()),'target_max':float(target.max())})
        V=np.maximum(0.,model.predict(ZN).max(axis=1)).reshape(n,3)
        assert np.isfinite(V).all()
    info.update(center=scaler.mean_.tolist(),scale=scaler.scale_.tolist())
    return {'scaler':scaler,'stages':models},info


def raw_prediction(model,row):
    if model is None or not row.feature_ready:return None
    Z=model['scaler'].transform(row[COLUMNS].to_numpy(float)[None,:])
    return {'planning':model['stages'][-1].predict(Z)[0].tolist(),'myopic':model['stages'][0].predict(Z)[0].tolist()}


def calibrate(bank,ids,history,mode):
    records={r['at']:r for r in bank};eligible=set(ids);old=[]
    for at in sorted(history):
        if at not in eligible or history[at] is None:continue
        rec=records[at]
        if mode=='planning' and any(a not in history or history[a] is None for a in rec['next_at']):continue
        old.append(at)
    old=old[-120:];residuals=[]
    for at in old:
        rec=records[at];reward=np.array(rec['reward']);duration=np.array(rec['duration'])
        future=np.array([max(0.,*history[a][mode]) for a in rec['next_at']]) if mode=='planning' else np.zeros(3)
        target=GAMMA**duration*(reward+future)
        residuals.append(target-np.array(history[at][mode]))
    w=weights(len(old),63)
    correction=np.average(residuals,axis=0,weights=w).tolist() if old else None
    return {'ready':len(old)>=60,'members':old,'weights':w.tolist(),'correction':correction,
            'next_forecast_members':[records[t]['next_at'] for t in old] if mode=='planning' else []}


def decide(raw,cal,mode,feedback,uncalibrated=False):
    out={'ready':False,'action':'CASH','allocation':0.,'raw':raw[mode] if raw else None,'adjusted':None}
    if raw is None or not cal['ready'] or feedback['expired_unresolved']:return out
    values=np.array(raw[mode])+(0 if uncalibrated else np.array(cal['correction']))
    wait=max(0.,float(values[0]));j=1 if values[1]>=values[2] else 2
    buy=values[j]>wait+1e-12
    out.update(ready=True,action=ACTIONS[j] if buy else 'CASH',allocation=float(buy),adjusted=values.tolist(),
               waiting_value=wait,buy_advantage=float(values[j]-wait))
    return out
