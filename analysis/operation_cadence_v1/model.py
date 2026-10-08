"""Causal features, next-open labels, portable calibrated logistic probabilities."""
from datetime import date, timedelta
import importlib.util
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

VERSION = 'operation_cadence_v1.0'
FEATURES = ['G','V','M','J','S','return20','efficiency20','efficiency60','turns20',
            'vol_ratio','drawdown60','log_dollar_volume20','return252','jump252']
TARGETS = {
    'cycle': {'name':'往返波动','horizon':20,'description':'20日内至少两轮5%确认往返；环境机会，不是利润'},
    'quick': {'name':'短线先赢','horizon':10,'description':'10日内收盘净+5%先于净−5%；超时算未达成'},
    'wave': {'name':'波段上行','horizon':20,'description':'20日末净+5%，收盘最大回撤≤10%'},
    'hold': {'name':'持有上行','horizon':60,'description':'60日末净+10%，收盘最大回撤≤15%'},
    'risk': {'name':'下行压力','horizon':20,'description':'20日内收盘相对起点净跌10%'}
}
spec=importlib.util.spec_from_file_location('five_traits',Path(__file__).resolve().parents[1]/'ai_trend_quadrant_v2/model.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

HOLIDAYS = set('''2024-01-01 2024-01-15 2024-02-19 2024-03-29 2024-05-27 2024-06-19 2024-07-04 2024-09-02 2024-11-28 2024-12-25
2025-01-01 2025-01-09 2025-01-20 2025-02-17 2025-04-18 2025-05-26 2025-06-19 2025-07-04 2025-09-01 2025-11-27 2025-12-25
2026-01-01 2026-01-19 2026-02-16 2026-04-03 2026-05-25 2026-06-19 2026-07-03 2026-09-07 2026-11-26 2026-12-25
2027-01-01 2027-01-18 2027-02-15 2027-03-26'''.split())
EARLY = ['2024-07-03','2024-11-29','2024-12-24','2025-07-03','2025-11-28','2025-12-24','2026-11-27','2026-12-24']
def calendar(start='2024-01-01',end='2027-03-31'):
    d=date.fromisoformat(start);out=[]
    while d.isoformat()<=end:
        if d.weekday()<5 and d.isoformat() not in HOLIDAYS: out.append(d.isoformat())
        d+=timedelta(days=1)
    return out

def turns(prices, threshold=.05):
    """Only confirmed directional changes count; final unconfirmed extremum is absent."""
    p=np.asarray(prices,float)
    if len(p)<2: return []
    lo=hi=float(p[0]);direction=0;out=[]; extreme_index=0
    for i,x in enumerate(p[1:],1):
        if direction==0:
            lo=min(lo,x);hi=max(hi,x)
            if x>=lo*(1+threshold):direction=1;hi=x;extreme_index=i;out.append({'i':i,'direction':1,'price':float(x)})
            elif x<=hi*(1-threshold):direction=-1;lo=x;extreme_index=i;out.append({'i':i,'direction':-1,'price':float(x)})
        elif direction==1:
            if x>hi:hi=x;extreme_index=i
            elif x<=hi*(1-threshold):direction=-1;lo=x;out.append({'i':i,'direction':-1,'price':float(x)})
        else:
            if x<lo:lo=x;extreme_index=i
            elif x>=lo*(1+threshold):direction=1;hi=x;out.append({'i':i,'direction':1,'price':float(x)})
    return out

def efficiency(p):
    r=np.diff(np.log(p));den=np.abs(r).sum()
    return float(r.sum()/den) if den>1e-12 else 0.
def dd(p):
    p=np.asarray(p,float);return float(np.max(1-p/np.maximum.accumulate(p)))

def feature(bars,idx,sessions):
    if idx<140:return None,'insufficient_141_closes'
    b=bars[max(0,idx-252):idx+1];days=[x[0] for x in b]
    needed=[s for s in sessions if days[-141]<=s<=days[-1]]
    if days[-141:]!=needed:return None,'missing_feature_sessions'
    c=np.array([x[4] for x in b]);r=np.diff(np.log(c));five=old.measure(r)[0]
    if not np.isfinite(five).all():return None,'undefined_five_traits'
    sd=np.std(r[-60:],ddof=1);short=np.std(r[-20:],ddof=1)
    volumes=np.array([x[5] for x in b[-20:]],float)
    money=np.median(c[-20:]*volumes)
    yr=None;jr=None
    if len(c)>=253 and days==[s for s in sessions if days[0]<=s<=days[-1]]:
        yr=float(c[-1]/c[-253]-1);a=r[-252:];e=(a-a.mean())**2
        jr=float(np.sort(e)[-26:].sum()/e.sum()) if e.sum()>0 else None
    values=[*map(float,five),float(c[-1]/c[-21]-1),efficiency(c[-21:]),efficiency(c[-61:]),
       len(turns(c[-21:])),float(short/sd) if sd>0 else None,dd(c[-61:]),float(np.log10(money)) if money>0 else None,yr,jr]
    return values,'scored'

def label(bars,idx,key,sessions,cutoff):
    h=TARGETS[key]['horizon'];t=bars[idx][0];si=sessions.index(t);expected=sessions[si+1:si+h+1]
    if len(expected)<h:return {'status':'calendar_incomplete','y':None}
    maturity=expected[-1]
    if maturity>cutoff:return {'status':'pending','y':None,'maturity':maturity}
    b=bars[idx+1:idx+h+1]
    if [x[0] for x in b]!=expected:return {'status':'missing_label_sessions','y':None,'maturity':maturity}
    entry=b[0][1];c=np.array([x[4] for x in b]);net=(c/entry)*.998-1
    path=np.r_[entry,c];draw=dd(path);ev=None
    if key=='cycle':
        ev=turns(path);y=int(len(ev)>=4)
    elif key=='quick':
        hits=np.flatnonzero((net>=.05)|(net<=-.05));first=int(hits[0]) if len(hits) else None
        y=int(first is not None and net[first]>=.05)
    elif key=='wave':y=int(net[-1]>=.05 and draw<=.10)
    elif key=='hold':y=int(net[-1]>=.10 and draw<=.15)
    else:y=int(net.min()<=-.10)
    return {'status':'mature','y':y,'maturity':maturity,'entry_day':b[0][0],'entry_open':float(entry),
         'end_net':float(net[-1]),'max_drawdown':draw,'min_net':float(net.min()),
         'turn_count':len(ev) if ev is not None else None}

def dateweights(rows):
    from collections import Counter
    counts=Counter(r['day'] for r in rows)
    return np.array([len(rows)/(len(counts)*counts[r['day']]) for r in rows])

def fit(x,y,w):
    x=np.asarray(x,float);med=np.array([np.nanmedian(a) if np.isfinite(a).any() else 0. for a in x.T])
    v=np.concatenate([np.where(np.isfinite(x),x,med),~np.isfinite(x)],axis=1).astype(float)
    mean=np.average(v,axis=0,weights=w);scale=np.sqrt(np.average((v-mean)**2,axis=0,weights=w));scale=np.where(scale>1e-12,scale,1.)
    lr=LogisticRegression(C=.3,max_iter=1500,solver='lbfgs').fit((v-mean)/scale,y,sample_weight=w)
    return {'medians':med.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),'coef':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0])}
def transform(m,x):
    x=np.asarray(x,float)
    if x.ndim==1:x=x[None,:]
    v=np.concatenate([np.where(np.isfinite(x),x,m['medians']),~np.isfinite(x)],axis=1).astype(float)
    return (v-np.array(m['mean']))/np.array(m['scale'])
def logit(m,x):return transform(m,x)@np.array(m['coef'])+m['intercept']
def predict(m,x):
    if m.get('constant') is not None:return np.full(len(np.atleast_2d(x)),m['constant'])
    z=logit(m,x);c=m['calibrator'];return expit(c['slope']*z+c['intercept'])
def calibrate(m,x,y,w):
    lr=LogisticRegression(C=.1,max_iter=1000).fit(logit(m,x).reshape(-1,1),y,sample_weight=w)
    m['calibrator']={'slope':float(lr.coef_[0,0]),'intercept':float(lr.intercept_[0])}
    return m

def cadence(p):
    if any(p.get(k) is None for k in TARGETS):return '覆盖不足'
    calm=p['risk']<=.30;slow=p['hold']>=.60 and calm
    quick=p['quick']>=.60 and p['cycle']>=.40 and calm
    if slow and quick:return '底仓＋波段'
    if slow:return '20–60日审查'
    if quick:return '2–10日审查'
    if p['wave']>=.60 and calm:return '10–20日审查'
    return '观察／低置信度'
