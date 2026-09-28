"""Evaluate the frozen discovery rules once on disjoint earlier dates."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def apply_rule(frame, rule):
    available=np.ones(len(frame),dtype=bool)
    hit=np.ones(len(frame),dtype=bool)
    for field,op,value in rule['atoms']:
        x=frame[field]
        available &= x.notna().to_numpy()
        if op=='>=':hit &= x.ge(value).to_numpy()
        elif op=='<=':hit &= x.le(value).to_numpy()
        elif op=='>':hit &= x.gt(value).to_numpy()
        elif op=='<':hit &= x.lt(value).to_numpy()
        else:raise ValueError(op)
    return available, hit & available


def wilson(k,n):
    if not n:return [None,None]
    z=1.95996398454;p=k/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half),float(center+half)]


def cluster_ci(d, target, hit, seed=260926):
    a=d[['date',target]].copy();a['hit']=hit
    rows=[]
    for date,g in a.groupby('date'):
        rows.append([len(g),g[target].sum(),g.hit.sum(),g.loc[g.hit,target].sum()])
    x=np.array(rows,float)
    if not len(x) or x[:,2].sum()==0:return {'rate':[None,None],'date_matched_lift':[None,None],'difference':[None,None]}
    rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(x),size=(4000,len(x)))
    sample=x[idx]
    n=sample[:,:,2].sum(1);k=sample[:,:,3].sum(1)
    expected=(sample[:,:,2]*(sample[:,:,1]/sample[:,:,0])).sum(1)
    keep=n>0;ratio_keep=keep&(expected>0)
    return {'rate':np.quantile(k[keep]/n[keep],[.025,.975]).tolist(),
            'date_matched_lift':np.quantile(k[ratio_keep]/expected[ratio_keep],[.025,.975]).tolist() if ratio_keep.any() else [None,None],
            'difference':np.quantile((k[keep]-expected[keep])/n[keep],[.025,.975]).tolist()}


def summarize(frame,rule,window,target):
    available,hit=apply_rule(frame,rule)
    d=frame.loc[available].reset_index(drop=True);h=hit[available]
    y=d[target].to_numpy(float)
    s=d.loc[h]
    n=len(s);k=int(s[target].sum());events=int(y.sum());baseline=float(y.mean()) if len(y) else None
    same_date=d.groupby('date')[target].transform('mean').to_numpy()
    same_stock=d.groupby('symbol')[target].transform('mean').to_numpy()
    same_sector_date=d.groupby(['date','sector'])[target].transform('mean').to_numpy()
    expected=float(same_date[h].sum());stock_expected=float(same_stock[h].sum());sector_expected=float(same_sector_date[h].sum())
    failures=s[s[target].eq(0)].sort_values('close_return').head(3)
    out={'window':window,'rule':rule['id'],'target':target,'start':frame.date.min(),'end':frame.date.max(),
         'eligible':len(frame),'available':len(d),'missing_features':int((~available).sum()),'n':n,'k':k,
         'trigger_dates':int(s.date.nunique()),'symbols':int(s.symbol.nunique()),'rate':k/n if n else None,
         'baseline':baseline,'lift':k/n/baseline if n and baseline else None,
         'nontrigger_rate':float(d.loc[~h,target].mean()) if (~h).any() else None,
         'recall_available':k/events if events else None,'recall_all':k/frame[target].sum() if frame[target].sum() else None,
         'false_positives':n-k,'date_expected':expected,'date_lift':k/expected if expected else None,
         'stock_expected':stock_expected,'stock_lift':k/stock_expected if stock_expected else None,
         'sector_date_expected':sector_expected,'sector_date_lift':k/sector_expected if sector_expected else None,
         'wilson95':wilson(k,n),'date_bootstrap95':cluster_ci(d,target,h),
         'mean_open_close':float(s.open_close.mean()) if n else None,
         'median_open_close':float(s.open_close.median()) if n else None,
         'open_close_negative':float(s.open_close.lt(0).mean()) if n else None,
         'open_low_q10':float(s.open_low.quantile(.1)) if n else None,
         'open_high5_rate':float(s.openhigh5.mean()) if n else None,
         'failures':failures[['date','symbol','close_return','open_close']].to_dict('records')}
    return out


def permutation_family(frame,rules,target,iterations=4000):
    """Conditional randomization within date x sector; controls same-day sector surges.

    Full labels permuted, then each fixed rule applies its own availability mask.
    Not a causal test; per-symbol identity remains a separate descriptive control.
    """
    d=frame.reset_index(drop=True)
    h=np.stack([apply_rule(d,r)[1] for r in rules]).astype(float)
    y=d[target].to_numpy(float)
    observed=h@y
    groups=[np.asarray(idx) for idx in d.groupby(['date','sector']).indices.values()]
    rng=np.random.default_rng(926)
    exceed=np.zeros(len(rules),int)
    for _ in range(iterations):
        shuffled=y.copy()
        for idx in groups:
            if len(idx)>1 and y[idx].min()!=y[idx].max():
                shuffled[idx]=rng.permutation(y[idx])
        exceed += (h@shuffled >= observed-1e-9)
    p=(exceed+1)/(iterations+1)
    order=np.argsort(p);adj=np.zeros(len(p));maximum=0
    for i,j in enumerate(order):
        maximum=max(maximum,p[j]*(len(p)-i));adj[j]=min(1,maximum)
    return {r['id']:{'p':float(p[i]),'holm_p':float(adj[i]),'null':'outcome exchangeability within date x sector'} for i,r in enumerate(rules)}


def main():
    f=HERE/'frozen_rules.json';frozen=json.loads(f.read_text());rules=frozen['rules']
    panel=pd.read_parquet(HERE/'panel.parquet')
    panel=panel[panel.valid_label].copy()
    # Sep 18 was explicitly inspected as discovery T-1; do not reuse its outcome as unseen validation.
    dates=[s['session_date'] for s in json.loads((ROOT/'market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())['sessions'] if s['session_date']<'2026-09-18']
    windows={'discovery_5':['2026-09-21','2026-09-22','2026-09-23','2026-09-24','2026-09-25'],
             'prior_5':dates[-5:],'prior_15':dates[-20:-5],'prior_40':dates[-60:-20],'prior_60':dates[-60:]}
    result=[]
    for name,days in windows.items():
        d=panel[panel.date.isin(days)]
        for rule in rules:
            for target in ['close8','high10']:
                result.append(summarize(d,rule,name,target))
    earlier=panel[panel.date.isin(windows['prior_60'])]
    p=permutation_family(earlier,rules,'close8')
    # Multiple-testing family covers all frozen primary-label rules; secondary target descriptive only.
    for r in result:
        if r['window']=='prior_60' and r['target']=='close8':r.update(p[r['rule']])
    summary={'frozen_rules_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'windows':windows,'results':result,
             'primary_test':'one-sided, 4000 date-sector conditional permutations, Holm across all frozen rules',
             'secondary_test':'high10 descriptive; no independent significance claim',
             'direction':'reverse-chronology historical transfer; not prospective holdout'}
    (HERE/'validation.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))
    pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in result]).to_csv(HERE/'validation.csv',index=False)
    sectors=[]
    for name in ['discovery_5','prior_5','prior_15','prior_40','prior_60']:
        d=panel[panel.date.isin(windows[name])]
        for sector,g in d.groupby('sector'):
            sectors.append({'window':name,'sector':sector,'n':len(g),'k':int(g.close8.sum()),'rate':float(g.close8.mean()),
                            'symbols':int(g.symbol.nunique()),'event_dates':int(g.loc[g.close8.eq(1),'date'].nunique()),
                            'high10':int(g.high10.sum())})
    pd.DataFrame(sectors).to_csv(HERE/'sectors.csv',index=False)
    triggers=[]
    for rule in rules:
        _,mask=apply_rule(panel,rule)
        a=panel.loc[mask].copy();a['rule']=rule['id'];triggers.append(a)
    pd.concat(triggers).to_csv(HERE/'all_triggers.csv',index=False)
    latest=pd.read_parquet(HERE/'daily.parquet')
    latest=latest[latest.date.eq('2026-09-25') & latest.symbol.ne('QQQ') & latest.rth_valid].copy()
    fields={c:'prev_'+c for c in latest.columns if c not in ['symbol','date']}
    latest.rename(columns=fields,inplace=True)
    nextday=[]
    for rule in rules:
        _,mask=apply_rule(latest,rule)
        for _,row in latest.loc[mask].iterrows():
            nextday.append({'observation_date':'2026-09-25','target_date':'2026-09-28','symbol':row.symbol,
                            'rule':rule['id'],'prev_cc':row.prev_cc,'prev_relvol':row.prev_relvol,
                            'prev_location':row.prev_location,'prev_last60':row.prev_last60,
                            'prev_post_return':row.prev_post_return,'status':'unvalidated_research_candidate'})
    pd.DataFrame(nextday).to_csv(HERE/'next_session_watchlist.csv',index=False)
    print(pd.DataFrame(result).query("target=='close8'")[['window','rule','n','k','rate','baseline','date_lift','stock_lift','sector_date_lift','holm_p']].round(4).to_string(index=False))
    print(pd.DataFrame(sectors).query("window in ['discovery_5','prior_60']").to_string(index=False))


if __name__=='__main__':main()
