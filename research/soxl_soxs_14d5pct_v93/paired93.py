"""Learn both mature ETF candidates and compare their as-issued expected values."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v92'))
from bounded92 import *
import bounded92 as previous
import calibration91 as calibration
SOURCE=Path(__file__).resolve().parent
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v93'
PREVIOUS=ROOT/'.cache/soxl_soxs_14d5pct_v92/runs/development'
PAIRED_MODELS=['fundamental','reaction']
POLICIES={'paired_fundamental':('fundamental','reselect'),
    'paired_reaction':('reaction','reselect'),
    'paired_raw':('raw_fundamental','reselect'),
    'paired_fixed_top':('fundamental','fixed_top'),
    'paired_fixed_source':('fundamental','fixed_source'),
    'paired_identity':('identity','reselect')}
ROUTES=list(POLICIES)+previous.ROUTES+['paired_premarket_only','paired_double_cost']


def verify_parent():
    hashes=previous.verify_parent();m=read(previous.SOURCE/'evidence/delivery_manifest.json')
    for f,h in m['delivery_files_sha256'].items():assert sha(previous.SOURCE/f)==h,f
    for f,h in m['cache_files_sha256'].items():assert sha(PREVIOUS/f)==h,f
    v=read(PREVIOUS/'verification.json');assert v['passed'] and v['results_sha256']==sha(PREVIOUS/'results.json')
    hashes['92']=sha(previous.SOURCE/'evidence/delivery_manifest.json');return hashes


def paired_samples(panel,parents,history):
    rows=[];masses=[];dates=[];n=len(parents)
    for i,p in enumerate(parents):
        day=[]
        for at in p['members']:
            at=pd.Timestamp(at);e=history[at]['events']['recent_ratio'];r=panel.loc[at]
            assert e['ready'] and r.calendar_known and r.fundamental_known
            for j in range(2):
                y=int(r['L_gap' if j==0 else 'S_gap']);assert 0<=y<4
                day.append((at,j,e['probabilities'][j],y,state(r,j),CLOCKS.index(r.clock)))
        if day:
            mass=2.**((i-n+1)/63);masses.append(mass);dates.append(p['date'])
            rows.extend([(*r,mass/len(day)) for r in day])
    b=np.asarray(masses);ess=float(b.sum()**2/(b@b)) if len(b) else 0.
    w=np.array([r[-1] for r in rows]);w*=ess/b.sum() if len(b) else 1.
    meta=dict(dates=dates,real_dates=len(dates),date_ess=ess,members=[[r[0],r[1]] for r in rows],weights=w)
    return (np.array([r[2] for r in rows]).reshape(-1,4),np.array([r[3] for r in rows],int),w,
        np.array([r[4] for r in rows]).reshape(-1,19),np.array([r[1] for r in rows],int),np.array([r[5] for r in rows],int),meta)


def context_groups(model,arrays):
    P,y,w,X,sides,clocks,meta=arrays;Z,mu,sd=calibration.design(X,sides,clocks,w,model['mode'])
    scores=centered(Z@model['beta']);groups={}
    for side in range(2):
        for clock in range(8):
            ids=np.flatnonzero((sides==side)&(clocks==clock));members=[meta['members'][i] for i in ids]
            dates=sorted({str(pd.Timestamp(t).tz_convert(core.ET).date()) for t,j in members})
            groups[f'{side}:{clock}']=dict(corrections=scores[ids],members=members,dates=dates)
    return groups


def daily_models(panel,parents,feedback,history,at,baseline_ready):
    chosen=[p for p in parents if all(t in history and history[t]['events']['recent_ratio']['ready'] for t in p['members'])][-120:]
    eligible=calibration.samples(panel,chosen,history,'eligible')[-1]
    arrays=paired_samples(panel,chosen,history);P,y,w,X,sides,clocks,meta=arrays
    population_ready=len(chosen)>=60 and eligible['real_dates']>=30
    models={mode:{**meta,**(calibration.fit(P,y,w,X,sides,clocks,mode) if population_ready else dict(ready=False,reason='insufficient_dates'))} for mode in PAIRED_MODELS}
    ready=population_ready and baseline_ready and all(m['ready'] for m in models.values()) and not feedback['expired_unresolved_prefix']
    groups={mode:context_groups(m,arrays) for mode,m in models.items()} if ready else {}
    return dict(at=at,cutoff=day_cutoff(at),parents=chosen,feedback=feedback,eligible_dates=eligible['real_dates'],
        population_ready=population_ready,baseline_ready=baseline_ready,ready=ready,models=models,supports=groups)


def estimates(row,daily,src,payoffs):
    if not (daily['ready'] and src['ready'] and row.calendar_known and row.fundamental_known) or payoffs is None:return {}
    clock=CLOCKS.index(row.clock);out={};P=np.asarray(src['probabilities']);means=np.asarray(payoffs)
    out['identity']=dict(probabilities=P,values=np.sum(P*means,axis=1),outside=[[],[]])
    for mode in PAIRED_MODELS:
        model=daily['models'][mode];bounded=[];raw=[];proofs=[];outside=[]
        for side in range(2):
            x=state(row,side);group=daily['supports'][mode][f'{side}:{clock}']
            info=project(raw_correction(model,x,side,clock),group['corrections'])
            info['context_dates']=len(group['dates']);proofs.append(info)
            raw.append(apply_correction(P[side],info['raw']));bounded.append(apply_correction(P[side],info['correction']))
            outside.append([STATE_FIELDS[k] for k in INDICES[mode] if x[k]<model['input_min'][k]-1e-12 or x[k]>model['input_max'][k]+1e-12])
        Q=np.array(bounded);out[mode]=dict(probabilities=Q,values=np.sum(Q*means,axis=1),projections=proofs,outside=outside)
        if mode=='fundamental':
            Q=np.array(raw);out['raw_fundamental']=dict(probabilities=Q,values=np.sum(Q*means,axis=1),outside=outside)
    return out


def decision(est,src,policy):
    if policy not in ['reselect','fixed_top','fixed_source']:raise ValueError('Unknown selection rule')
    j=int(est['values'][1]>est['values'][0]) if policy=='reselect' else core.ETFS.index(src['top_symbol'])
    buy=est['values'][j]>1e-12 and (policy!='fixed_source' or src['action'] in core.ETFS)
    return dict(ready=True,action=core.ETFS[j] if buy else 'CASH',allocation=float(buy),
        probabilities=est['probabilities'][j],value=float(est['values'][j]),top_symbol=core.ETFS[j],
        source_eligible=src['action'] in core.ETFS,source_top_symbol=src['top_symbol'],
        source_value=float(src['values'][j]),outside_state_fields=est['outside'][j])


def issue(at,row,daily,old,base,previous_events):
    src=old['events']['recent_ratio'];gap=base['distributions']['gap']
    pairs=estimates(row,daily,src,gap['payoff_means'] if gap is not None else None)
    header=dict(at=at,date=row.date,clock=row.clock);out=dict(previous_events)
    for name,(model,policy) in POLICIES.items():
        e=decision(pairs[model],src,policy) if pairs else dict(ready=False,action='CASH',allocation=0.,probabilities=None,value=None)
        out[name]={**header,**e}
    out['paired_premarket_only']=dict(out['paired_fundamental']) if row.clock==CLOCKS[0] else out['cash']
    out['paired_double_cost']=dict(out['paired_fundamental'])
    return out,pairs
