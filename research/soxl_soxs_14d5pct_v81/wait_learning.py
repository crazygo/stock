"""Causal paired immediate-versus-next-session policy returns, with original execution."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v80'))
from shift_learning import *
import shift_learning as parent
SOURCE=Path(__file__).resolve().parent;CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v81'
PARENT=ROOT/'.cache/soxl_soxs_14d5pct_v80/runs/development'
VARIANTS=['state','score','mean']
CONTROL_MAP={'source_recent':'recent_ratio','source_lagged':'lagged_ratio','source_uniform':'uniform_top','source_blend':'source_blend'}
ROUTES=['paired_state','paired_score','paired_mean','absolute_pair','absolute_now','source_matched']+list(CONTROL_MAP)+['premarket_only','cash','double_cost']


def verify_parent():
    parent.verify_parent();folder=ROOT/'research/soxl_soxs_14d5pct_v80';m=read(folder/'evidence/delivery_manifest.json')
    for f,h in m['delivery_files_sha256'].items():assert sha(folder/f)==h,f
    for f,h in m['cache_files_sha256'].items():assert sha(PARENT/f)==h,f
    for f,h in m['input_files_sha256'].items():assert sha(ROOT/m['input_root']/f)==h,f
    assert read(PARENT/'verification.json')['passed'];return sha(folder/'evidence/delivery_manifest.json')


def target_registry(panel,market,records):
    reg=registry(panel,market);lookup={p['date']:p for p in reg};out=[]
    for p in reg:
        if not p['feature_ready'] or not all(records[t]['events']['recent_ratio']['ready'] for t in p['members']):continue
        future=market.sessions[market.sessions>p['date']];nxt=future[0] if len(future) else None
        next_at=pd.Timestamp(str(nxt.date())+' '+CLOCKS[0],tz=core.ET).tz_convert('UTC') if nxt is not None else None
        other=lookup.get(nxt);okay=other is not None and other['feature_ready'] and next_at in records and records[next_at]['events']['recent_ratio']['ready']
        released=okay and p['release'] is not None and other['release'] is not None
        ready_at=max(p['release'],other['release'],next_at) if released else None
        out.append({'date':p['date'],'members':p['members'],'next_date':nxt,'next_at':next_at,'release':ready_at,
            'expired_after':nxt+pd.Timedelta(days=14) if nxt is not None else None})
    return out


def admitted(bank,at):
    cutoff=day_cutoff(at);today=cutoff.tz_convert(core.ET).tz_localize(None).normalize();past=[p for p in bank if p['date']<today];chosen=[]
    for p in past:
        if p['release'] is None or p['release']>=cutoff:break
        chosen.append(p)
    pending=past[len(chosen)] if len(chosen)<len(past) else None
    return chosen[-120:],{'cutoff':cutoff,'mature_prefix_dates':len(chosen),'first_pending':pending['date'] if pending else None,
        'expired_unresolved_prefix':bool(pending and pending['expired_after'] is not None and pending['expired_after']<=today)}


def inputs(row,event):
    assert event['ready'];j=core.ETFS.index(event['top_symbol']);values=event['values']
    return np.r_[signed_price(row[FIELDS].to_numpy(float),1 if j==0 else -1),float(j),values[j],values[1-j]],j


def arrays(panel,parents,records):
    X=[];Y=[];clocks=[];members=[];masses=[]
    n=len(parents);dw=2.**(-np.arange(n)[::-1]/63.);ess=float(dw.sum()**2/(dw@dw)) if n else 0.;dw=dw*(ess/dw.sum()) if n else dw
    for i,p in enumerate(parents):
        future=records[p['next_at']]['events']['recent_ratio'];symbol=future['action']
        wait=0. if symbol=='CASH' else float(panel.loc[p['next_at'],'L_log' if symbol=='SOXL' else 'S_log'])
        assert np.isfinite(wait)
        for t in p['members']:
            row=panel.loc[t];e=records[t]['events']['recent_ratio'];x,j=inputs(row,e);now=float(row['L_log' if j==0 else 'S_log']);assert np.isfinite(now)
            X.append(x);Y.append([now,wait]);clocks.append(CLOCKS.index(row.clock));masses.append(dw[i]/len(p['members']))
            members.append({'at':t,'side':j,'wait_at':p['next_at'],'wait_action':symbol})
    return np.array(X).reshape(-1,16),np.array(Y).reshape(-1,2),np.array(clocks,int),np.array(masses),{
        'members':members,'dates':[p['date'] for p in parents],'date_ess':ess,'real_dates':n,'date_weights':dw}


def fit(X,Y,clocks,w,mode):
    X=np.asarray(X,float);Y=np.asarray(Y,float);w=np.asarray(w,float);clocks=np.asarray(clocks,int)
    if X.shape!=(len(w),16) or Y.shape!=(len(w),2) or clocks.shape!=w.shape or not len(w):raise ValueError('Aligned nonempty pairs required')
    if not all(np.isfinite(v).all() for v in [X,Y,w]) or (w<=0).any() or not set(clocks)<=set(range(8)):raise ValueError('Finite paired labels and positive weights required')
    ids=list(range(16)) if mode=='state' else [13,14,15] if mode=='score' else [] if mode=='mean' else None
    if ids is None:raise ValueError('Unknown model')
    selected=X[:,ids];center=np.average(selected,axis=0,weights=w);scale=np.sqrt(np.average((selected-center)**2,axis=0,weights=w));scale[scale<1e-12]=1.
    A=np.c_[np.eye(8)[clocks],(selected-center)/scale] if mode!='mean' else np.zeros((len(w),0))
    acenter=np.average(A,axis=0,weights=w);b=np.average(Y,axis=0,weights=w);Z=A-acenter;penalty=np.r_[np.full(8,10.),np.full(len(ids),100.)] if mode!='mean' else np.zeros(0)
    augmented=np.vstack([Z*np.sqrt(w)[:,None],np.diag(np.sqrt(penalty))]);target=np.vstack([(Y-b)*np.sqrt(w)[:,None],np.zeros((len(penalty),2))])
    coef=np.linalg.lstsq(augmented,target,rcond=None)[0] if len(penalty) else np.zeros((0,2))
    return {'mode':mode,'ids':ids,'center':center,'scale':scale,'design_center':acenter,'intercept':b,'coefficients':coef,'penalty':penalty,'weights':w,'rows':len(w)}


def predict(model,x,clock):
    ids=model['ids'];a=np.r_[np.eye(8)[clock],(np.asarray(x)[ids]-model['center'])/model['scale']] if model['mode']!='mean' else np.zeros(0)
    return np.asarray(model['intercept'])+(a-model['design_center'])@np.asarray(model['coefficients']).reshape(-1,2)


def daily_models(panel,bank,records,at):
    parents,feedback=admitted(bank,at);ready=len(parents)>=60 and not feedback['expired_unresolved_prefix'];models={};meta={}
    if ready:
        X,Y,clocks,w,meta=arrays(panel,parents,records)
        models={mode:fit(X,Y,clocks,w,mode) for mode in VARIANTS}
    return {'at':at,'cutoff':day_cutoff(at),'parents':parents,'feedback':feedback,'ready':ready,'models':models,'metadata':meta}


def issue(at,row,daily,old):
    src=old['events']['recent_ratio'];ready=daily['ready'] and src['ready'];base={'at':at,'date':row.date,'clock':row.clock}
    cash={**base,'ready':True,'action':'CASH','allocation':0.,'values':None};inactive={**cash,'ready':False};out={}
    for mode in VARIANTS:
        e=dict(inactive)
        if ready:
            x,j=inputs(row,src);values=predict(daily['models'][mode],x,CLOCKS.index(row.clock));delta=float(values[0]-values[1])
            buy=src['action'] in core.ETFS and delta>1e-12
            e={**base,'ready':True,'action':core.ETFS[j] if buy else 'CASH','allocation':float(buy),'values':values,
                'advantage':delta,'top_symbol':src['top_symbol'],'source_eligible':src['action'] in core.ETFS,'source_value':src['values'][j]}
        out['paired_'+mode]=e
    for name,compare in [('absolute_pair',True),('absolute_now',False)]:
        e=dict(out['paired_state'])
        if ready:
            buy=e['source_eligible'] and e['values'][0]>1e-12 and (e['advantage']>1e-12 or not compare)
            e.update(action=e['top_symbol'] if buy else 'CASH',allocation=float(buy))
        out[name]=e
    out['source_matched']=src if ready else inactive
    for name,source in CONTROL_MAP.items():out[name]=old['events'][source]
    out['cash']=cash;out['double_cost']=dict(out['paired_state']);out['premarket_only']=dict(out['paired_state']) if row.clock==CLOCKS[0] else cash
    return out
