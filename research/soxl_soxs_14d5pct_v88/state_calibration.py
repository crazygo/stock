"""Daily state-conditional correction of historically issued top-class probabilities."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v80'))
from shift_learning import *
from event_calendar import CATALOG, features as event_features
from reaction_inputs import prepare_reactions, verify_prices
import shift_learning as parent
SOURCE=Path(__file__).resolve().parent
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v88'
PARENT=ROOT/'.cache/soxl_soxs_14d5pct_v80/runs/development'
BASE=ROOT/'.cache/soxl_soxs_14d5pct_v78/runs/development'
WAIT=ROOT/'.cache/soxl_soxs_14d5pct_v81/runs/development'
MODELS={'eligible_reaction':('eligible','reaction'),'eligible_event':('eligible','event'),
        'eligible_state':('eligible','state'),'top_reaction':('top','reaction')}
INDICES={'reaction':list(range(13)),'event':list(range(9)),'state':list(range(5))}
PRIOR=ROOT/'.cache/soxl_soxs_14d5pct_v87/runs/development'
PRICE_PRIOR=ROOT/'.cache/soxl_soxs_14d5pct_v83/runs/development'
CONTROL_MAP={'source_recent':'recent_ratio','source_uniform':'uniform_top','source_blend':'source_blend'}
ROUTES=list(MODELS)+['identity_matched']+list(CONTROL_MAP)+['source_wait','premarket_only','cash','double_cost']
STATE_FIELDS=['aligned_trend5','aligned_current','aligned_relative5','log_vol20','trend5_current_product','planned_activity_days_before_expiry','next_activity_day_proximity','planned_activity_days_previous14','previous_activity_day_proximity','observed_activity_days','aligned_issuer_excess_reaction','aligned_peer_excess_reaction','issuer_log_volume_reaction']


def verify_parent():
    parent.verify_parent()
    manifests={}
    for version,run in [(80,PARENT),(81,WAIT),(83,PRICE_PRIOR),(87,PRIOR)]:
        folder=ROOT/f'research/soxl_soxs_14d5pct_v{version}'
        m=read(folder/'evidence/delivery_manifest.json')
        for f,h in m['delivery_files_sha256'].items():assert sha(folder/f)==h,f
        for f,h in m['cache_files_sha256'].items():assert sha(run/f)==h,f
        for f,h in m['input_files_sha256'].items():assert sha(ROOT/m['input_root']/f)==h,f
        assert read(run/'verification.json')['passed']
        manifests[str(version)]=sha(folder/'evidence/delivery_manifest.json')
    manifests['prices']=verify_prices()
    return manifests


def state(row,side):
    sign=1 if side==0 else -1
    return np.array([sign*row.trend5,sign*row.current,sign*row.relative5,
                     row.log_vol20,row.trend5*row.current,row.event_future_count,row.event_future_proximity,row.event_recent_count,row.event_recent_proximity,row.reaction_count,sign*row.issuer_reaction,sign*row.peer_reaction,row.volume_reaction],float)


def probabilities(P,Z,beta):
    logits=log_probabilities(P)+np.asarray(Z)@np.asarray(beta)
    return np.exp(logits-logsumexp(logits,axis=1)[:,None])


def objective(beta,P,y,w,Z,hessian=False):
    logits=log_probabilities(P)+Z@beta
    lp=logits-logsumexp(logits,axis=1)[:,None]
    q=np.exp(lp)
    loss=float(-w@lp[np.arange(len(y)),y]+5*np.sum(beta**2))
    grad=Z.T@(w[:,None]*(q-np.eye(4)[y]))+10*beta
    if not hessian:return loss,grad
    cov=np.einsum('ik,kl->ikl',q,np.eye(4))-np.einsum('ik,il->ikl',q,q)
    H=np.einsum('i,ij,il,ikm->jklm',w,Z,Z,cov,optimize=True).reshape(beta.size,beta.size)
    H+=10*np.eye(beta.size)
    return loss,grad,H


def design(X,sides,clocks,w,mode):
    n=len(X);center=np.average(X,axis=0,weights=w);scale=np.sqrt(np.average((X-center)**2,axis=0,weights=w))
    scale[scale<1e-12]=1.
    Z=np.ones((n,1))
    if mode not in INDICES:raise ValueError('Unknown design')
    Z=np.c_[Z,sides,np.eye(8)[clocks,1:],((X-center)/scale)[:,INDICES[mode]]]
    return Z,center,scale


def fit(P,y,w,X,sides,clocks,mode):
    P=np.asarray(P,float);log_probabilities(P)
    y=np.asarray(y,int);w=np.asarray(w,float);X=np.asarray(X,float);sides=np.asarray(sides,int);clocks=np.asarray(clocks,int)
    if X.shape!=(len(P),13) or any(a.shape!=(len(P),) for a in [y,w,sides,clocks]):raise ValueError('Aligned training arrays required')
    if not np.isfinite(X).all() or not np.isfinite(w).all() or (w<=0).any():raise ValueError('Finite inputs and positive weights required')
    if not set(y)<=set(range(4)) or not set(sides)<={0,1} or not set(clocks)<=set(range(8)):raise ValueError('Invalid labels or discrete fields')
    if not len(y):return {'ready':False,'reason':'no_rows'}
    unsupported=int((P[np.arange(len(y)),y]==0).sum())
    if unsupported:return {'ready':False,'reason':'issued_true_class_had_zero_support','unsupported_rows':unsupported}
    Z,center,scale=design(X,sides,clocks,w,mode);beta=np.zeros((Z.shape[1],4))
    for iteration in range(80):
        loss,g,H=objective(beta,P,y,w,Z,True)
        if np.max(np.abs(g))<=1e-9:break
        step=np.linalg.solve(H,g.ravel()).reshape(beta.shape);rate=1.
        for _ in range(50):
            candidate=beta-rate*step;value,ng=objective(candidate,P,y,w,Z)
            if value<=loss-1e-4*rate*np.sum(g*step) or (value<=loss+64*np.finfo(float).eps*max(1.,abs(loss)) and np.max(np.abs(ng))<np.max(np.abs(g))/2):
                beta=candidate;break
            rate*=.5
        else:raise ValueError('State calibration line search failed')
    loss,g=objective(beta,P,y,w,Z);norm=float(np.max(np.abs(g)))
    if norm>1e-7:raise ValueError('State calibration gradient failed')
    return dict(ready=True,mode=mode,beta=beta,center=center,scale=scale,
                objective=loss,gradient_inf=norm,iterations=iteration,rows=len(y),
                input_min=X.min(axis=0),input_max=X.max(axis=0))


def query(model,P,x,side,clock):
    z=[1.]
    z.extend([side,*np.eye(8)[clock,1:]])
    z.extend(((np.asarray(x)-model['center'])/model['scale'])[INDICES[model['mode']]])
    return probabilities(np.asarray(P).reshape(1,4),np.array(z).reshape(1,-1),np.asarray(model['beta']))[0]


def samples(panel,parents,history,population):
    if population not in ['eligible','top']:raise ValueError('Unknown population')
    rows=[];date_weights=[];dates=[];n=len(parents)
    for i,p in enumerate(parents):
        day=[]
        for at in p['members']:
            e=history[at]['events']['recent_ratio']
            if population=='eligible' and e['action']=='CASH':continue
            j=core.ETFS.index(e['top_symbol']);r=panel.loc[at];assert r.calendar_known; y=int(r['L_gap' if j==0 else 'S_gap'])
            assert y>=0
            day.append((at,j,e['probabilities'][j],y,state(r,j),CLOCKS.index(r.clock)))
        if day:
            mass=2.**((i-n+1)/63);date_weights.append(mass);dates.append(p['date'])
            rows.extend([(*r,mass/len(day)) for r in day])
    b=np.array(date_weights);ess=float(b.sum()**2/(b@b)) if len(b) else 0.
    w=np.array([r[-1] for r in rows]);w*=ess/b.sum() if len(b) else 1.
    meta={'dates':dates,'real_dates':len(dates),'date_ess':ess,'members':[[r[0],r[1]] for r in rows],'weights':w}
    return (np.array([r[2] for r in rows]).reshape(-1,4),np.array([r[3] for r in rows],int),w,
            np.array([r[4] for r in rows]).reshape(-1,13),np.array([r[1] for r in rows],int),np.array([r[5] for r in rows],int),meta)


def daily_models(panel,parents,feedback,history,at):
    chosen=[p for p in parents if all(t in history and history[t]['events']['recent_ratio']['ready'] for t in p['members'])][-120:]
    arrays={pop:samples(panel,chosen,history,pop) for pop in ['eligible','top']}
    population_ready=len(chosen)>=60 and arrays['eligible'][-1]['real_dates']>=30
    models={}
    for name,(population,mode) in MODELS.items():
        P,y,w,X,sides,clocks,meta=arrays[population]
        model=fit(P,y,w,X,sides,clocks,mode) if population_ready else {'ready':False,'reason':'insufficient_dates'}
        models[name]={**meta,**model}
    ready=population_ready and all(m['ready'] for m in models.values()) and not feedback['expired_unresolved_prefix']
    return dict(at=at,cutoff=day_cutoff(at),parents=chosen,feedback=feedback,population_ready=population_ready,ready=ready,models=models)


def issue(at,row,daily,old,base,wait):
    src=old['events']['recent_ratio'];ready=daily['ready'] and src['ready'] and row.calendar_known
    header={'at':at,'date':row.date,'clock':row.clock}
    cash={**header,'ready':True,'action':'CASH','allocation':0.,'probabilities':None,'value':None}
    out={}
    for name in list(MODELS)+['identity_matched']:
        e={**cash,'ready':False}
        if ready:
            j=core.ETFS.index(src['top_symbol']);x=state(row,j);P=src['probabilities'][j]
            q=np.array(P) if name=='identity_matched' else query(daily['models'][name],P,x,j,CLOCKS.index(row.clock))
            m=np.array(base['distributions']['gap']['payoff_means'][j]);value=float(q@m)
            buy=src['action'] in core.ETFS and value>1e-12
            outside=[] if name=='identity_matched' else [STATE_FIELDS[k] for k in INDICES[daily['models'][name]['mode']] if x[k]<daily['models'][name]['input_min'][k]-1e-12 or x[k]>daily['models'][name]['input_max'][k]+1e-12]
            e={**header,'ready':True,'action':src['top_symbol'] if buy else 'CASH','allocation':float(buy),
               'probabilities':q,'value':value,'top_symbol':src['top_symbol'],'source_eligible':src['action'] in core.ETFS,
               'source_value':float(src['values'][j]),'outside_state_fields':outside}
        out[name]=e
    for name,route in CONTROL_MAP.items():out[name]=old['events'][route]
    out['source_wait']=wait['events']['paired_state'];out['premarket_only']=dict(out['eligible_reaction']) if row.clock==CLOCKS[0] else cash
    out['cash']=cash;out['double_cost']=dict(out['eligible_reaction'])
    return out


def prepare_panel(panel,market,output=None):
    catalog=read(CATALOG);out=panel.copy();rows=[]
    for at,row in panel.iterrows():
        deadline=row.date+pd.Timedelta(days=13)
        dates=market.sessions[(market.sessions>=row.date)&(market.sessions<=deadline)]
        assert len(dates)
        expiry=pd.Timestamp(market.calendar[dates[-1]]['close_at'])
        e=event_features(catalog,at.isoformat(),expiry.isoformat())
        assert e['known'],('Missing as-known calendar coverage',at,e)
        rows.append(dict(at=at,expiry=expiry,known=e['known'],
            **dict(zip(['event_future_count','event_future_proximity','event_recent_count','event_recent_proximity'],e['values'])),
            future_days=json.dumps(e['future_days']),previous_days=json.dumps(e['previous_days']),
            visible_record_ids=json.dumps(e['visible_record_ids'])))
    table=pd.DataFrame(rows).set_index('at')
    for name in ['event_future_count','event_future_proximity','event_recent_count','event_recent_proximity']:out[name]=table[name]
    out['calendar_known']=table.known
    out=prepare_reactions(out,table)
    if output is not None:table.to_csv(output,index=True)
    return out
