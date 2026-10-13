"""As-issued candidate class-bias calibration with unchanged source direction."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/soxl_soxs_14d5pct_v78'))
from class_learning import *
import class_learning as base
SOURCE=Path(__file__).resolve().parent;CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v79'
PARENT=ROOT/'.cache/soxl_soxs_14d5pct_v78/runs/development'
MODES=['eligible','top','both']
CONTROL_MAP={'source_blend':'gap_blend','source_prior':'gap_prior','source_raw':'gap_raw','source_merged':'merged_blend'}
ROUTES=[k+'_tilt' for k in MODES]+['identity']+list(CONTROL_MAP)+['premarket_only','cash','double_cost']


def verify_parent():
    base.verify_parent();folder=ROOT/'research/soxl_soxs_14d5pct_v78';m=read(folder/'evidence/delivery_manifest.json')
    for f,h in m['delivery_files_sha256'].items():assert sha(folder/f)==h,f
    for f,h in m['cache_files_sha256'].items():assert sha(PARENT/f)==h,f
    for f,h in m['input_files_sha256'].items():assert sha(ROOT/m['input_root']/f)==h,f
    assert read(PARENT/'verification.json')['passed']
    return sha(folder/'evidence/delivery_manifest.json')


def log_probabilities(P):
    P=np.asarray(P,float)
    if P.ndim!=2 or P.shape[1]!=4 or not np.isfinite(P).all() or (P<0).any() or not np.allclose(P.sum(1),1,atol=1e-12,rtol=0):
        raise ValueError('Four finite normalized probabilities required')
    out=np.full_like(P,-np.inf);np.log(P,out=out,where=P>0);return out


def tilt(P,beta):
    logits=log_probabilities(P)+np.asarray(beta,float)
    return np.exp(logits-logsumexp(logits,axis=1)[:,None])


def derivatives(beta,P,y,w):
    z=log_probabilities(P)+beta;logs=z-logsumexp(z,axis=1)[:,None];q=np.exp(logs)
    value=float(-np.dot(w,logs[np.arange(len(y)),y])+5*np.dot(beta,beta))
    gradient=np.sum(w[:,None]*(q-np.eye(4)[y]),axis=0)+10*beta
    H=np.diag(w@q)-q.T@(w[:,None]*q)+10*np.eye(4)
    return value,gradient,H


def fit(P,y,w):
    P=np.asarray(P,float);y=np.asarray(y,int);w=np.asarray(w,float);log_probabilities(P)
    if y.shape!=(len(P),) or w.shape!=y.shape or not set(y)<=set(range(4)) or not np.isfinite(w).all() or (w<=0).any():
        raise ValueError('Known labels and positive weights required')
    if not len(y):return {'ready':False,'reason':'no_rows'}
    unsupported=int((P[np.arange(len(y)),y]==0).sum())
    if unsupported:return {'ready':False,'reason':'issued_true_class_had_zero_support','unsupported_rows':unsupported}
    beta=np.zeros(4)
    for iteration in range(80):
        loss,g,H=derivatives(beta,P,y,w)
        if max(abs(g))<=1e-9:break
        step=np.linalg.solve(H,g);rate=1.
        for _ in range(50):
            candidate=beta-rate*step;v,ng,_=derivatives(candidate,P,y,w)
            if v<=loss-1e-4*rate*np.dot(g,step) or (v<=loss+64*np.finfo(float).eps*max(1.,abs(loss)) and max(abs(ng))<max(abs(g))/2):
                beta=candidate;break
            rate*=.5
        else:raise ValueError('Calibration Newton line search failed')
    loss,g,_=derivatives(beta,P,y,w)
    if max(abs(g))>1e-7:raise ValueError('Calibration gradient failed')
    return {'ready':True,'beta':beta,'objective':loss,'gradient_inf':float(max(abs(g))),'iterations':iteration,'rows':len(y)}


def select_parents(parents,history):
    return [p for p in parents if all(history.get(t) is not None and history[t]['events']['gap_blend']['ready'] for t in p['members'])][-120:]


def samples(panel,parents,history,mode):
    if mode not in MODES:raise ValueError('Unknown calibration population')
    n=len(parents);items=[];date_masses=[];days=[]
    for i,p in enumerate(parents):
        rows=[]
        for at in p['members']:
            old=history[at];info=old['estimates']['gap_blend'];top=int(info['values'][1]>info['values'][0])
            sides=[0,1] if mode=='both' else [top]
            if mode=='eligible' and old['events']['gap_blend']['action']=='CASH':continue
            for j in sides:
                y=int(panel.loc[at,['L_gap','S_gap'][j]]);assert y>=0
                rows.append((at,j,info['probabilities'][j],y))
        if rows:
            mass=2.**((i-n+1)/63);date_masses.append(mass);days.append(p['date'])
            items.extend([(*row,mass/len(rows)) for row in rows])
    b=np.asarray(date_masses);ess=float(b.sum()**2/(b@b)) if len(b) else 0.
    w=np.array([r[-1] for r in items]);w*=ess/b.sum() if len(b) else 1.
    metadata={'dates':days,'real_dates':len(days),'date_ess':ess,'members':[[r[0],r[1]] for r in items],'weights':w}
    return np.array([r[2] for r in items]).reshape(-1,4),np.array([r[3] for r in items],int),w,metadata


def daily_models(panel,parents,feedback,history,at):
    chosen=select_parents(parents,history);models={}
    arrays={m:samples(panel,chosen,history,m) for m in MODES}
    population_ready=len(chosen)>=60 and arrays['eligible'][3]['real_dates']>=30
    for m,(P,y,w,meta) in arrays.items():
        model=fit(P,y,w) if population_ready else {'ready':False,'reason':'insufficient_dates'}
        models[m]={**meta,**model}
    ready=population_ready and all(m['ready'] for m in models.values()) and not feedback['expired_unresolved_prefix']
    return {'at':at,'cutoff':day_cutoff(at),'parents':chosen,'feedback':feedback,'population_ready':population_ready,'ready':ready,'models':models}


def estimate(old,daily,mode):
    src=old['events']['gap_blend'];info=old['estimates']['gap_blend']
    if not daily['ready'] or not src['ready']:return {'ready':False,'action':'CASH','allocation':0.,'probabilities':None,'values':None}
    top=int(info['values'][1]>info['values'][0]);beta=np.zeros(4) if mode=='identity' else daily['models'][mode]['beta']
    probabilities=tilt(info['probabilities'],beta);means=np.asarray(old['distributions']['gap']['payoff_means']);values=np.sum(probabilities*means,axis=1)
    buy=src['action'] in core.ETFS and values[top]>1e-12
    return {'ready':True,'action':core.ETFS[top] if buy else 'CASH','allocation':float(buy),'probabilities':probabilities,'values':values,
        'top_symbol':core.ETFS[top],'buy_advantage':float(values[top]),'source_eligible':src['action'] in core.ETFS}


def events(at,row,estimates,old):
    out={k:{'at':at,'date':row.date,'clock':row.clock,**v} for k,v in estimates.items()}
    for k,s in CONTROL_MAP.items():out[k]=old['events'][s]
    cash={'at':at,'date':row.date,'clock':row.clock,'ready':True,'action':'CASH','allocation':0.,'values':None}
    out['cash']=cash;out['double_cost']=dict(out['eligible_tilt'])
    out['premarket_only']=dict(out['eligible_tilt']) if row.clock==CLOCKS[0] else cash
    return out
