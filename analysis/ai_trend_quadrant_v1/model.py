"""Continuous Pugh + dependent-block resampling: conditional style membership."""
import hashlib
import numpy as np

VERSION='trend_quadrant_v1'
NAMES=['平稳累积','波动上行','事件台阶','高波动反转']
FEATURE_NAMES=['年化波动','最大收盘回撤','尾部亏损','日振幅','大波动集中度','隔夜能量占比','漂移变化','方向碎片化']
DATUM=np.array([.35,.15,.035,.025,.55,.55,1.25,.50])
SCALE=np.array([.25,.15,.025,.020,.20,.30,1.25,.50])
WX=np.array([.35,.30,.20,.15]);WY=np.array([.30,.20,.25,.25])
B=512

def measurements(a):
    """a: (...,n,4) columns close log return, gap, intraday, log(high/low)."""
    a=np.asarray(a,float)
    if a.ndim==2:a=a[None,:,:]
    r=a[:,:,0];n=r.shape[1];h=min(20,max(5,n//3));sd=np.std(r,axis=1,ddof=1)
    levels=np.concatenate([np.zeros((len(r),1)),np.cumsum(r,axis=1)],axis=1)
    dd=np.max(1-np.exp(levels-np.maximum.accumulate(levels,axis=1)),axis=1)
    k=max(1,int(np.ceil(.1*n)))
    losses=np.maximum(-r,0)
    tail=np.sort(losses,axis=1)[:,-k:].sum(axis=1)/np.maximum(1,np.minimum(k,(r<0).sum(axis=1)))
    energy=np.square(r).sum(axis=1)
    concentration=np.divide(np.sort(np.square(r),axis=1)[:,-k:].sum(axis=1),energy,out=np.zeros_like(energy),where=energy>0)
    g=np.square(a[:,:,1]).sum(axis=1);i=np.square(a[:,:,2]).sum(axis=1)
    gap=np.divide(g,g+i,out=np.zeros_like(g),where=g+i>0)
    change=np.divide(np.abs(r[:,-h:].mean(axis=1)-r.mean(axis=1)),sd/np.sqrt(h),out=np.zeros_like(sd),where=sd>1e-12)
    cum=np.concatenate([np.zeros((len(r),1)),np.cumsum(r,axis=1)],axis=1)
    rolling=cum[:,h:]-cum[:,:-h]
    fragmentation=1-np.abs(np.sign(rolling).mean(axis=1))
    return np.column_stack([sd*np.sqrt(252),dd,tail,np.median(a[:,:,3],axis=1),concentration,gap,change,fragmentation])

def normalize(m,datum_factor=1):return np.tanh((np.asarray(m)-DATUM*datum_factor)/SCALE)

def quadrants(x,y):
    return np.where(y>0,np.where(x>0,3,2),np.where(x>0,1,0))

def resample(a,code,asof,window,block=5,datum_factor=1,reps=B):
    key=f'{VERSION}|{code}|{asof}|{window}|{block}|{datum_factor}|{reps}'
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little'))
    n=len(a);block=min(block,n);nb=(n+block-1)//block
    starts=rng.integers(0,n-block+1,size=(reps,nb))
    ix=(starts[:,:,None]+np.arange(block)).reshape(reps,-1)[:,:n]
    z=normalize(measurements(a[ix]),datum_factor)
    wx=rng.dirichlet(40*WX,size=reps);wy=rng.dirichlet(40*WY,size=reps)
    x=np.sum(z[:,:4]*wx,axis=1);y=np.sum(z[:,4:]*wy,axis=1)
    counts=np.bincount(quadrants(x,y),minlength=4)
    p=(counts+.5)/(reps+2)
    return {'p':p,'xy_interval':np.percentile(np.column_stack([x,y]),[10,90],axis=0).tolist(),'counts':counts.tolist()}

def describe(a,code,asof,window,sensitivity=False):
    m=measurements(a)[0];z=normalize(m);x=float(z[:4]@WX);y=float(z[4:]@WY)
    boot=resample(a,code,asof,window)
    p=boot['p'];r=a[:,0];s=float(np.std(r,ddof=1))
    drift_ratio=float(r[-min(20,len(r)):].mean()/max(s,1e-12)*np.sqrt(min(20,len(r))))
    direction='上行' if drift_ratio>.5 else '下行' if drift_ratio<-.5 else '震荡'
    energy=float(np.abs(r).sum());er=float(r.sum()/energy) if energy>0 else 0.
    out={'as_of':asof,'window':window,'x':x,'y':y,'p':p.tolist(),'raw_p':p.tolist(),'dominant':int(np.argmax(p)),
        'entropy':float(-np.sum(p*np.log(p))/np.log(4)),'xy_interval':boot['xy_interval'],
        'measurements':m.tolist(),'z':z.tolist(),'contribution_x':(z[:4]*WX).tolist(),'contribution_y':(z[4:]*WY).tolist(),
        'price_return':float(np.expm1(r.sum())),'signed_efficiency':er,'drift_ratio':drift_ratio,'direction':direction,
        'bootstrap_counts':boot['counts'],'sensitivity':[],'clear_tendency':False}
    if sensitivity:
        for block,factor in [(10,1),(5,.85),(5,1.15)]:
            v=resample(a,code,asof,window,block,factor)
            out['sensitivity'].append({'block':block,'datum_factor':factor,'p':v['p'].tolist(),'dominant':int(np.argmax(v['p']))})
        out['clear_tendency']=bool(max(p)>=.65 and all(v['dominant']==out['dominant'] for v in out['sensitivity']))
    return out

def smooth(history):
    previous_raw=None;previous_p=None;previous_asof=None;weeks=0;previous_dom=None
    for row in history:
        if row.get('status')!='scored':
            previous_raw=previous_p=previous_asof=None;weeks=0;previous_dom=None;continue
        raw=np.array(row['raw_p']);reset=previous_p is None
        if previous_raw is not None:
            from datetime import date
            reset=bool(.5*np.abs(raw-previous_raw).sum()>=.40 or (date.fromisoformat(row['as_of'])-date.fromisoformat(previous_asof)).days>10)
        p=raw if reset else .65*raw+.35*previous_p
        dom=int(np.argmax(p));weeks=weeks+1 if max(p)>=.65 and dom==previous_dom and not reset else 1 if max(p)>=.65 else 0
        row.update(p=p.tolist(),dominant=dom,entropy=float(-np.sum(p*np.log(p))/np.log(4)),smoothing_reset=reset,
            observed_stable_weeks=weeks,clear_tendency=bool(row.get('clear_tendency') and max(p)>=.65 and dom==int(np.argmax(raw))))
        previous_raw=raw;previous_p=p;previous_asof=row['as_of'];previous_dom=dom
    return history
