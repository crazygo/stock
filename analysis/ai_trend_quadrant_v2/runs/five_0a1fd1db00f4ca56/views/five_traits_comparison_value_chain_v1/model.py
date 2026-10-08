"""Exploratory five-trait descriptors. No future-price labels or order APIs."""
import hashlib
import numpy as np

VERSION='five_traits_v2'
KEYS=['G','V','M','J','S']
WINDOWS=[120,60,60,60,140]
NAMES=['长期方向','波动幅度','收益组织代理','跳变集中代理','历史稳定度']
CATEGORIES=[['下行','中性','上行'],['较低','较高'],['反向','接近零','延续'],['较分散','较集中'],['较易变','较稳定']]

def four(a):
    a=np.asarray(a,float)
    if a.ndim==1:a=a[None,:]
    n=a.shape[1];out=np.full((len(a),4),np.nan)
    if n>=120:out[:,0]=a[:,-120:].mean(axis=1)*252
    if n>=60:
        r=a[:,-60:];out[:,1]=np.std(r,axis=1,ddof=1)*np.sqrt(252)
        left=r[:,:-1]-r[:,:-1].mean(axis=1,keepdims=True);right=r[:,1:]-r[:,1:].mean(axis=1,keepdims=True)
        denom=np.sqrt(np.square(left).sum(axis=1)*np.square(right).sum(axis=1))
        out[:,2]=np.divide((left*right).sum(axis=1),denom,out=np.full(len(a),np.nan),where=denom>1e-20)
        e=np.square(r-r.mean(axis=1,keepdims=True));energy=e.sum(axis=1)
        out[:,3]=np.divide(np.sort(e,axis=1)[:,-6:].sum(axis=1),energy,out=np.full(len(a),np.nan),where=energy>1e-20)
    return out

def project(a):
    return np.column_stack([np.tanh(a[:,0]/.5),np.tanh((a[:,1]-.35)/.25),a[:,2],np.tanh((a[:,3]-.55)/.20)])

def measure(a):
    a=np.asarray(a,float)
    if a.ndim==1:a=a[None,:]
    base=four(a);s=np.full(len(a),np.nan)
    if a.shape[1]>=140:
        sequence=np.stack([project(four(a[:,:a.shape[1]-i])) for i in range(20,-1,-1)],axis=1)
        delta=np.mean(np.abs(np.diff(sequence,axis=1)),axis=(1,2))/2
        s=np.exp(-10*delta)
    return np.column_stack([base,s])

def category(k,a):
    if k==0:return np.where(a<-.10,0,np.where(a>.10,2,1))
    if k==1:return (a>.35).astype(int)
    if k==2:return np.where(a<-.15,0,np.where(a>.15,2,1))
    if k==3:return (a>.60).astype(int)
    return (a>=.75).astype(int)

def boot(a,code,asof,reps=256):
    a=np.asarray(a,float)[-140:];n=len(a)
    if n<60:return [None]*5
    seed=int.from_bytes(hashlib.sha256(f'{VERSION}|{code}|{asof}|{reps}'.encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed);nb=(n+4)//5
    starts=rng.integers(0,n-4,size=(reps,nb));ix=(starts[:,:,None]+np.arange(5)).reshape(reps,-1)[:,:n]
    m=measure(a[ix]);out=[]
    for k in range(5):
        values=m[:,k];values=values[np.isfinite(values)]
        if not len(values):out.append(None);continue
        counts=np.bincount(category(k,values),minlength=len(CATEGORIES[k]))
        p=(counts+.5)/(len(values)+.5*len(counts))
        out.append({'range':np.percentile(values,[10,90]).tolist(),'p':p.tolist(),'categories':CATEGORIES[k],'replicates':len(values)})
    return out
