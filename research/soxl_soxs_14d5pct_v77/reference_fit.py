"""Full intercept normal equations in reference-loss units, independent of fit_state."""
import numpy as np


def solve(X,Y,pairs,weights,real_dates,reference_mass,kind):
    X=np.array(X,float,copy=True);Y=np.array(Y,float);pairs=np.asarray(pairs)
    if kind in ['short_price','long_price','long_native']:X[:,11:16]=0
    names=sorted(set(pairs));k=len(names);p=X.shape[1];indices=np.array([names.index(s) for s in pairs])
    actual=np.asarray(weights,float)*real_dates/sum(weights)
    objective_mass=real_dates if kind=='long_native' else reference_mass
    loss_weights=actual*objective_mass/real_dates
    centers=[];scales=[];Z=np.zeros_like(X)
    for j in range(k):
        keep=indices==j;w=loss_weights[keep];mean=(w[:,None]*X[keep]).sum(axis=0)/w.sum()
        sd=np.sqrt((w[:,None]*(X[keep]-mean)**2).sum(axis=0)/w.sum());sd[sd<1e-12]=1
        centers.append(mean);scales.append(sd);Z[keep]=(X[keep]-mean)/sd
    onehot=np.eye(k)[indices];slopes=(onehot[:,:,None]*Z[:,None,:]).reshape(len(X),k*p)
    mean=np.tile(np.eye(p)/k,(1,k));penalty=100*(mean.T@mean)
    for j in range(k):
        individual=np.zeros((p,k*p));individual[:,j*p:(j+1)*p]=np.eye(p)
        penalty+=300/k*((individual-mean).T@(individual-mean))
    A=np.column_stack([onehot,slopes]);P=np.zeros((A.shape[1],A.shape[1]));P[k:,k:]=penalty
    beta=np.linalg.solve(A.T@(loss_weights[:,None]*A)+P,A.T@(loss_weights[:,None]*Y))
    return {'names':names,'centers':np.array(centers),'scales':np.array(scales),
            'intercepts':beta[:k],'coefficients':beta[k:].reshape(k,p,2),
            'weights':actual,'penalty':penalty*real_dates/objective_mass,
            'reference_mass':reference_mass,'objective_loss_mass':objective_mass}


def from_panels(panels,ids,reference_mass,kind,fields):
    n=len(ids);w=np.array([2.**((i-n+1)/126) for i in range(n)]);w*=n/w.sum()
    if kind=='long_mean':
        y=panels['semiconductor'].loc[ids,['C','D']].to_numpy(float)
        return {'distinct_dates':n,'weights':w,'mean_cd':np.sum(w[:,None]*y,axis=0)/sum(w)}
    X=np.concatenate([p.loc[ids,fields].to_numpy(float) for p in panels.values()])
    Y=np.concatenate([p.loc[ids,['C','D']].to_numpy(float) for p in panels.values()])
    return solve(X,Y,np.repeat(list(panels),n),np.tile(w/len(panels),len(panels)),n,reference_mass,kind)
