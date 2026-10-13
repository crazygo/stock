"""Full intercept normal equations, independently enumerated polynomial terms."""
from itertools import combinations_with_replacement
from quadratic_learning import *


def design(X,Y,pair,weights,distinct_dates,degree=2):
    names=sorted(set(pair));X=np.asarray(X,float);Y=np.asarray(Y,float);pair=np.asarray(pair)
    w=np.asarray(weights,float)*distinct_dates/sum(weights);group=np.array([names.index(x) for x in pair])
    means=[];stds=[];centers=[];scales=[];intercepts=[];mass=[];blocks=[]
    mapped=np.zeros((len(X),X.shape[1]+X.shape[1]*(X.shape[1]+1)//2 if degree==2 else X.shape[1]))
    for i,name in enumerate(names):
        mask=group==i;a=X[mask];v=w[mask];center=np.sum(v[:,None]*a,axis=0)/sum(v)
        sd=np.sqrt(np.sum(v[:,None]*(a-center)**2,axis=0)/sum(v));sd[sd<1e-12]=1
        z=(a-center)/sd
        terms=[z[:,j] for j in range(z.shape[1])]
        if degree==2:terms += [z[:,j]*z[:,k] for j,k in combinations_with_replacement(range(z.shape[1]),2)]
        values=np.stack(terms,axis=1);cm=np.sum(v[:,None]*values,axis=0)/sum(v)
        cs=np.sqrt(np.sum(v[:,None]*(values-cm)**2,axis=0)/sum(v));cs[cs<1e-12]=1
        mapped[mask]=(values-cm)/cs
        means.append(center);stds.append(sd);centers.append(cm);scales.append(cs)
        intercepts.append(np.sum(v[:,None]*Y[mask],axis=0)/sum(v));mass.append(sum(v)/distinct_dates)
    return {'names':names,'original_center':np.array(means),'original_scale':np.array(stds),'centers':np.array(centers),
            'scales':np.array(scales),'intercepts':np.array(intercepts),'pair_mass':np.array(mass),
            'weights':w},mapped,group


def reference(X,Y,pair,w,n,degree=2,common=100.,deviation=300.):
    meta,Z,group=design(X,Y,pair,w,n,degree);k=len(meta['names']);d=Z.shape[1];onehot=np.eye(k)[group]
    weights=meta['weights'];mass=meta['pair_mass'];coeff={}
    for mode in VARIANTS:
        if mode=='pooled':A=Z;P=common*np.eye(d)
        else:
            A=(onehot[:,:,None]*Z[:,None,:]).reshape(len(Z),k*d);P=np.zeros((k*d,k*d))
            mean=np.concatenate([mass[i]*np.eye(d) for i in range(k)],axis=1)
            if mode=='partial':P+=common*mean.T@mean
            for i in range(k):
                select=np.zeros((d,k*d));select[:,i*d:(i+1)*d]=np.eye(d)
                delta=select-mean if mode=='partial' else select
                P+=(deviation if mode=='partial' else common)*mass[i]*(delta.T@delta)
        A=np.column_stack([onehot,A]);full=np.zeros((A.shape[1],A.shape[1]));full[k:,k:]=P
        beta=np.linalg.solve(A.T@(weights[:,None]*A)+full,A.T@(weights[:,None]*Y))
        np.testing.assert_allclose(beta[:k],meta['intercepts'],atol=1e-10,rtol=1e-9)
        coeff[mode]=np.repeat(beta[k:][None,:,:],k,axis=0) if mode=='pooled' else beta[k:].reshape(k,d,Y.shape[1])
    meta.update(degree=degree,distinct_dates=n,common=common,deviation=deviation,coefficients=coeff)
    return meta


def compare(saved,want):
    assert saved['names']==want['names'] and saved['degree']==want['degree'] and saved['distinct_dates']==want['distinct_dates']
    for field in ['original_center','original_scale','centers','scales','intercepts','pair_mass','weights']:
        np.testing.assert_allclose(saved[field],want[field],atol=1e-10,rtol=1e-9)
    for mode in VARIANTS:np.testing.assert_allclose(saved['coefficients'][mode],want['coefficients'][mode],atol=1e-10,rtol=1e-8)


def query(model,x,name,mode):
    i=model['names'].index(name);z=[(float(v)-model['original_center'][i,j])/model['original_scale'][i,j] for j,v in enumerate(x)]
    values=z+[z[j]*z[k] for j,k in combinations_with_replacement(range(len(z)),2)] if model['degree']==2 else z
    normalized=[(v-model['centers'][i,j])/model['scales'][i,j] for j,v in enumerate(values)]
    return model['intercepts'][i]+np.asarray(normalized)@model['coefficients'][mode][i]
