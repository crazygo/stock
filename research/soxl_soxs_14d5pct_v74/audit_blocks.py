"""Independent original-coordinate full equations with diagonal block penalties."""
from block_learning import *
from audit_quadratic import design,compare,query


def reference(X,Y,pair,w,n,degree=2,factors=None):
    meta,Z,groups=design(X,Y,pair,w,n,degree);k=len(meta['names']);d=Z.shape[1];onehot=np.eye(k)[groups]
    if factors is None:
        raw=X.shape[1];f=np.array([raw/18]*raw+[raw*(raw+1)/2/18]*(d-raw))
    else:f=np.asarray(factors,float)
    common=np.diag(100*f);deviation=np.diag(300*f);weights=meta['weights'];mass=meta['pair_mass'];coeff={}
    for mode in VARIANTS:
        if mode=='pooled':A=Z;P=common
        else:
            A=(onehot[:,:,None]*Z[:,None,:]).reshape(len(Z),k*d);P=np.zeros((k*d,k*d))
            mean=np.concatenate([mass[i]*np.eye(d) for i in range(k)],axis=1)
            if mode=='partial':P+=mean.T@common@mean
            for i in range(k):
                select=np.zeros((d,k*d));select[:,i*d:(i+1)*d]=np.eye(d)
                delta=select-mean if mode=='partial' else select
                P+=mass[i]*(delta.T@(deviation if mode=='partial' else common)@delta)
        A=np.column_stack([onehot,A]);full=np.zeros((A.shape[1],A.shape[1]));full[k:,k:]=P
        beta=np.linalg.solve(A.T@(weights[:,None]*A)+full,A.T@(weights[:,None]*Y))
        np.testing.assert_allclose(beta[:k],meta['intercepts'],atol=1e-10,rtol=1e-9)
        coeff[mode]=np.repeat(beta[k:][None,:,:],k,axis=0) if mode=='pooled' else beta[k:].reshape(k,d,Y.shape[1])
    meta.update(degree=degree,distinct_dates=n,common=100*f,deviation=300*f,coefficients=coeff)
    return meta
