"""Independent tree traversal, softmax, first-order equations and anchor refits."""
import numpy as np
from scipy.special import expit,softmax
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joint_loss import PARAMS
from verify_continuation import near


def manual_probability(model,X):
    if model is None:return None
    if model['scaler'] is not None:X=(X-model['scaler'].mean_)/model['scaler'].scale_
    e=model['estimator']
    if e is None:return model['prior'].copy()
    if model['kind']=='hgb':
        logits=np.asarray(e._baseline_prediction).reshape(-1).copy()
        for stage in e._predictors:
            for k,tree in enumerate(stage):
                node=0
                while not tree.nodes[node]['is_leaf']:
                    n=tree.nodes[node];assert not n['is_categorical']
                    node=int(n['left'] if X[0,int(n['feature_idx'])]<=n['num_threshold'] else n['right'])
                logits[k]+=tree.nodes[node]['value']
    else:logits=(X@e.coef_.T+e.intercept_)[0]
    values=np.r_[1-expit(logits[0]),expit(logits[0])] if len(e.classes_)==2 else softmax(logits)
    out=np.zeros(4);out[e.classes_.astype(int)]=values;return out


def check_fit(model,info,X,y,half,minimum,refit=False):
    n=len(y);w=np.array([2**((i-n+1)/half) for i in range(n)])
    if n:w*=n/w.sum()
    np.testing.assert_allclose(info['weights'],w,atol=1e-12,rtol=1e-10)
    assert info['rows']==n and info['ready']==(n>=minimum) and info['classes']==sorted(map(int,np.unique(y)))
    if n<minimum:assert model is None;return 0
    prior=np.array([sum(w[y==k]) for k in range(4)]);prior/=sum(prior)
    np.testing.assert_allclose(info['prior'],prior,atol=1e-12);np.testing.assert_allclose(model['prior'],prior,atol=1e-12)
    transformed=X.copy()
    if info['kind']=='logit':
        center=np.average(X,axis=0,weights=w);scale=np.sqrt(np.average((X-center)**2,axis=0,weights=w));scale[scale<1e-12]=1.
        for saved in [info['center'],model['scaler'].mean_]:np.testing.assert_allclose(saved,center,atol=1e-10,rtol=1e-10)
        for saved in [info['scale'],model['scaler'].scale_]:np.testing.assert_allclose(saved,scale,atol=1e-10,rtol=1e-10)
        transformed=(X-center)/scale
    e=model['estimator'];checks=1
    if len(np.unique(y))<2:assert e is None;return checks
    assert list(e.classes_)==sorted(np.unique(y))
    if info['kind']=='logit':
        B=e.coef_;a=e.intercept_;z=transformed@B.T+a
        if len(e.classes_)==2:residual=w[:,None]*(expit(z)-(y==e.classes_[1])[:,None])
        else:
            target=np.column_stack([y==c for c in e.classes_]);residual=w[:,None]*(softmax(z,axis=1)-target)
        grad=residual.T@transformed+B/.1
        assert max(abs(grad).max(),abs(residual.sum(axis=0)).max())<2e-6*max(1,n)
        np.testing.assert_allclose(info['coefficients'],B,atol=1e-12);np.testing.assert_allclose(info['intercept'],a,atol=1e-12)
        checks+=1
    else:
        for k,val in PARAMS.items():assert e.get_params()[k]==val
    if refit:
        fresh=HistGradientBoostingClassifier(**PARAMS) if info['kind']=='hgb' else LogisticRegression(C=.1,solver='lbfgs',max_iter=2000,tol=1e-10)
        fresh.fit(transformed,y,sample_weight=w)
        if info['kind']=='hgb':
            np.testing.assert_allclose(fresh._baseline_prediction,e._baseline_prediction,atol=1e-12)
            for oldstage,newstage in zip(e._predictors,fresh._predictors):
                for oldtree,newtree in zip(oldstage,newstage):np.testing.assert_array_equal(oldtree.nodes,newtree.nodes)
        else:
            np.testing.assert_allclose(fresh.coef_,e.coef_,atol=1e-9,rtol=1e-8);np.testing.assert_allclose(fresh.intercept_,e.intercept_,atol=1e-9,rtol=1e-8)
        checks+=1
    return checks
