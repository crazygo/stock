"""Independent node traversal and weighted Newton equations for every tree leaf."""
import numpy as np
from scipy.special import expit,logit
from verify_continuation import near


def leaf_indices(nodes,X):
    out=np.zeros(len(X),dtype=int);pending=[(0,np.arange(len(X)))]
    while pending:
        j,ids=pending.pop();node=nodes[j]
        assert not node['is_categorical']
        if node['is_leaf']:out[ids]=j
        else:
            assert np.isfinite(X[ids,int(node['feature_idx'])]).all()
            left=X[ids,int(node['feature_idx'])]<=node['num_threshold']
            pending.append((int(node['left']),ids[left]));pending.append((int(node['right']),ids[~left]))
    return out


def probability(model,X):
    raw=np.full(len(X),float(model._baseline_prediction[0,0]))
    for step in model._predictors:
        assert len(step)==1;nodes=step[0].nodes;raw+=nodes['value'][leaf_indices(nodes,X)]
    return expit(raw)


def fit_check(model,X,y,w,depth):
    assert model.n_iter_==80 and model.early_stopping is False and not model.do_early_stopping_
    assert model.max_depth==depth and model.max_leaf_nodes==2**depth and model.min_samples_leaf==30
    assert model.max_bins==31 and model.l2_regularization==10. and model.learning_rate==.05
    assert model.class_weight is None and model.categorical_features is None
    q=float(np.average(y,weights=w));near(model._baseline_prediction[0,0],logit(q))
    raw=np.full(len(y),logit(q));checks=0
    for step in model._predictors:
        nodes=step[0].nodes;ids=leaf_indices(nodes,X);p=expit(raw);g=(p-y)*w;h=p*(1-p)*w
        assert nodes['depth'].max()<=depth and sum(nodes['is_leaf'])<=2**depth
        for j,node in enumerate(nodes):
            if not node['is_leaf']:continue
            members=np.flatnonzero(ids==j);assert len(members)==int(node['count'])
            assert len(members)>=30 or len(nodes)==1
            expected=-.05*g[members].sum()/(h[members].sum()+10.)
            np.testing.assert_allclose(node['value'],expected,atol=2e-8,rtol=2e-6);checks+=1
        raw+=nodes['value'][ids]
    np.testing.assert_allclose(expit(raw),model.predict_proba(X)[:,1],atol=1e-12,rtol=1e-10)
    return checks
