"""Replace only the amount-weighted direction link with shallow boosted trees."""
from pathlib import Path
import sys,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v35'))
from financial_learning import *
from sklearn.ensemble import HistGradientBoostingClassifier
from financial_learning import columns as source_columns
SOURCE=Path(__file__).resolve().parent
VARIANTS=['tree','stump','linear']
ROUTES=['tree_band','tree_point','stump_band','linear_band','source_v35','premarket_only','cash','double_cost']
PARAMETERS=dict(loss='log_loss',learning_rate=.05,max_iter=80,min_samples_leaf=30,l2_regularization=10.,
    max_bins=31,categorical_features=None,early_stopping=False,random_state=826)


def columns(clock,kind=None):return source_columns(clock,'full')


def fit_bases(panel,at):
    same=panel[panel.clock==panel.loc[at,'clock']];dates,fb=training_dates(same,day_cutoff(at));ids=dates['early'];cols=columns(panel.loc[at,'clock'])
    linear,linear_info=fit_clock(same,ids,cols);models={'linear':linear};fits={'linear':linear_info}
    for kind,depth in [('tree',2),('stump',1)]:
        model=copy.deepcopy(linear);info=copy.deepcopy(linear_info)
        if model is not None:
            rows=same.loc[pd.to_datetime(info['members'],utc=True)];Z=model['scaler'].transform(rows[cols].to_numpy())
            D=rows.D.to_numpy();weights=np.array(info['classification_weights']);y=(D>0).astype(int)
            model['heads'].pop('q',None);info['heads'].pop('q',None)
            if len(np.unique(y[weights>0]))==2:
                classifier=HistGradientBoostingClassifier(**PARAMETERS,max_depth=depth,max_leaf_nodes=2**depth)
                classifier.fit(Z,y,sample_weight=weights);model['heads']['q']=classifier
                info['heads']['q']={'algorithm':'HistGradientBoostingClassifier','parameters':classifier.get_params(),
                    'iterations':classifier.n_iter_,'baseline':classifier._baseline_prediction.tolist()}
            info['direction_algorithm']='shallow_boosted_tree' if depth==2 else 'boosted_stumps'
        models[kind]=model;fits[kind]=info
    outputs={};raws={}
    for kind in VARIANTS:
        raw=predict(models[kind],same.loc[[at]]);outputs[kind]={'fit':fits[kind],'feedback':fb,'raw':raw}
        raws[kind]=np.array([raw[s] for s in core.ETFS]) if raw else None
    return models,outputs,raws


def common_history(histories):
    return {t:histories['linear'][t] if all(t in histories[k] and histories[k][t] is not None for k in VARIANTS) else None for t in sorted(histories['linear'])}


def fit_calibration(panel,histories,reg,at):
    parents,fb=pool(reg,common_history(histories),at);ids=[t for p in parents for t in p['members']];n=len(parents)
    d={'date':panel.loc[at,'date'],'cutoff':day_cutoff(at),'feedback':fb,'parents':parents,'members':ids,'ready':n>=60,'variants':{}}
    if n<60:return d
    dw=2.**(-np.arange(n)[::-1]/63.);dw*=n/dw.sum();groups=np.repeat(np.arange(n),[len(p['members']) for p in parents])
    w=np.concatenate([np.repeat(dw[j]/len(p['members']),len(p['members'])) for j,p in enumerate(parents)])
    truth=panel.loc[ids,['C','D']].to_numpy(float);Ytrue=np.column_stack([truth[:,0]+truth[:,1],truth[:,0]-truth[:,1]])
    counts,starts,seed=block_counts(n,at,10);d.update(day_weights=dw,row_weights=w,row_groups=groups,block_starts=starts,seed=seed)
    for kind in VARIANTS:
        X=encode(panel,histories[kind],ids);Y=Ytrue-X[:,:2]
        d['variants'][kind]={'point_model':fit_map(X,Y,w,'partial',n),
            'bootstrap_models':[fit_map(X,Y,w*c[groups],'partial',n) for c in counts]}
    return d


def issue(panel,at,raws,daily):
    out={}
    for kind in VARIANTS:
        raw=raws[kind];ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready'];info={'ready':bool(ready),'point':None,'lower':None}
        if ready:
            x=encode(panel,{at:raw},[at])[0];m=daily['variants'][kind];point=raw+forecast(m['point_model'],x)[0]
            draws=np.array([raw+forecast(z,x)[0] for z in m['bootstrap_models']]);radius=max(0.,float(np.quantile(np.max(draws-point,axis=1),.9,method='higher')))
            info.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=draws.tolist())
        out[kind]=info
    return out


def events(at,date,clock,estimates,feedback,old):
    primary=decide(estimates['tree'],feedback,True);cash={'ready':True,'action':'CASH','allocation':0.}
    choices={'tree_band':primary,'tree_point':decide(estimates['tree'],feedback,False),'stump_band':decide(estimates['stump'],feedback,True),
        'linear_band':decide(estimates['linear'],feedback,True),'source_v35':old,
        'premarket_only':primary if clock==CLOCKS[0] else cash,'cash':cash,'double_cost':primary}
    return {n:{**e,'at':at,'date':date} for n,e in choices.items()}


def verified_source(phase,p):
    root=core.ROOT/p['source_runs'][phase]
    if phase=='development':
        d=read(core.ROOT/'research/soxl_soxs_14d5pct_v35/evidence/delivery_manifest.json')
        want={str(Path(folder)/f):h for field,folder in [('cache_outputs_sha256',''),('training_records_sha256','training'),('daily_records_sha256','daily')] for f,h in d[field].items()}
    else:
        delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v39/evidence/delivery_manifest.json')['runs']['financial_extension_20261013']
        assert sha(root/'extension_receipt.json')==delivery['top_level_sha256']['extension_receipt.json']
        want=read(root/'extension_receipt.json')['files_sha256']
    for f,h in want.items():assert sha(root/f)==h
    v=read(root/'verification.json');assert v['passed'] and v['results_sha256']==sha(root/'results.json')
    return root,want
