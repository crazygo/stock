"""Daily base refits and date-clustered calibration for causal feature ablations."""
from financial_features import *
from hierarchical_error import (read,dump,sha,labels,prepare_inputs,base_forecast,day_cutoff,restrict_market,
    development_gate,CLOCKS,tag,registry,pool,block_counts,decide,encode,fit_map,forecast)
from hierarchical_error import fit_day as source_fit_day,issue_hierarchical as source_issue
from clock_training import fit_clock,predict,training_dates,columns as old_columns

VARIANTS=['full','state','baseline']
ROUTES=['full_band','full_point','state_band','baseline_band','source_v32','premarket_only','cash','double_cost']


def columns(clock,kind):
    return old_columns(clock,'context')+(FINANCIAL if kind=='full' else STATE if kind=='state' else [])


def fit_bases(panel,at):
    same=panel[panel.clock==panel.loc[at,'clock']];dates,fb=training_dates(same,day_cutoff(at));out={};models={};raws={}
    for kind in VARIANTS:
        model,info=fit_clock(same,dates['early'],columns(panel.loc[at,'clock'],kind));raw=predict(model,same.loc[[at]])
        models[kind]=model;out[kind]={'fit':info,'feedback':fb,'raw':raw}
        raws[kind]=np.array([raw[s] for s in core.ETFS]) if raw else None
    return models,out,raws


def common_history(histories):
    keys=set.intersection(*(set(h) for h in histories.values()))
    return {t:histories['baseline'][t] if all(histories[k][t] is not None for k in VARIANTS) else None for t in sorted(keys)}


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
        raw=raws[kind];ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready']
        info={'ready':bool(ready),'point':None,'lower':None}
        if ready:
            x=encode(panel,{at:raw},[at])[0];m=daily['variants'][kind];point=raw+forecast(m['point_model'],x)[0]
            draws=np.array([raw+forecast(z,x)[0] for z in m['bootstrap_models']])
            radius=max(0.,float(np.quantile(np.max(draws-point,axis=1),.9,method='higher')))
            info.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=draws.tolist())
        out[kind]=info
    return out


def events(at,date,clock,estimates,fb,old):
    primary=decide(estimates['full'],fb,True);cash={'ready':True,'action':'CASH','allocation':0.}
    actions={'full_band':primary,'full_point':decide(estimates['full'],fb,False),
        'state_band':decide(estimates['state'],fb,True),'baseline_band':decide(estimates['baseline'],fb,True),
        'source_v32':old,'premarket_only':primary if clock==CLOCKS[0] else cash,'cash':cash,'double_cost':primary}
    return {k:{**e,'at':at,'date':date} for k,e in actions.items()}
