"""Additional previous-session cross-asset risk proxies with original learners."""
from pathlib import Path
import sys,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v35'))
from financial_learning import *
from macro_features import MACRO,EQUITY,BONDS,load_raw,build_features,calendar
from financial_learning import columns as source_columns
SOURCE=Path(__file__).resolve().parent
VARIANTS=['macro','equity','bonds','baseline']
ROUTES=['macro_band','macro_point','equity_band','bonds_band','baseline_band','source_v35','premarket_only','cash','double_cost']

def columns(clock,kind='baseline'):
    return source_columns(clock,'full')+({'macro':MACRO,'equity':EQUITY,'bonds':BONDS,'baseline':[]}[kind])


def fit_bases(panel,at):
    same=panel[panel.clock==panel.loc[at,'clock']];dates,fb=training_dates(same,day_cutoff(at));models={};out={};raws={}
    for kind in VARIANTS:
        model,info=fit_clock(same,dates['early'],columns(panel.loc[at,'clock'],kind));raw=predict(model,same.loc[[at]])
        models[kind]=model;out[kind]={'fit':info,'feedback':fb,'raw':raw};raws[kind]=np.array([raw[s] for s in core.ETFS]) if raw else None
    return models,out,raws


def common_history(histories):
    return {t:histories['baseline'][t] if all(t in histories[k] and histories[k][t] is not None for k in VARIANTS) else None for t in sorted(histories['baseline'])}


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


def events(at,date,clock,estimates,fb,old):
    primary=decide(estimates['macro'],fb,True);cash={'ready':True,'action':'CASH','allocation':0.}
    out={'macro_band':primary,'macro_point':decide(estimates['macro'],fb,False),
        'equity_band':decide(estimates['equity'],fb,True),'bonds_band':decide(estimates['bonds'],fb,True),
        'baseline_band':decide(estimates['baseline'],fb,True),'source_v35':old,
        'premarket_only':primary if clock==CLOCKS[0] else cash,'cash':cash,'double_cost':primary}
    return {k:{**e,'at':at,'date':date} for k,e in out.items()}


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
