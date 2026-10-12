"""Mature-history centered cross-asset by sector interactions."""
from pathlib import Path
import sys,copy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v42'))
from macro_learning import *
from macro_features import MACRO,EQUITY,BONDS,load_raw,build_features,calendar
from macro_learning import columns as source_columns
SOURCE=Path(__file__).resolve().parent
VARIANTS=['joint','trend','risk','additive']
ROUTES=['joint_band','joint_point','trend_band','risk_band','additive_band','source_v35','premarket_only','cash','double_cost']

STATES=['sector_5','breadth_5','log_sector_volatility']

def pairs(kind):
    states=STATES if kind=='joint' else STATES[:1] if kind=='trend' else STATES[1:] if kind=='risk' else []
    return [(m,s) for m in MACRO for s in states]

def cross_name(m,s):return 'cross__'+m+'__'+s

def columns(clock,kind='additive'):
    return source_columns(clock,'macro')+[cross_name(m,s) for m,s in pairs(kind)]

def expanded(panel,basis):
    out=panel.copy()
    for m,s in basis['pairs']:
        out[cross_name(m,s)]=((panel[m]-basis['center'][m])/basis['scale'][m])*((panel[s]-basis['center'][s])/basis['scale'][s])
    return out

def predict_joint(model,current):
    if model is not None and 'basis' in model:current=expanded(current,model['basis'])
    return predict(model,current)

def fit_bases(panel,at):
    same=panel[panel.clock==panel.loc[at,'clock']];dates,fb=training_dates(same,day_cutoff(at));models={};out={};raws={};clock=panel.loc[at,'clock']
    additive,add_info=fit_clock(same,dates['early'],columns(clock,'additive'))
    for kind in VARIANTS:
        if kind=='additive':model,info=additive,add_info
        elif additive is None:
            model=None;info=copy.deepcopy(add_info);info['columns']=columns(clock,kind)
        else:
            names=columns(clock,'additive');scaler=additive['scaler'];basis={'pairs':pairs(kind),
                'center':dict(zip(names,scaler.mean_.tolist())),'scale':dict(zip(names,scaler.scale_.tolist()))}
            model,info=fit_clock(expanded(same,basis),dates['early'],columns(clock,kind));model['basis']=basis;info['basis']=basis
        raw=predict_joint(model,same.loc[[at]])
        models[kind]=model;out[kind]={'fit':info,'feedback':fb,'raw':raw};raws[kind]=np.array([raw[s] for s in core.ETFS]) if raw else None
    return models,out,raws


def common_history(histories):
    return {t:histories['additive'][t] if all(t in histories[k] and histories[k][t] is not None for k in VARIANTS) else None for t in sorted(histories['additive'])}


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
    primary=decide(estimates['joint'],fb,True);cash={'ready':True,'action':'CASH','allocation':0.}
    out={'joint_band':primary,'joint_point':decide(estimates['joint'],fb,False),
        'trend_band':decide(estimates['trend'],fb,True),'risk_band':decide(estimates['risk'],fb,True),
        'additive_band':decide(estimates['additive'],fb,True),'source_v35':old,
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


def verified_v42():
    root=core.ROOT/'.cache/soxl_soxs_14d5pct_v42/runs/macro_20261013'
    delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v42/evidence/delivery_manifest.json')
    want={f:h for f,h in delivery['cache_outputs_sha256'].items()}
    for folder in ['training','daily']:
        want.update({str(Path(folder)/f):h for f,h in delivery[folder+'_sha256'].items()})
    for f,h in want.items():assert sha(root/f)==h,f
    v=read(root/'verification.json');assert v['passed'] and v['results_sha256']==sha(root/'results.json')
    return root,want
