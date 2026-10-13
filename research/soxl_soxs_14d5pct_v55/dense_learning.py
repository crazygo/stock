"""Eight independent base clocks and date-weighted partial sharing."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v35'))
from financial_learning import (core,np,pd,read,dump,sha,labels,day_cutoff,restrict_market,
    development_gate,tag,block_counts,decide,fit_clock,predict,training_dates,columns,load_sources,build)
from clock_training import (clock_volumes,prepare,joint_releases,factors,terminal_panel,baseline_labels,load_panel,PANEL_ROOT)
from run_breadth import build_features
from shared_error import pool
SOURCE=Path(__file__).resolve().parent
CLOCKS=['09:25:30','10:05:30','10:35:30','11:05:30','12:05:30','13:05:30','14:05:30','15:05:30']
OLD_CLOCKS=['09:25:30','10:05:30','10:35:30','13:05:30']
ROUTES=['dense_band','dense_point','source_dense','old_four_band','dense_four_only','added_only','premarket_only','cash','double_cost']


def verified_source(phase,p):
    root=core.ROOT/p['source_runs'][phase]
    if phase=='development':
        d=read(core.ROOT/'research/soxl_soxs_14d5pct_v35/evidence/delivery_manifest.json')
        want={str(Path(folder)/f):h for field,folder in [('cache_outputs_sha256',''),('training_records_sha256','training'),('daily_records_sha256','daily')] for f,h in d[field].items()}
    else:
        d=read(core.ROOT/'research/soxl_soxs_14d5pct_v39/evidence/delivery_manifest.json')['runs']['financial_extension_20261013']
        assert sha(root/'extension_receipt.json')==d['top_level_sha256']['extension_receipt.json'];want=read(root/'extension_receipt.json')['files_sha256']
    for f,h in want.items():assert sha(root/f)==h
    v=read(root/'verification.json');assert v['passed'] and v['results_sha256']==sha(root/'results.json')
    return root,want


def registry(panel,market,clocks=CLOCKS):
    out=[]
    for day,g in panel.groupby('date',sort=True):
        expected=[c for c in clocks if pd.Timestamp(str(day.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[day]['close_at'])]
        ready=list(g.clock)==expected and g.feature_ready.all()
        release=g.terminal_release.max() if ready and g.terminal_release.notna().all() else None
        out.append({'date':day,'members':list(g.index),'feature_ready':bool(ready),'release':release})
    return out


def encode(panel,history,ids,clocks=CLOCKS):return np.array([[*history[t],clocks.index(panel.loc[t,'clock'])] for t in ids],float)


def penalty(k):
    assert k>=1
    deviation=80./k
    return np.kron(deviation*np.eye(k)+(20./k**2-deviation/k)*np.ones((k,k)),np.eye(2))


def fit_map(X,Y,w,n,k=8):
    w=np.asarray(w,float);w=w*n/w.sum();keep=w>0;X=np.asarray(X)[keep];Y=np.asarray(Y)[keep];w=w[keep]
    clock=X[:,2].astype(int);assert np.array_equal(clock,X[:,2]) and set(clock)==set(range(k))
    means=[];scales=[];intercepts=[];Z=np.zeros((len(X),2));R=np.zeros_like(Y)
    for c in range(k):
        mask=clock==c;v=w[mask];x=X[mask,:2];y=Y[mask]
        center=np.average(x,axis=0,weights=v);scale=np.sqrt(np.average((x-center)**2,axis=0,weights=v));scale[scale<1e-12]=1.
        intercept=np.average(y,axis=0,weights=v);Z[mask]=(x-center)/scale;R[mask]=y-intercept
        means.append(center);scales.append(scale);intercepts.append(intercept)
    A=(np.eye(k)[clock,:,None]*Z[:,None,:]).reshape(len(Z),2*k);P=penalty(k)
    L=np.linalg.cholesky(P);augmented=np.vstack((A*np.sqrt(w)[:,None],L.T));target=np.vstack((R*np.sqrt(w)[:,None],np.zeros((len(P),2))))
    beta=np.linalg.lstsq(augmented,target,rcond=None)[0].reshape(k,2,2)
    return {'kind':'partial','input_center':np.array(means).tolist(),'input_scale':np.array(scales).tolist(),'intercepts':np.array(intercepts).tolist(),'coefficients':beta.tolist()}


def forecast(model,x):
    out=[];k=len(model['coefficients'])
    for row in np.atleast_2d(x):
        c=int(row[2]);assert c==row[2] and c in range(k)
        z=(row[:2]-model['input_center'][c])/model['input_scale'][c]
        out.append(z@np.array(model['coefficients'][c])+model['intercepts'][c])
    return np.array(out)


def fit_day(panel,history,reg,at,clocks=CLOCKS):
    parents,fb=pool(reg,history,at);ids=[t for p in parents for t in p['members']];n=len(parents)
    out={'date':panel.loc[at,'date'],'cutoff':day_cutoff(at),'feedback':fb,'parents':parents,'members':ids,'ready':n>=60,'clocks':clocks,'model':{}}
    if n<60:return out
    dw=2.**(-np.arange(n)[::-1]/63.);dw*=n/dw.sum();groups=np.repeat(np.arange(n),[len(p['members']) for p in parents])
    w=np.concatenate([np.repeat(dw[j]/len(p['members']),len(p['members'])) for j,p in enumerate(parents)])
    X=encode(panel,history,ids,clocks);truth=panel.loc[ids,['C','D']].to_numpy(float)
    Y=np.column_stack([truth[:,0]+truth[:,1],truth[:,0]-truth[:,1]])-X[:,:2];counts,starts,seed=block_counts(n,at,10)
    out.update(day_weights=dw,row_weights=w,row_groups=groups,block_starts=starts,seed=seed)
    out['model']={'point_model':fit_map(X,Y,w,n,len(clocks)),'bootstrap_models':[fit_map(X,Y,w*c[groups],n,len(clocks)) for c in counts]}
    return out


def issue(panel,at,raw,daily):
    ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready'];info={'ready':bool(ready),'point':None,'lower':None}
    if ready:
        x=encode(panel,{at:raw},[at],daily['clocks'])[0];models=daily['model'];point=raw+forecast(models['point_model'],x)[0]
        draws=np.array([raw+forecast(m,x)[0] for m in models['bootstrap_models']]);radius=max(0.,float(np.quantile(np.max(draws-point,axis=1),.9,method='higher')))
        info.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=draws.tolist())
    return info


def fit_base(panel,at):
    same=panel[panel.clock==panel.loc[at,'clock']];dates,fb=training_dates(same,day_cutoff(at))
    model,info=fit_clock(same,dates['early'],columns(panel.loc[at,'clock'],'full'));raw=predict(model,same.loc[[at]])
    return model,{'fit':info,'feedback':fb,'raw':raw},np.array([raw[s] for s in core.ETFS]) if raw else None


def events(at,date,clock,info,feedback,old,source_dense):
    primary=decide(info,feedback,True);cash={'ready':True,'action':'CASH','allocation':0.}
    out={'dense_band':primary,'dense_point':decide(info,feedback,False),'source_dense':source_dense,'old_four_band':old if clock in OLD_CLOCKS else cash,
        'dense_four_only':primary if clock in OLD_CLOCKS else cash,'added_only':primary if clock not in OLD_CLOCKS else cash,
        'premarket_only':primary if clock==CLOCKS[0] else cash,'cash':cash,'double_cost':primary}
    return {k:{**e,'at':at,'date':date} for k,e in out.items()}


def verified_dense_source():
    src=core.ROOT/'research/soxl_soxs_14d5pct_v54/evidence/delivery_manifest.json'
    manifest=read(src);run=core.ROOT/'.cache/soxl_soxs_14d5pct_v54/runs/development_20261013'
    record=manifest['runs'][run.name]
    for key,folder in [('top_level_sha256',''),('training_records_sha256','training'),('daily_records_sha256','daily'),('base_models_sha256','models'),('executed_sources_sha256','executed_sources')]:
        for f,h in record[key].items():assert sha(run/folder/f)==h
    for f,h in manifest['delivery_files_sha256'].items():assert sha(src.parents[1]/f)==h
    v=read(run/'verification.json');assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    return run,sha(src)


def source_module(filename, name):
    import importlib.util
    spec=importlib.util.spec_from_file_location(name,core.ROOT/'research/soxl_soxs_14d5pct_v54'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
