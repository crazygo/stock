"""As-issued component blending with a shared date bootstrap and a risk floor."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v35'))
from financial_learning import (core,np,pd,read,dump,sha,CLOCKS,tag,day_cutoff,
    registry,pool,block_counts,encode,fit_map,forecast,restrict_market,labels,
    development_gate,decide)

SOURCE=Path(__file__).resolve().parent
ROUTES=['adaptive_band','adaptive_point','shared_band','equal_band','raw_band','source_v35','cash','double_cost']


def components(x):
    x=np.asarray(x,float)
    return np.stack(((x[...,0]+x[...,1])/2,(x[...,0]-x[...,1])/2),axis=-1)


def restore(x):
    x=np.asarray(x,float)
    return np.stack((x[...,0]+x[...,1],x[...,0]-x[...,1]),axis=-1)


def weights(parents):
    n=len(parents);dw=2.**(-np.arange(n)[::-1]/63.);dw*=n/dw.sum()
    groups=np.repeat(np.arange(n),[len(p['members']) for p in parents])
    rw=np.concatenate([np.repeat(dw[j]/len(p['members']),len(p['members'])) for j,p in enumerate(parents)])
    return dw,rw,groups


def blend_fit(raw,cal,truth,clocks,groups,w,kind):
    """v18 scalar bounded loss; effective sample size counts weighted dates."""
    raw,cal,truth=[components(x) for x in [raw,cal,truth]]
    out=np.empty((4,2));heads=[]
    for c in range(4) if kind=='clock' else [-1]:
        m=(clocks==c) if c>=0 else np.ones(len(w),bool)
        m=m&(w>0);v=w[m]/w[m].sum();datew=np.bincount(groups[m],weights=v)
        effective=1/float(datew@datew)
        for k in range(2):
            delta=cal[m,k]-raw[m,k];residual=truth[m,k]-raw[m,k]
            A=float(v@(delta*delta));B=float(v@(delta*residual));penalty=2*A/effective
            alpha=.5 if A<=1e-16 else float(np.clip((B+penalty/2)/(A+penalty),0,1))
            out[c if c>=0 else slice(None),k]=alpha
            heads.append({'clock':c,'component':k,'curvature':A,'linear':B,
                          'effective_dates':effective,'penalty':penalty,'alpha':alpha})
    return {'alpha':out.tolist(),'heads':heads}


def mixture(raw,cal,alpha):
    r=components(raw);return restore(r+np.asarray(alpha)*(components(cal)-r))


def rows(panel,records,ids):
    raw=np.array([records[t]['raws']['full'] for t in ids])
    cal=np.array([records[t]['estimates']['full']['point'] for t in ids])
    truth=restore(panel.loc[ids,['C','D']].to_numpy(float))
    clocks=np.array([CLOCKS.index(panel.loc[t,'clock']) for t in ids])
    return raw,cal,truth,clocks


def fit_day(panel,records,reg,at,source_day):
    history={t:r['raws']['full'] if r['estimates']['full']['ready'] else None for t,r in records.items() if t<day_cutoff(at)}
    parents,fb=pool(reg,history,at);ids=[t for p in parents for t in p['members']]
    ready=len(parents)>=60 and source_day['ready']
    out={'date':panel.loc[at,'date'],'cutoff':day_cutoff(at),'parents':parents,'members':ids,
         'feedback':fb,'ready':bool(ready),'models':{},'bootstrap':[]}
    if not ready:return out
    raw,cal,truth,clocks=rows(panel,records,ids);_,w,groups=weights(parents)
    out.update(row_weights=w,row_groups=groups)
    for kind in ['clock','shared']:out['models'][kind]=blend_fit(raw,cal,truth,clocks,groups,w,kind)
    baseparents=source_day['parents'];baseids=list(pd.to_datetime(source_day['members'],utc=True))
    base_dates=[pd.Timestamp(p['date']) for p in baseparents];meta_dates=[p['date'] for p in parents]
    union=sorted(set(base_dates+meta_dates));out['union_dates']=union
    # The same calendar-date multiplicity perturbs both fits. Separate pools must
    # not be coupled by their row numbers: their oldest dates are different.
    _,bw,bgroups=weights(baseparents);ug={d:j for j,d in enumerate(union)}
    bg=np.array([ug[base_dates[g]] for g in bgroups]);mg=np.array([ug[meta_dates[g]] for g in groups])
    history_raw={t:records[t]['raws']['full'] for t in baseids}
    X=encode(panel,history_raw,baseids);Y=restore(panel.loc[baseids,['C','D']].to_numpy(float))-X[:,:2]
    point=fit_map(X,Y,bw,'partial',len(baseparents));out['calibration_model']=point
    counts,starts,seed=block_counts(len(union),at,10);out.update(block_starts=starts,seed=seed)
    for count in counts:
        meta_weight=w*count[mg];base_weight=bw*count[bg]
        out['bootstrap'].append({'calibration':fit_map(X,Y,base_weight,'partial',len(baseparents)),
            **{kind:blend_fit(raw,cal,truth,clocks,groups,meta_weight,kind) for kind in ['clock','shared']}})
    return out


def issue(panel,at,record,daily):
    old=record['estimates']['full'];raw=record['raws']['full']
    ready=bool(old['ready'] and panel.loc[at,'feature_ready'])
    out={k:{'ready':ready,'point':None,'lower':None} for k in ['clock','shared','equal','raw']}
    if not ready:return out
    if not daily['ready']:
        # There cannot be 60 mature calibration-error dates before 60 forecasts
        # existed. Preserve the already issued source policy during warmup.
        for info in out.values():
            info.update(alpha=[1.,1.],point=old['point'],lower=old['lower'],radius=old['radius'],
                source_radius=old['radius'],joint_radius=old['radius'],
                bootstrap_predictions=old['bootstrap_predictions'],source_fallback=True)
        return out
    raw=np.array(raw);cal=np.array(old['point']);c=CLOCKS.index(panel.loc[at,'clock'])
    x=np.array([*raw,c]);bootstrap_cal=np.array([raw+forecast(b['calibration'],x)[0] for b in daily['bootstrap']])
    for kind in out:
        alpha=np.array(daily['models'][kind]['alpha'][c]) if kind in ['clock','shared'] else np.full(2,.5 if kind=='equal' else 0.)
        point=mixture(raw,cal,alpha)
        draws=np.array([mixture(raw,z,np.array(b[kind]['alpha'][c]) if kind in ['clock','shared'] else alpha)
                        for z,b in zip(bootstrap_cal,daily['bootstrap'])])
        joint_radius=max(0.,float(np.quantile(np.max(draws-point,axis=1),.9,method='higher')))
        radius=max(float(old['radius']),joint_radius)
        out[kind].update(alpha=alpha,point=point,lower=point-radius,radius=radius,
            source_radius=old['radius'],joint_radius=joint_radius,bootstrap_predictions=draws,source_fallback=False)
    return out


def events(at,date,estimates,daily,record):
    primary=decide(estimates['clock'],daily['feedback'],True)
    actions={'adaptive_band':primary,'adaptive_point':decide(estimates['clock'],daily['feedback'],False),
        'shared_band':decide(estimates['shared'],daily['feedback'],True),
        'equal_band':decide(estimates['equal'],daily['feedback'],True),
        'raw_band':decide(estimates['raw'],daily['feedback'],True),'source_v35':record['events']['full_band'],
        'cash':{'ready':True,'action':'CASH','allocation':0.},'double_cost':primary}
    return {k:{**v,'at':at,'date':date} for k,v in actions.items()}


def load_source(path=None):
    p=read(SOURCE/'protocol.json');root=core.ROOT/p['forecast_source'] if path is None else Path(path)
    delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v35/evidence/delivery_manifest.json')
    hashes={}
    for key,folder in [('cache_outputs_sha256',''),('daily_records_sha256','daily'),('training_records_sha256','training')]:
        for f,h in delivery[key].items():
            name=str(Path(folder)/f);assert sha(root/name)==h;hashes[name]=h
    v=read(root/'verification.json');assert v['passed'] and v['results_sha256']==sha(root/'results.json')
    records={pd.Timestamp(r['at']):r for f in sorted((root/'training').glob('*.json')) if (r:=read(f))}
    return root,records,hashes
