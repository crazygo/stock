"""Past-only state innovations, matched causal input controls and old contract."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v58'))
import shared_learning as previous
from shared_learning import (core,np,pd,read,dump,sha,labels,day_cutoff,restrict_market,development_gate,tag,registry,fit_day,issue,decide,CLOCKS,OLD_CLOCKS,load_sources,load_panel,PANEL_ROOT,original_fit,old55_columns)
SOURCE=Path(__file__).resolve().parent
sys.path.insert(0,str(SOURCE))
STATES=['relative_5','breadth_5','sector_relative_activity']
INNOV=['state_innovation_'+str(i) for i in range(3)]
DIFF=['state_difference_'+str(i) for i in range(3)]
VARIANTS=['innovation','difference','matched']
ROUTES=['innovation_band','innovation_point','difference_band','matched_band','source_v56','source_v55','premarket_only','cash','double_cost']


def columns(clock,kind='matched'):
    return previous.columns(clock)+(INNOV if kind=='innovation' else DIFF if kind=='difference' else [])


def verified_source():
    run,source_sha,_=previous.verified_source();path=core.ROOT/'research/soxl_soxs_14d5pct_v58/evidence/delivery_manifest.json';m=read(path)
    for f,h in m['delivery_files_sha256'].items():assert sha(path.parents[1]/f)==h,f
    v=read(path.parent/'verification.json');assert v['passed'] and v['results_sha256']==sha(path.parent/'results.json')
    return run,source_sha,sha(path)


def state_pairs(panel,sessions):
    positions={d:i for i,d in enumerate(sessions)};out=[]
    for at,row in panel.iterrows():
        j=positions[row.date];before=sessions[j-1] if j else None
        lag=pd.Timestamp(str(before.date())+' '+row.clock,tz=core.ET).tz_convert('UTC') if before is not None else None
        good=bool(lag in panel.index and np.isfinite(panel.loc[at,STATES].to_numpy(float)).all() and np.isfinite(panel.loc[lag,STATES].to_numpy(float)).all())
        out.append({'at':at,'date':row.date,'clock':row.clock,'lag_at':lag,'ready':good})
    return out


def fit_state(panel,parents):
    n=len(parents);out={'ready':n>=60,'members':[p['at'] for p in parents],'lag_members':[p['lag_at'] for p in parents],'n':n}
    if n<60:return out
    x=np.ascontiguousarray(panel.loc[out['lag_members'],STATES].to_numpy(float));y=np.ascontiguousarray(panel.loc[out['members'],STATES].to_numpy(float))
    w=2.**(-np.arange(n)[::-1]/126.);w*=n/w.sum();mean=np.average(x,axis=0,weights=w);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale[scale<1e-12]=1.
    z=(x-mean)/scale;ym=np.average(y,axis=0,weights=w);coef=np.linalg.solve(z.T@(w[:,None]*z)+20*np.eye(3),z.T@(w[:,None]*(y-ym)))
    residual=y-(ym+z@coef);rs=np.sqrt(np.average(residual**2,axis=0,weights=w));ds=np.sqrt(np.average((y-x)**2,axis=0,weights=w));rs[rs<1e-12]=1.;ds[ds<1e-12]=1.
    out.update(weights=w.tolist(),input_center=mean.tolist(),input_scale=scale.tolist(),output_center=ym.tolist(),coefficients=coef.tolist(),residual_scale=rs.tolist(),difference_scale=ds.tolist())
    return out


def state_query(panel,pair,fit):
    ready=bool(pair['ready'] and fit['ready']);out={'ready':ready,'prediction':None,'innovation':None,'difference':None}
    if not ready:return out
    now=panel.loc[pair['at'],STATES].to_numpy(float);lag=panel.loc[pair['lag_at'],STATES].to_numpy(float)
    prediction=np.array(fit['output_center'])+((lag-fit['input_center'])/fit['input_scale'])@np.array(fit['coefficients'])
    out.update(prediction=prediction.tolist(),innovation=((now-prediction)/fit['residual_scale']).tolist(),difference=((now-lag)/fit['difference_scale']).tolist())
    return out


def build_state_features(panel,sessions):
    assert panel.index.is_unique and panel.index.is_monotonic_increasing
    pairs=state_pairs(panel,sessions);byclock={c:[] for c in CLOCKS};rows=[];receipts=[]
    for pair in pairs:
        at=pair['at'];clock=pair['clock'];cutoff=day_cutoff(at)
        parents=[p for p in byclock[clock] if p['ready'] and p['at']<cutoff and p['lag_at']<cutoff]
        fit=fit_state(panel,parents);value=state_query(panel,pair,fit)
        rows.append({'at':at,'state_ready':value['ready'],**dict(zip(INNOV,value['innovation'] if value['ready'] else [np.nan]*3)),**dict(zip(DIFF,value['difference'] if value['ready'] else [np.nan]*3))})
        receipts.append({'at':at,'cutoff':cutoff,'pair':pair,'fit':fit,'query':value});byclock[clock].append(pair)
    derived=panel.join(pd.DataFrame(rows).set_index('at'));derived['feature_ready']=panel.feature_ready&derived.state_ready
    return derived,receipts


def fit_bases(panel,at):
    models={};fits={};raws={}
    for k in VARIANTS:models[k],fits[k],raws[k]=original_fit(panel,at,columns(panel.loc[at,'clock'],k))
    return models,fits,raws


def events(at,date,clock,estimates,daily,fits,old56,old55):
    fb={k:{**daily[k]['feedback'],'expired_unresolved_prefix':bool(daily[k]['feedback']['expired_unresolved_prefix'] or fits[k]['feedback']['expired_unresolved_prefix'])} for k in VARIANTS}
    primary=decide(estimates['innovation'],fb['innovation'],True);cash={'ready':True,'action':'CASH','allocation':0.}
    choices={'innovation_band':primary,'innovation_point':decide(estimates['innovation'],fb['innovation'],False),'difference_band':decide(estimates['difference'],fb['difference'],True),'matched_band':decide(estimates['matched'],fb['matched'],True),'source_v56':old56,'source_v55':old55,'premarket_only':primary if clock==CLOCKS[0] else cash,'cash':cash,'double_cost':primary}
    return {k:{**e,'at':at,'date':date} for k,e in choices.items()}
