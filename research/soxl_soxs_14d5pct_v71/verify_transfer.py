"""Separate normal-equation, chronological calibration, and capital verification."""
import argparse
import gzip
import json
from threadpoolctl import threadpool_limits
from transfer_learning import *
from audit_dense_calibration import daily_check, input_rows, predict as calibration_predict
from audit_capital import book_check
from audit_statistics import audit_statistics


def reference_models(panels, ids):
    """Solve full intercept normal equations, rather than augmented least squares."""
    n=len(ids);names=sorted(panels);k=len(names);width=len(FIELDS)
    weights=np.array([2.**((i-n+1)/126.) for i in range(n)]);weights*=n/sum(weights)
    X=np.concatenate([panels[s].loc[ids,FIELDS].to_numpy(float) for s in names])
    Y=np.concatenate([panels[s].loc[ids,['C','D']].to_numpy(float) for s in names])
    w=np.tile(weights/k,k);group=np.repeat(np.arange(k),n);onehot=np.eye(k)[group]
    centers=[];scales=[];Z=np.zeros_like(X)
    for i in range(k):
        mask=group==i;v=w[mask];mean=np.sum(v[:,None]*X[mask],axis=0)/sum(v)
        sd=np.sqrt(np.sum(v[:,None]*(X[mask]-mean)**2,axis=0)/sum(v));sd[sd<1e-12]=1
        centers.append(mean);scales.append(sd);Z[mask]=(X[mask]-mean)/sd
    out={}
    for mode in VARIANTS:
        if mode=='pooled':
            slopes=Z;pen=100*np.eye(width)
        else:
            slopes=(onehot[:,:,None]*Z[:,None,:]).reshape(len(X),k*width)
            mean=np.tile(np.eye(width)/k,(1,k));pen=100*(mean.T@mean) if mode=='partial' else np.zeros((k*width,k*width))
            for i in range(k):
                selector=np.zeros((width,k*width));selector[:,i*width:(i+1)*width]=np.eye(width)
                delta=selector-mean if mode=='partial' else selector
                pen+=(300 if mode=='partial' else 100)/k*(delta.T@delta)
        A=np.column_stack([onehot,slopes]);P=np.zeros((A.shape[1],A.shape[1]));P[k:,k:]=pen
        beta=np.linalg.solve(A.T@(w[:,None]*A)+P,A.T@(w[:,None]*Y))
        coeff=np.repeat(beta[k:][None,:,:],k,axis=0) if mode=='pooled' else beta[k:].reshape(k,width,2)
        out[mode]={'names':names,'centers':centers,'scales':scales,'intercepts':beta[:k],'coefficients':coeff,'weights':w,'penalty':pen}
    i=names.index('semiconductor');z=Z[group==i];y=Y[group==i]
    A=np.column_stack([np.ones(n),z]);P=np.diag([0]+[100]*width)
    solo=np.linalg.solve(A.T@(weights[:,None]*A)+P,A.T@(weights[:,None]*y))
    np.testing.assert_allclose(solo[0],out['independent']['intercepts'][i],atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(solo[1:],out['independent']['coefficients'][i],atol=1e-10,rtol=1e-10)
    return out


def reference_parents(panels,market,at):
    today=at.tz_convert(core.ET).tz_localize(None).normalize();cut=day_cutoff(at);prior=[]
    for day in sorted(set(panels['semiconductor'].date)):
        if day>=today:break
        ids=[pd.Timestamp(str(day.date())+' '+c,tz=core.ET).tz_convert('UTC') for c in CLOCKS if pd.Timestamp(str(day.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[day]['close_at'])]
        frames=[p[p.date==day] for p in panels.values()]
        if not all(list(f.index)==ids and f.feature_ready.all() for f in frames):continue
        rel=max(f.terminal_release.max() for f in frames) if all(f.terminal_release.notna().all() for f in frames) else None
        prior.append({'date':day,'members':ids,'feature_ready':True,'release':rel})
    parents=[]
    for row in prior:
        if row['release'] is None or row['release']>=cut:break
        parents.append(row)
    pending=prior[len(parents)] if len(parents)<len(prior) else None
    feedback={'cutoff':cut,'real_dates':len(parents),'first_pending':pending['date'] if pending else None,
              'expired_unresolved_prefix':bool(pending and pending['date']+pd.Timedelta(days=14)<=today)}
    return parents,feedback


def canonical(value):
    from run_planning import json_ready
    return json.loads(json.dumps(json_ready(value),default=str))


def reference_estimate(panel,at,raw,daily,models):
    ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready'];out={'ready':bool(ready),'point':None,'lower':None}
    if ready:
        x=input_rows(panel,{at:raw},[at],CLOCKS);point=raw+calibration_predict(models['point_model'],x)[0]
        draws=np.array([raw+calibration_predict(m,x)[0] for m in models['bootstrap_models']]);radius=max(0.,sorted(np.max(draws-point,axis=1))[179])
        out.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=draws.tolist())
    return out


def check_events(at,row,estimates,daily,feedback):
    out={}
    for kind in VARIANTS:
        for suffix in ['band','point']:
            info=estimates[kind];e={'ready':False,'action':'CASH','allocation':0.,'values':None}
            if info['ready'] and not daily[kind]['feedback']['expired_unresolved_prefix'] and not feedback['expired_unresolved_prefix']:
                v=np.array(info['lower' if suffix=='band' else 'point']);j=int(v[1]>v[0]);buy=v[j]>1e-12
                e.update(ready=True,action=['SOXL','SOXS'][j] if buy else 'CASH',allocation=float(buy),values=v.tolist(),top_symbol=['SOXL','SOXS'][j],buy_advantage=float(v[j]))
            out[kind+'_'+suffix]=e
    out['premarket_only']=dict(out['partial_band']) if row.clock==CLOCKS[0] else {'ready':True,'action':'CASH','allocation':0.,'values':None}
    out['cash']={'ready':True,'action':'CASH','allocation':0.,'values':None};out['double_cost']=dict(out['partial_band'])
    return {k:dict(at=at,date=row.date,clock=row.clock,**e) for k,e in out.items()}


def verify(run):
    run=Path(run);registration=read(run/'registration.json');prereg=registration['preregistration'];p=prereg['protocol']
    assert registration['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in prereg['code_sha256'].items():assert sha(ROOT/f)==h,f
    manifest=read(run/'input_manifest.json')
    for f,h in manifest['derived_files_sha256'].items():assert sha(run/f)==h
    assert sha(run/'input_manifest.json')==prereg['input_manifest_sha256']
    assert sha(run/'input_audit.json')==prereg['input_audit_sha256'] and read(run/'input_audit.json')['passed']
    market=load_market();assert market.manifest==manifest['market_manifest']
    panels={k:pd.read_parquet(run/(k+'_panel.parquet')) for k in PAIRS};panel=panels['semiconductor']
    predictions=read(run/'predictions.json');histories={k:{} for k in VARIANTS};previous=None
    base_checks=cal_checks=days=action_checks=0
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel.iterrows()):
            record=read(run/'training'/(tag(at)+'.json'))
            assert pd.Timestamp(record['at'])==at and pd.Timestamp(record['cutoff'])==day_cutoff(at)
            if row.date!=previous:
                filename=str(row.date.date());saved=read(run/'daily'/(filename+'.json'));parents,feedback=reference_parents(panels,market,at)
                assert saved['parents']==canonical(parents) and saved['feedback']==canonical(feedback)
                with gzip.open(run/'models'/(filename+'.json.gz'),'rt') as f:stored=json.load(f)
                refs={}
                for c in CLOCKS:
                    ids=[t for r in parents for t in r['members'] if panel.loc[t,'clock']==c] if len(parents)>=120 else []
                    assert saved['members'][c]==canonical(ids)
                    refs[c]=reference_models(panels,ids) if len(ids)>=120 else {}
                    assert set(refs[c])==set(stored[c])
                    for k,ref in refs[c].items():
                        model=stored[c][k];assert model['names']==ref['names'] and model['distinct_dates']==len(ids)
                        for field in ['centers','scales','intercepts','coefficients','penalty']:np.testing.assert_allclose(model[field],ref[field],rtol=1e-9,atol=1e-10)
                        # Stored rows retain insertion pair order; each group has the same date weights.
                        np.testing.assert_allclose(model['weights'],ref['weights'],atol=1e-12,rtol=1e-12)
                        np.testing.assert_allclose(model['pair_mass'],[1/3]*3,atol=1e-12,rtol=1e-12);base_checks+=1
                daily=saved['calibration'];calrefs={}
                for k in VARIANTS:calrefs[k],count=daily_check(panel,market,histories[k],at,daily[k]);cal_checks+=count
                previous=row.date;days+=1
            estimates={}
            for k in VARIANTS:
                raw=None
                if k in refs[row.clock] and row.feature_ready:
                    model=refs[row.clock][k];j=model['names'].index('semiconductor');z=(row[FIELDS].to_numpy(float)-model['centers'][j])/model['scales'][j]
                    cd=model['intercepts'][j]+z@model['coefficients'][j];raw=np.array([cd[0]+cd[1],cd[0]-cd[1]])
                if raw is None:assert record['raws'][k] is None
                else:np.testing.assert_allclose(record['raws'][k],raw,atol=1e-10,rtol=1e-9)
                estimates[k]=reference_estimate(panel,at,raw,daily[k],calrefs[k]);got=record['estimates'][k]
                assert got['ready']==estimates[k]['ready']
                if got['ready']:
                    for f in ['point','lower','radius','bootstrap_predictions']:np.testing.assert_allclose(got[f],estimates[k][f],atol=1e-9,rtol=1e-8)
                else:assert got==estimates[k]
                histories[k][at]=raw
            expected=check_events(at,row,estimates,daily,feedback)
            for k,e in expected.items():
                got=record['events'][k];assert got==predictions[k][i]
                for f in ['at','date','clock','ready','action','allocation']:assert got[f]==canonical(e[f])
                for f in ['values','buy_advantage']:
                    if e.get(f) is not None:np.testing.assert_allclose(got[f],e[f],atol=1e-9,rtol=1e-8)
                action_checks+=1
            if (i+1)%400==0:print('Verified',i+1,'/',len(panel),at,flush=True)
    frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json') and frozen['new_capital_outcomes_seen'] is False
    result=read(run/'results.json');start=p['development']['evaluation_start'];end=p['development']['evaluation_end'];months=p['development']['months']
    features=pd.read_parquet(run/'features.parquet');test=features[features.date.between(start,end)]
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(test.index)==expected and result['coverage']['complete']
    base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json');books=[]
    for k in ROUTES:
        es=[e for e in predictions[k] if start<=str(pd.Timestamp(e['date']).date())<=end];assert frozen['routes'][k]==es
        books.append(book_check(run,k,market,test,stress if k=='double_cost' else base,es,2 if k=='double_cost' else 1,result['routes'][k],months,require_full_cash=False))
    stat_checks=audit_statistics(run);gate=development_gate(result['primary'],True,p)
    assert result['primary_name']==p['primary'] and result['primary']==result['routes'][p['primary']] and result['development_gate']==gate
    assert result['case_regression']['executed'] is False and result['goal_complete'] is False
    receipt={'passed':True,'results_sha256':sha(run/'results.json'),'source_sha256':sha(Path(__file__)),
             'prediction_rows':len(panel),'dates':days,'base_models_normal_equations':base_checks,
             'calibration_equation_checks':cal_checks,'route_actions':action_checks,'capital_books':books,
             'statistics_checks':stat_checks,'independent_equals_solo':True,'development_gate':gate,
             'strategy_independent_validation':False,'goal_complete':False}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();verify(a.run)
