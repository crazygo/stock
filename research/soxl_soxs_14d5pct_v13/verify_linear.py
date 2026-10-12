"""Independent feature, weight, daily fit, calibration, execution and capital audit."""
import argparse
import gzip
import hashlib
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from linear_entry import core, MODELS, ENTRY_COLUMNS, FACTOR_COLUMNS, restrict_market
from verify_continuation import read, labels, key, near, raw_exit, independent_state
from verify_uncertainty import book_check


def verify(run):
    run = Path(run); reg = read(run/'registration.json'); p = reg['preregistration']['protocol']; manifest = read(run/'input_manifest.json')
    for name,digest in reg['source_sha256'].items(): assert hashlib.sha256((core.ROOT/name).read_bytes()).hexdigest()==digest
    assert hashlib.sha256(Path(manifest['source_feature_file']).read_bytes()).hexdigest()==manifest['source_feature_sha256']
    assert hashlib.sha256((run/'panel.parquet').read_bytes()).hexdigest()==manifest['panel_sha256']
    cutoff = pd.Timestamp(manifest['observation_cutoff_exclusive']); original = core.load_market(); assert original.manifest==manifest['full_market']
    market = restrict_market(original,cutoff)
    assert all((f.end_at<cutoff).all() and (f.available_at<cutoff).all() for f in market.bars.values())
    features = pd.read_parquet(run/'features.parquet'); panel = pd.read_parquet(run/'panel.parquet')
    assert (features.index<cutoff).all() and features.clock.eq('09:25:30').all()
    base = labels(run/'baseline_labels.json'); stress = labels(run/'stress_labels.json'); releases = {}; matrices = {'factor':[],'full':[]}; ids = []; parent_dates = []; outcomes = []; hits = []
    for at,row in features.iterrows():
        pair = [base[key(at,s)] for s in core.ETFS]
        if all(x['complete_window'] and x.get('label_available_at') is not None and np.isfinite(x.get('net_return',np.nan)) for x in pair):
            assert all(x['label_available_at']>=x['expiry_at']+pd.Timedelta(seconds=1) for x in pair)
            releases[row.date] = max(x['label_available_at'] for x in pair)
        for s in core.ETFS:
            sign = 1 if s=='SOXL' else -1; sv=max(float(row.SOXX_vol20),.0001); qv=max(float(row.QQQ_vol20),.0001)
            vals = [sign, *np.clip([sign*row.SOXX_ret1/sv,sign*row.SOXX_ret5/(sv*np.sqrt(5)),
                sign*row.SOXX_ret20/(sv*np.sqrt(20)),sign*row.QQQ_ret5/(qv*np.sqrt(5)),
                sign*row.SOXX_today_return/sv,sign*row.breadth_today_mean/sv],-5,5),np.log(max(row[s+'_vol20'],.0001))]
            state = independent_state(row,s); full = [state[c] for c in ENTRY_COLUMNS]; ident=key(at,s)
            np.testing.assert_allclose(panel.loc[ident,['factor_'+c for c in FACTOR_COLUMNS]].astype(float),vals,atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(panel.loc[ident,['full_'+c for c in ENTRY_COLUMNS]].astype(float),full,atol=1e-12,rtol=1e-12)
            matrices['factor'].append(vals); matrices['full'].append(full); ids.append(ident); parent_dates.append(row.date)
            x=base[ident]; outcomes.append(np.log1p(x['net_return']) if np.isfinite(x.get('net_return',np.nan)) else np.nan); hits.append(x.get('target_hit',np.nan))
    for name in matrices: matrices[name]=np.asarray(matrices[name],dtype=float)
    parent_dates=np.asarray(parent_dates); outcomes=np.asarray(outcomes); hits=np.asarray(hits); ids=np.asarray(ids)
    predictions=read(run/'predictions.json'); histories={name:[] for name in MODELS}; membership=forecasts=refits=calchecks=0
    anchors=['2025-09-02','2025-12-01','2026-03-02']; recorded_actions={name:{} for name in predictions}
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            record=read(run/'training'/(str(row.date.date())+'.json')); today=row.date.tz_localize(core.ET).tz_convert('UTC')
            assert pd.Timestamp(record['cutoff'])==today
            dates=[d for d,t in sorted(releases.items()) if t<today and d.tz_localize(core.ET)<today]
            idx=np.flatnonzero(np.isin(parent_dates,dates)); current=np.array([2*i,2*i+1])
            with gzip.open(run/'models'/(str(row.date.date())+'.pkl.gz'),'rb') as f: models=pickle.load(f)
            for name,(kind,group,recent) in MODELS.items():
                saved=record['routes'][name]; info=saved['fit']; cal=saved['calibration']; event=saved['event']; model=models[name]
                assert event==predictions[name][i] and pd.Timestamp(event['at'])==at
                assert list(pd.to_datetime(info['dates']))==dates and info['members']==list(ids[idx]); membership+=1
                weights=2.**(-np.arange(len(dates))[::-1]/126.) if recent else np.ones(len(dates))
                if len(weights): weights*=len(dates)/weights.sum()
                np.testing.assert_allclose(info['parent_weights'],weights,atol=1e-12,rtol=1e-12)
                w=np.repeat(weights/2,2); near(info['weight_sum'],len(dates))
                if len(dates):
                    near(info['parent_effective_n'],weights.sum()**2/(weights@weights))
                    assert pd.Timestamp(saved['latest_label_release'])==max(releases[d] for d in dates)
                assert info['ready']==(len(dates)>=120)
                old=[e for e in histories[name] if pd.Timestamp(e['date']) in set(dates) and e['top_symbol'] is not None][-120:]
                old_ids=[key(e['at'],e['top_symbol']) for e in old]
                assert cal['members']==old_ids and list(pd.to_datetime(cal['dates']))==[pd.Timestamp(e['date']) for e in old]
                assert cal['ready']==(len(old)>=60)
                expected_cal=None
                if old:
                    x=np.array([e['raw'][e['top_symbol']]['score'] for e in old]); y=np.array([np.log1p(base[k]['net_return']) for k in old_ids])
                    center=x.mean(); scale=x.std() if x.std()>=1e-12 else 1.; z=(x-center)/scale; intercept=y.mean()
                    rawcoef=(z@(y-intercept))/(z@z+20.); coefficient=max(0.,rawcoef); offset=(y-x).mean()
                    for field,value in [('center',center),('scale',scale),('intercept',intercept),('coefficient',coefficient),('unconstrained_coefficient',rawcoef),('offset',offset),('score_min',x.min()),('score_max',x.max())]: near(cal[field],value)
                    expected_cal=(center,scale,intercept,coefficient,offset,x.min(),x.max())
                calchecks+=1
                if model is not None:
                    X=matrices[group][idx]; Q=matrices[group][current]; y=outcomes[idx]; h=hits[idx].astype(int)
                    center=np.average(X,axis=0,weights=w); variance=np.average((X-center)**2,axis=0,weights=w); scale=np.sqrt(variance); scale[scale<1e-12]=1.
                    np.testing.assert_allclose(model['scaler'].mean_,center,atol=1e-10,rtol=1e-10)
                    np.testing.assert_allclose(model['scaler'].scale_,scale,atol=1e-10,rtol=1e-10)
                    Z=(X-center)/scale; R=(Q-center)/scale; estimator=model['estimator']
                    for j,s in enumerate(core.ETFS):
                        mask=np.arange(len(idx))%2==j; won=mask&(h==1); lost=mask&(h==0)
                        means={'p':np.average(h[mask],weights=w[mask]),'value':np.average(y[mask],weights=w[mask]),
                            'win':np.average(y[won],weights=w[won]) if won.any() else 0.,'failure':np.average(y[lost],weights=w[lost]) if lost.any() else 0.}
                        for field,value in means.items(): near(model['means'][s][field],value); near(info['means'][s][field],value)
                        prob=float(1/(1+np.exp(-(R[j]@estimator.coef_[0]+estimator.intercept_[0])))) if kind=='logit' and estimator is not None else means['p']
                        score=prob*means['win']+(1-prob)*means['failure'] if kind=='logit' else float(R[j]@estimator.coef_+estimator.intercept_) if estimator is not None else means['value']
                        for field,value in [('p',prob),('score',score),('baseline_p',means['p']),('win',means['win']),('failure',means['failure'])]: near(event['raw'][s][field],value)
                        forecasts+=1
                    if str(row.date.date()) in anchors and estimator is not None:
                        fitted=(LogisticRegression(C=.1,solver='lbfgs',max_iter=2000,tol=1e-10) if kind=='logit' else Ridge(alpha=100.,solver='svd'))
                        fitted.fit(Z,h if kind=='logit' else y,sample_weight=w)
                        if kind=='logit': np.testing.assert_allclose(fitted.predict_proba(R),estimator.predict_proba(R),rtol=1e-9,atol=1e-9)
                        else: np.testing.assert_allclose(fitted.predict(R),estimator.predict(R),rtol=1e-9,atol=1e-9)
                        refits+=1
                else: assert not info['ready'] and not event['raw']
                variants=[(name,event,'conditional')]
                if name==p['primary']: variants += [(n,predictions[n][i],mode) for n,mode in [('same_primary_offset','offset'),('same_primary_raw','raw')]]
                for n,e,mode in variants:
                    assert e['raw']==event['raw']; top=max(core.ETFS,key=lambda s:e['raw'][s]['score']) if e['raw'] else None
                    assert e['top_symbol']==top; ready=top is not None and cal['ready']; assert e['ready']==ready
                    if ready:
                        c,scale,b,m,offset,low,high=expected_cal; v=e['raw'][top]['score']
                        value=b+m*(v-c)/scale if mode=='conditional' else v+offset if mode=='offset' else v
                        near(e['adjusted'],value); assert e['action']==(top if value>0 else 'CASH')
                        assert e['within_calibration_score_range']==(low<=v<=high)
                    else: assert e['action']=='CASH' and e['adjusted'] is None
                    recorded_actions[n][at]=e['action']
                histories[name].append(event)
            if (i+1)%75==0: print('verified through',row.date.date(),flush=True)
    exits=0
    for lab,multiple in [(base,1),(stress,2)]:
        for label in lab.values():
            if 'entry_fill_economic' in label: raw_exit(market,label,None,multiple); exits+=1
    results=read(run/'results.json'); frozen=read(run/'decisions_frozen.json'); books=[]; settings=p[reg['phase']]
    months=settings.get('months',settings.get('regression_months'))
    for name,stats in results['routes'].items():
        actions={pd.Timestamp(e['at']):e['action'] for e in frozen['actions'][name]}
        for at,a in actions.items():
            if name in recorded_actions: assert a==recorded_actions[name][at]
            elif name=='double_cost': assert a==recorded_actions[p['primary']][at]
            elif name=='cash': assert a=='CASH'
            elif name=='trend20':
                v=features.loc[at,'SOXX_ret20']; assert a==('SOXL' if v>0 else 'SOXS' if v<0 else 'CASH')
            else: raise ValueError(name)
        books.append(book_check(run,name,market,features,stress if name=='double_cost' else base,actions,2 if name=='double_cost' else 1,stats,months))
    result={'passed':True,'phase':reg['phase'],'feature_vectors_checked':len(ids),'training_membership_checks':membership,
        'candidate_forecast_checks':forecasts,'selected_calibration_checks':calchecks,'anchor_model_refits':refits,'anchors':anchors,
        'raw_exit_checks':exits,'capital_books':books,'observation_cutoff_exclusive':cutoff,
        'results_sha256':hashlib.sha256((run/'results.json').read_bytes()).hexdigest(),'all_inputs_before_cutoff':True,
        'engineering_verification_is_not_effect_acceptance':True}
    core.dump(run/'verification.json',result); print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True); verify(p.parse_args().run)
