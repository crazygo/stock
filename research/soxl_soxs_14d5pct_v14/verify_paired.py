"""Independent prefix, factors, paired outcomes, model arithmetic and capital audit."""
import argparse
import gzip
import hashlib
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_limits
from paired_entry import core, MODELS, BASE_COLUMNS, REGIME_COLUMNS, PANEL, restrict_market, load_panel
from verify_continuation import read, labels, key, near, raw_exit
from verify_uncertainty import book_check


def verify(run):
    run=Path(run); reg=read(run/'registration.json'); p=reg['preregistration']['protocol']; manifest=read(run/'input_manifest.json')
    for name,digest in reg['source_sha256'].items():
        current=hashlib.sha256((core.ROOT/name).read_bytes()).hexdigest()
        if current!=digest:
            # An audit-only repair must name exactly this file and preserve its
            # originally executed copy. Forecast/feature/ledger sources cannot change.
            amendment=read(run/'verification_amendment.json')
            assert (core.ROOT/name).resolve()==Path(__file__).resolve()
            assert amendment['file']==name and amendment['before_sha256']==digest and amendment['after_sha256']==current
            assert hashlib.sha256((run/'executed_sources'/Path(name).parent.name/Path(name).name).read_bytes()).hexdigest()==digest
    for f,field in [('panel.parquet','panel_sha256'),('features.parquet','feature_sha256'),('premarket_volumes.parquet','volume_sha256')]:
        assert hashlib.sha256((run/f).read_bytes()).hexdigest()==manifest[field]
    source=Path(manifest['source_feature_file']); assert hashlib.sha256(source.read_bytes()).hexdigest()==manifest['source_feature_sha256']
    cutoff=pd.Timestamp(manifest['observation_cutoff_exclusive']); original=core.load_market(); assert original.manifest==manifest['full_market']
    market=restrict_market(original,cutoff); bars,_,panel_manifest=load_panel(original.calendar); assert panel_manifest==manifest['panel_market']
    assert all((f.end_at<cutoff).all() and (f.available_at<cutoff).all() for f in market.bars.values())
    features=pd.read_parquet(run/'features.parquet'); panel=pd.read_parquet(run/'panel.parquet'); volumes=pd.read_parquet(run/'premarket_volumes.parquet')
    original_features=pd.read_parquet(source)
    pd.testing.assert_frame_equal(features,original_features[original_features.clock.eq('09:25:30')&(original_features.index<cutoff)])
    prefix_count=0
    for symbol in PANEL:
        f=bars[symbol]; local=f.end_at.dt.tz_convert(core.ET)
        f=f[(local.dt.hour*60+local.dt.minute>240)&(local.dt.hour*60+local.dt.minute<=565)]
        grouped={d:g for d,g in f.groupby('date')}
        for day in volumes.index:
            g=grouped[day]; t=day.tz_localize(core.ET)+pd.Timedelta(hours=9,minutes=25)
            g=g[(g.end_at<cutoff)&(g.available_at<cutoff)]
            expected=pd.date_range(day.tz_localize(core.ET)+pd.Timedelta(hours=4,minutes=5),t,freq='5min').tz_convert('UTC')
            assert g.index.equals(expected) and (g.available_at<=t+pd.Timedelta(seconds=30)).all()
            assert g.econ_volume.ge(0).all() and g.econ_volume.sum()>0
            near(volumes.loc[day,symbol],g.econ_volume.sum()); prefix_count+=1
    base=labels(run/'baseline_labels.json'); stress=labels(run/'stress_labels.json'); releases={}; matrix=[]; outcomes=[]
    breadth=2*features.breadth_ret5_positive_fraction.to_numpy()-1
    for i,(at,r) in enumerate(features.iterrows()):
        sv=max(r.SOXX_vol20,.0001);qv=max(r.QQQ_vol20,.0001)
        vals=[*np.clip([r.SOXX_ret1/sv,r.SOXX_ret5/sv/np.sqrt(5),r.SOXX_ret20/sv/np.sqrt(20),
            r.QQQ_ret5/qv/np.sqrt(5),r.SOXX_today_return/sv,r.breadth_today_mean/sv],-5,5),np.log(sv)]
        history=volumes[volumes.index<r.date].iloc[-20:]
        ready=i>=5 and len(history)==20 and history.notna().all().all() and volumes.loc[r.date].notna().all()
        assert bool(panel.loc[at,'feature_ready'])==ready
        if ready:
            ratio=volumes.loc[r.date].values/history.mean().values; log=np.clip(np.log(ratio),-3,3)
            today=np.array([r['leader_'+s+'_today'] for s in PANEL]); trend=np.array([r['leader_'+s+'_ret5'] for s in PANEL])
            relative=np.clip(r.sector_relative_5/sv/np.sqrt(5),-5,5); vm=log.mean()
            extra=[relative,np.clip(r.sector_relative_20/sv/np.sqrt(20),-5,5),breadth[i],
                2*r.breadth_ret20_positive_fraction-1,2*r.breadth_today_positive_fraction-1,
                np.sign(features.sector_relative_5.iloc[i-4:i+1]).mean(),breadth[i]-breadth[i-5:i].mean(),vm,
                np.mean((ratio>1)&(today>0))-np.mean((ratio>1)&(today<0)),np.mean(np.sign(trend)*log),relative*vm,breadth[i]*vm]
        else:extra=[np.nan]*12
        vals+=extra; matrix.append(vals)
        np.testing.assert_allclose(panel.loc[at,BASE_COLUMNS+REGIME_COLUMNS].astype(float),vals,atol=1e-11,rtol=1e-11,equal_nan=True)
        pair=[base[key(at,s)] for s in core.ETFS]; L,S=[np.log1p(x.get('net_return',np.nan)) for x in pair]
        C=(L+S)/2;D=(L-S)/2;outcomes.append([C,D,abs(D)])
        np.testing.assert_allclose(panel.loc[at,['C','D','M']].astype(float),[C,D,abs(D)],equal_nan=True)
        if ready and all(x['complete_window'] and x.get('label_available_at') is not None and np.isfinite(x.get('net_return',np.nan)) for x in pair):
            assert all(x['label_available_at']>=x['expiry_at']+pd.Timedelta(seconds=1) for x in pair)
            releases[r.date]=max(x['label_available_at'] for x in pair)
    matrix=np.array(matrix);outcomes=np.array(outcomes); parent=np.array(features.date); ids=np.array(features.index)
    predictions=read(run/'predictions.json'); histories={n:[] for n in MODELS}; recorded={n:{} for n in predictions}
    membership=forecasts=calchecks=refits=0; anchors=['2025-09-02','2025-12-01','2026-03-02']
    with threadpool_limits(limits=1):
        for i,(at,r) in enumerate(features.iterrows()):
            record=read(run/'training'/(str(r.date.date())+'.json')); cutoff_day=r.date.tz_localize(core.ET).tz_convert('UTC')
            assert pd.Timestamp(record['cutoff'])==cutoff_day
            dates=[d for d,t in sorted(releases.items()) if t<cutoff_day and d.tz_localize(core.ET)<cutoff_day]
            idx=np.flatnonzero(np.isin(parent,np.asarray(dates,dtype='datetime64[ns]'))); w=2.**(-np.arange(len(dates))[::-1]/126.)
            if len(w):w*=len(w)/w.sum()
            with gzip.open(run/'models'/(str(r.date.date())+'.pkl.gz'),'rb') as f:models=pickle.load(f)
            for name,(kind,regime) in MODELS.items():
                saved=record['routes'][name];info=saved['fit'];cal=saved['calibration'];e=saved['event'];model=models[name]
                assert e==predictions[name][i] and pd.Timestamp(e['at'])==at
                assert list(pd.to_datetime(info['dates']))==dates and list(pd.to_datetime(info['members'],utc=True))==list(ids[idx])
                np.testing.assert_allclose(info['parent_weights'],w,rtol=1e-12,atol=1e-12);near(info['weight_sum'],len(w))
                assert info['ready']==(len(dates)>=120);membership+=1
                if len(w):
                    near(info['parent_effective_n'],w.sum()**2/(w@w))
                    assert pd.Timestamp(saved['latest_label_release'])==max(releases[d] for d in dates)
                old=[a for a in histories[name] if pd.Timestamp(a['date']) in set(dates) and a['raw']][-120:]
                old_ids=[pd.Timestamp(a['at']) for a in old]
                assert list(pd.to_datetime(cal['members'],utc=True))==old_ids
                assert list(pd.to_datetime(cal['dates']))==[pd.Timestamp(a['date']) for a in old]
                assert cal['ready']==(len(old)>=60);maps={}
                for j,target in enumerate(['C','D']):
                    if not old:continue
                    x=np.array([a['raw'][target] for a in old]);y=outcomes[features.index.get_indexer(old_ids),j]
                    center=x.mean();scale=x.std() if x.std()>=1e-12 else 1.;z=(x-center)/scale;b=y.mean();m=z@(y-b)/(z@z+20.)
                    expected={'center':center,'scale':scale,'intercept':b,'coefficient':m,'score_min':x.min(),'score_max':x.max()}
                    for k,v in expected.items():near(cal['maps'][target][k],v)
                    maps[target]=expected
                calchecks+=1
                if model is not None:
                    cols=19 if regime else 7;X=matrix[idx,:cols];Q=matrix[[i],:cols];C,D,M=outcomes[idx].T
                    center=np.average(X,axis=0,weights=w);scale=np.sqrt(np.average((X-center)**2,axis=0,weights=w));scale[scale<1e-12]=1.
                    np.testing.assert_allclose(model['scaler'].mean_,center,rtol=1e-10,atol=1e-10)
                    np.testing.assert_allclose(model['scaler'].scale_,scale,rtol=1e-10,atol=1e-10)
                    Z=(X-center)/scale;R=(Q-center)/scale;cw=w*M if kind=='regret' else w.copy()
                    if cw.sum()>0:cw*=len(w)/cw.sum()
                    np.testing.assert_allclose(info['classification_weights'],cw,rtol=1e-10,atol=1e-10)
                    fallback=np.average(D>0,weights=cw) if cw.sum()>0 else .5;near(model['q'],fallback)
                    near(model['means']['C'],np.average(C,weights=w));near(model['means']['D'],np.average(D,weights=w))
                    h=model['heads'];a=float(R[0]@h['C'].coef_+h['C'].intercept_) if 'C' in h else model['means']['C']
                    q=float(1/(1+np.exp(-(R[0]@h['q'].coef_[0]+h['q'].intercept_[0])))) if 'q' in h else fallback
                    magnitude=max(0.,float(R[0]@h['M'].coef_+h['M'].intercept_)) if 'M' in h else None
                    d=magnitude*(2*q-1) if magnitude is not None else float(R[0]@h['D'].coef_+h['D'].intercept_) if 'D' in h else model['means']['D']
                    for k,v in [('C',a),('D',d),('q',q),('SOXL',a+d),('SOXS',a-d)]:near(e['raw'][k],v)
                    if magnitude is None:assert e['raw']['M'] is None
                    else:near(e['raw']['M'],magnitude)
                    forecasts+=1
                    if str(r.date.date()) in anchors:
                        for target,est in h.items():
                            fresh=LogisticRegression(C=.1,solver='lbfgs',max_iter=2000,tol=1e-10) if target=='q' else Ridge(alpha=100.,solver='svd')
                            fresh.fit(Z,{'C':C,'D':D,'M':M,'q':(D>0).astype(int)}[target],sample_weight=cw if target=='q' else w)
                            np.testing.assert_allclose(fresh.coef_,est.coef_,rtol=1e-8,atol=1e-8);np.testing.assert_allclose(fresh.intercept_,est.intercept_,rtol=1e-8,atol=1e-8);refits+=1
                else:assert not info['ready'] and not e['raw']
                variants=[(name,e,'paired')]
                if name==p['primary']:variants += [(n,predictions[n][i],mode) for n,mode in [('same_primary_raw','raw'),('same_primary_direction','direction')]]
                for n,event,mode in variants:
                    assert event['raw']==e['raw'];ready=bool(e['raw']) and cal['ready'];assert event['ready']==ready
                    if ready:
                        a,d=[event['raw'][k] if mode=='raw' else maps[k]['intercept']+maps[k]['coefficient']*(event['raw'][k]-maps[k]['center'])/maps[k]['scale'] for k in ['C','D']]
                        for k,v in [('C',a),('D',d),('SOXL',a+d),('SOXS',a-d)]:near(event['adjusted'][k],v)
                        top='SOXL' if d>=0 else 'SOXS';assert event['top_symbol']==top
                        assert event['action']==(top if mode=='direction' or a+abs(d)>0 else 'CASH')
                        assert event['within_calibration_range']=={k:m['score_min']<=event['raw'][k]<=m['score_max'] for k,m in maps.items()}
                    else:assert event['action']=='CASH' and event['adjusted'] is None
                    recorded[n][at]=event['action']
                histories[name].append(e)
            if (i+1)%75==0:print('verified through',r.date.date(),flush=True)
    exits=0
    for lab,multiple in [(base,1),(stress,2)]:
        for label in lab.values():
            if 'entry_fill_economic' in label:raw_exit(market,label,None,multiple);exits+=1
    results=read(run/'results.json');frozen=read(run/'decisions_frozen.json');books=[];settings=p[reg['phase']]
    months=settings.get('months',settings.get('regression_months'))
    for name,stats in results['routes'].items():
        actions={pd.Timestamp(e['at']):e['action'] for e in frozen['actions'][name]}
        for at,a in actions.items():
            if name in recorded:assert a==recorded[name][at]
            elif name=='double_cost':assert a==recorded[p['primary']][at]
            elif name=='cash':assert a=='CASH'
            elif name=='always_SOXL':assert a=='SOXL'
            elif name=='trend20':
                v=features.loc[at,'SOXX_ret20'];assert a==('SOXL' if v>0 else 'SOXS' if v<0 else 'CASH')
            else:raise ValueError(name)
        books.append(book_check(run,name,market,features,stress if name=='double_cost' else base,actions,2 if name=='double_cost' else 1,stats,months))
    result={'passed':True,'phase':reg['phase'],'premarket_prefix_checks':prefix_count,'feature_vectors_checked':len(features),
        'training_membership_checks':membership,'paired_forecast_checks':forecasts,'paired_calibration_checks':calchecks,
        'anchor_head_refits':refits,'anchors':anchors,'raw_exit_checks':exits,'capital_books':books,
        'results_sha256':hashlib.sha256((run/'results.json').read_bytes()).hexdigest(),
        'all_inputs_before_cutoff':True,'engineering_verification_is_not_effect_acceptance':True}
    core.dump(run/'verification.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
