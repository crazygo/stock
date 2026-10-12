"""Independent joint labels, daily information boundary, probability and capital audit."""
import argparse,gzip,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from joint_loss import *
from run_joint import sha
from verify_continuation import read,labels,key,near
from verify_feedback_v2 import terminal_audit
from verify_allocation import book_check
from audit_probability import check_fit,manual_probability


def scenario_check(panel,ids,at,forecast,name,saved):
    event=saved['event'];dist=saved['distribution'];u=saved['utility']
    if not forecast['ready']:
        assert dist is None and u is None and not event['ready'] and event['action']=='CASH' and event['allocation']==0
        return 0
    ids=ids[-120:];assert list(pd.to_datetime(dist['members'],utc=True))==ids
    assert list(pd.to_datetime(dist['dates']))==list(panel.loc[ids,'date'])
    if len(ids)<60 or not panel.loc[at,'feature_ready']:
        assert not dist['ready'] and event['action']=='CASH';return 0
    b=np.array([2**((i-len(ids)+1)/63) for i in range(len(ids))]);b/=b.sum();w=b.copy()
    np.testing.assert_allclose(dist['base_weights'],b,atol=1e-12,rtol=1e-10)
    if name!='joint_global':
        X=panel.loc[ids,STATE_COLUMNS].to_numpy(float);Q=panel.loc[at,STATE_COLUMNS].to_numpy(float)
        center=np.average(X,axis=0,weights=b);scale=np.sqrt(np.average((X-center)**2,axis=0,weights=b));scale[scale<1e-12]=1.
        ds=np.sum(((X-Q)/scale)**2,axis=1)/len(STATE_COLUMNS);bw=max(sorted(ds)[39],1e-12)
        logw=np.log(b)-ds/(2*bw);w=np.exp(logw-logw.max());w/=w.sum();k=dist['kernel']
        for field,value in [('center',center),('scale',scale),('distance',ds)]:np.testing.assert_allclose(k[field],value,atol=1e-10,rtol=1e-10)
        near(k['bandwidth'],bw)
    else:assert dist['kernel'] is None
    np.testing.assert_allclose(dist['prior'],w,atol=1e-12,rtol=1e-10);near(dist['prior_concentration'],1/sum(w*w))
    if 1/(w@w)<20:
        assert not dist['ready'] and event['action']=='CASH' and dist['reason']=='insufficient_prior_concentration';return 0
    y=panel.loc[ids,'joint_class'].to_numpy(int);mass=np.array([sum(w[y==c]) for c in range(4)])
    p=mass if name=='prior_only' else np.array(forecast['raw'] if name=='joint_raw' else forecast['adjusted'])
    np.testing.assert_allclose(dist['class_prior'],mass,atol=1e-12);np.testing.assert_allclose(dist['probabilities'],p,atol=1e-12)
    if any(p[c]>0 and mass[c]<=0 for c in range(4)):
        assert not dist['ready'] and event['action']=='CASH' and dist['reason']=='predicted_class_without_observed_payoffs';return 0
    q=np.array([w[i]*p[y[i]]/mass[y[i]] if mass[y[i]]>0 else 0. for i in range(len(ids))])
    np.testing.assert_allclose(dist['posterior'],q,atol=1e-12,rtol=1e-10);near(dist['posterior_concentration'],1/sum(q*q))
    np.testing.assert_allclose([sum(q[y==c]) for c in range(4)],p,atol=1e-12)
    r=panel.loc[ids,['rL','rS']].to_numpy(float);np.testing.assert_allclose(dist['returns'],r,atol=1e-12)
    if 1/(q@q)<20:
        assert not dist['ready'] and event['action']=='CASH' and dist['reason']=='insufficient_posterior_concentration';return 1
    assert dist['ready'] and event['ready'];fractions=[1.] if name in ['full_size','risk_neutral'] else [.25,.5,.75,1.]
    choices=[['CASH',0.]]+[[s,f] for f in fractions for s in core.ETFS];scores=[0.]
    for s,f in choices[1:]:
        wealth=1+f*r[:,core.ETFS.index(s)];score=q@(wealth-1 if name=='risk_neutral' else (wealth**-4-1)/-4);scores.append(float(score))
    assert u['choices']==choices and u['neutral']==(name=='risk_neutral')
    np.testing.assert_allclose(u['scores'],scores,atol=1e-12,rtol=1e-10)
    best=next(i for i,x in enumerate(scores) if x>=max(scores)-1e-12)
    assert u['selected_index']==best and [event['action'],event['allocation']]==choices[best]
    return 1


def verify(run):
    run=Path(run);reg=read(run/'registration.json');manifest=read(run/'input_manifest.json');p=reg['preregistration']['protocol']
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    source=Path(manifest['source_run']);label_source=Path(manifest['label_source'])
    for f,h in manifest['source_files_sha256'].items():assert sha(source/f)==h
    for f,h in manifest['label_files_sha256'].items():assert sha(label_source/f)==h
    original=core.load_market();assert original.manifest==manifest['full_market'];market=restrict_market(original,pd.Timestamp(manifest['observation_cutoff_exclusive']))
    panel=pd.read_parquet(run/'panel.parquet');features=pd.read_parquet(run/'features.parquet')
    pd.testing.assert_frame_equal(features,pd.read_parquet(source/'features.parquet'))
    pd.testing.assert_frame_equal(panel.drop(columns=['rL','rS','joint_class']),pd.read_parquet(source/'panel.parquet'))
    base=labels(label_source/'baseline_labels.json');stress=labels(label_source/'stress_labels.json');terminals={};truthchecks=0
    for at,r in panel.iterrows():
        pair=[base[key(at,s)] for s in core.ETFS];returns=np.array([x.get('net_return',np.nan) for x in pair])
        np.testing.assert_allclose(r[['rL','rS']].to_numpy(float),returns,atol=1e-12,equal_nan=True)
        cls=int(2*(returns[0]>0)+(returns[1]>0)) if np.isfinite(returns).all() else -1
        assert r.joint_class==cls
        if r.feature_ready:
            ts=[terminal_audit(market,b) for b in pair]
            if all(t is not None for t in ts):terminals[at]=max(ts);assert r.terminal_release==max(ts)
            else:assert pd.isna(r.terminal_release)
        truthchecks+=1
    predictions=read(run/'predictions.json');old=read(source/'predictions.json')['early_base_cal'];histories={'hgb':[],'logit':[]}
    memberships=equations=predchecks=scenariochecks=refits=0
    anchors={'2025-09-02','2026-01-05','2026-04-01','2026-06-01','2026-09-01'}
    with threadpool_limits(limits=1):
        for i,(at,r) in enumerate(panel.iterrows()):
            rec=read(run/'training'/(str(r.date.date())+'.json'));dc=r.date.tz_localize(core.ET).tz_convert('UTC')
            assert pd.Timestamp(rec['cutoff'])==dc
            past=list(panel[panel.feature_ready&(panel.date<r.date)&(panel.index<dc)].index);ids=[]
            for t in past:
                if t not in terminals or terminals[t]>=dc:break
                ids.append(t)
            expired=len(ids)<len(past) and panel.loc[past[len(ids)],'date']+pd.Timedelta(days=14)<=r.date
            with gzip.open(run/'models'/(str(r.date.date())+'.pkl.gz'),'rb') as f:models=pickle.load(f)
            for kind in ['hgb','logit']:
                f=rec['learners'][kind];info=f['fit'];cal=f['calibration'];fb=f['feedback'];model=models[kind]
                assert fb['expired_unresolved_prefix']==bool(expired) and fb['selected_early_dates']==len(ids)
                assert fb['prior_dates']==len(past) and fb['known_but_waiting']==sum(t in terminals and terminals[t]<dc for t in past[len(ids)+1:])
                assert info['columns']==COLUMNS and list(pd.to_datetime(info['members'],utc=True))==ids
                assert list(pd.to_datetime(info['dates']))==list(panel.loc[ids,'date'])
                do_refit=str(r.date.date()) in anchors
                equations+=check_fit(model['base'],info,panel.loc[ids,COLUMNS].to_numpy(float),panel.loc[ids,'joint_class'].to_numpy(int),126,120,do_refit)
                raw=manual_probability(model['base'],panel.loc[[at],COLUMNS].to_numpy(float)) if r.feature_ready else None
                if raw is None:assert f['raw'] is None
                else:np.testing.assert_allclose(f['raw'],raw,atol=1e-10,rtol=1e-10);predchecks+=1
                allowed=set(ids);h=[e for e in histories[kind] if e['at'] in allowed and e['raw'] is not None][-120:]
                hids=[e['at'] for e in h];assert list(pd.to_datetime(cal['members'],utc=True))==hids
                assert list(pd.to_datetime(cal['dates']))==list(panel.loc[hids,'date'])
                X=np.log(np.maximum(np.array([e['raw'] for e in h]),1e-6)) if h else np.empty((0,4))
                equations+=check_fit(model['calibration'],cal,X,panel.loc[hids,'joint_class'].to_numpy(int),63,60,do_refit)
                adjusted=manual_probability(model['calibration'],np.log(np.maximum(raw,1e-6))[None,:]) if raw is not None else None
                if adjusted is None:assert f['adjusted'] is None
                else:np.testing.assert_allclose(f['adjusted'],adjusted,atol=1e-10,rtol=1e-10);predchecks+=1
                assert f['ready']==(raw is not None and adjusted is not None and not expired)
                histories[kind].append({'at':at,'raw':f['raw']});memberships+=1
                refits+=int(do_refit and info['ready'])+int(do_refit and cal['ready'])
            for n,saved in rec['capital'].items():
                f=rec['learners']['logit' if n=='joint_logit' else 'hgb']
                assert saved['event']==predictions[n][i]
                scenariochecks+=scenario_check(panel,ids,at,f,n,saved)
            assert predictions['source_early'][i]==old[i] and predictions['double_cost'][i]==predictions[p['primary']][i]
            assert predictions['cash'][i]['action']=='CASH' and predictions['cash'][i]['allocation']==0.
            if (i+1)%75==0:print('verified through',r.date.date(),flush=True)
    results=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    settings=p[reg['phase']];months=settings.get('months',settings.get('regression_months'));books=[]
    for n,stats in results['routes'].items():
        wanted=[e for e in predictions[n] if settings['evaluation_start']<=e['date'][:10]<=settings['evaluation_end']]
        assert frozen['routes'][n]==wanted
        books.append(book_check(run,n,market,features,stress if n=='double_cost' else base,wanted,2 if n=='double_cost' else 1,stats,months))
    for suffix in ['equity','trades']:
        a=pd.read_csv(run/('source_early_'+suffix+'.csv'));b=pd.read_csv(source/('early_base_cal_'+suffix+'.csv'))
        pd.testing.assert_frame_equal(a[b.columns],b,atol=1e-12,rtol=1e-12)
    if reg['phase']=='cases':
        previous=read(Path(reg['authorization']['run'])/'predictions.json')
        for n in ROUTES:assert [e for e in predictions[n] if e['date'][:10]<'2026-05-01']==previous[n]
    receipt={'passed':True,'phase':reg['phase'],'independent_paired_classes':truthchecks,'daily_model_memberships':memberships,
        'independent_scaling_and_classifier_checks':equations,'manual_tree_softmax_predictions':predchecks,'fresh_anchor_classifier_refits':refits,
        'joint_scenario_mass_and_utility_checks':scenariochecks,'capital_books':books,'original_v21_forecasts_and_ledger_exact':True,
        'original_exit_contract_and_asof_receipts_reused_unchanged':True,'results_sha256':sha(run/'results.json'),
        'engineering_verification_is_not_return_acceptance':True}
    core.dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);verify(parser.parse_args().run)
