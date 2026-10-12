"""Independent prefix, weighted normal equations, bands and capital audit."""
import argparse
from pathlib import Path
from threadpoolctl import threadpool_limits
from reliability import core,STATE_COLUMNS,restrict_market
from run_planning import read,labels,dump,sha
from verify_continuation import key,near
from verify_feedback_v2 import terminal_audit
from audit_books import book_check
import numpy as np
import pandas as pd


def close(a,b):np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-8,equal_nan=True)


def solve(X,Y,w,x):
    w=w*len(w)/sum(w);mean=np.average(Y,axis=0,weights=w)
    if not X.shape[1]:return mean,{'center':[],'scale':[],'coef':[[],[]],'intercept':mean.tolist()}
    center=np.average(X,axis=0,weights=w);scale=np.sqrt(np.average((X-center)**2,axis=0,weights=w));scale[scale<1e-12]=1.
    Z=(X-center)/scale;zmean=np.average(Z,axis=0,weights=w);A=Z-zmean
    b=np.linalg.solve(A.T@(w[:,None]*A)+20*np.eye(X.shape[1]),A.T@(w[:,None]*(Y-mean)))
    a=mean-zmean@b
    return (x-center)/scale@b+a,{'center':center,'scale':scale,'coef':b.T,'intercept':a}


def check_estimate(panel,history,ids,at,raw,kind,info):
    assert list(pd.to_datetime(info['members'],utc=True))==ids and info['variant']==kind
    ready=raw is not None and len(ids)>=60 and panel.loc[at,'feature_ready'];assert info['ready']==ready
    if not ready:assert info['point'] is None and info['lower'] is None;return 0
    base=np.array([history[t] for t in ids]);truth=panel.loc[ids,['C','D']].to_numpy(float)
    residual=np.column_stack((truth[:,0]+truth[:,1],truth[:,0]-truth[:,1]))-base
    xparts=[];qparts=[]
    if kind!='intercept':xparts.append(base);qparts.append(raw)
    if kind in ['context','context_iid']:
        xparts.append(panel.loc[ids,STATE_COLUMNS].to_numpy(float));qparts.append(panel.loc[at,STATE_COLUMNS].to_numpy(float))
    X=np.column_stack(xparts) if xparts else np.empty((len(ids),0));x=np.concatenate(qparts) if qparts else np.empty(0)
    n=len(ids);w=np.array([2.**((j-n+1)/63) for j in range(n)]);w*=n/w.sum();close(info['weights'],w)
    value,model=solve(X,residual,w,x);point=raw+value;close(info['point'],point)
    for k in model:close(info['point_model'][k],model[k])
    length=1 if kind=='context_iid' else 10;seed=826+at.tz_convert(core.ET).date().toordinal()
    assert info['seed']==seed and info['length']==length
    rng=np.random.default_rng(seed);starts=rng.integers(0,n,size=(199,(n+length-1)//length))
    assert starts.tolist()==info['block_starts'];pred=[]
    for draws in starts:
        indices=[(int(s)+j)%n for s in draws for j in range(length)][:n]
        # Explicit repeated observations, instead of training's multiplicity weights.
        bw=w[indices];v,_=solve(X[indices],residual[indices],bw,x);pred.append(raw+v)
    close(info['bootstrap_predictions'],pred)
    deviations=sorted(max(pair-point) for pair in pred);radius=max(0.,deviations[179])
    near(info['radius'],radius);close(info['lower'],point-radius)
    return 400


def verify(run):
    run=Path(run);reg=read(run/'registration.json');m=read(run/'input_manifest.json');p=reg['preregistration']['protocol']
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    source=Path(m['source_run']);label_source=Path(m['label_source'])
    for f,h in m['source_files_sha256'].items():assert sha(source/f)==h
    for f,h in m['label_files_sha256'].items():assert sha(label_source/f)==h
    assert reg['protocol_sha256']==sha(Path(__file__).parent/'protocol.json')
    market=core.load_market();assert market.manifest==m['full_market'];market=restrict_market(market,pd.Timestamp(m['observation_cutoff_exclusive']))
    panel=pd.read_parquet(run/'panel.parquet');features=pd.read_parquet(run/'features.parquet')
    pd.testing.assert_frame_equal(panel,pd.read_parquet(source/'panel.parquet'));pd.testing.assert_frame_equal(features,pd.read_parquet(source/'features.parquet'))
    base=labels(label_source/'baseline_labels.json');stress=labels(label_source/'stress_labels.json');terminal={}
    for at,row in panel[panel.feature_ready].iterrows():
        rel=[terminal_audit(market,base[key(at,s)]) for s in core.ETFS]
        if all(t is not None for t in rel):terminal[at]=max(rel);assert row.terminal_release==terminal[at]
        else:assert pd.isna(row.terminal_release)
    predictions=read(run/'predictions.json');old=read(source/'predictions.json');history={};heads=0;ticks=0
    assert predictions['source_early']==old['early_base_cal'] and predictions['source_no_risk']==old['early_no_risk']
    assert predictions['double_cost']==predictions[p['primary']]
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel.iterrows()):
            filename=str(row.date.date())+'.json';assert sha(source/'training'/filename)==m['source_training_sha256'][filename]
            original=read(source/'training'/filename);rec=read(run/'training'/filename)
            cutoff=row.date.tz_localize(core.ET).tz_convert('UTC');assert pd.Timestamp(rec['cutoff'])==cutoff
            prior=list(panel[panel.feature_ready&(panel.index<cutoff)&(panel.date<row.date)].index);prefix=[]
            for t in prior:
                if t not in terminal or terminal[t]>=cutoff:break
                prefix.append(t)
            pending=prior[len(prefix)] if len(prefix)<len(prior) else None
            expired=bool(pending is not None and panel.loc[pending,'date']+pd.Timedelta(days=14)<=row.date)
            assert rec['feedback']==original['feedback'] and rec['feedback']['expired_unresolved_prefix']==expired
            ids=[t for t in history if t in set(prefix) and history[t] is not None][-120:]
            source_raw=original['forecasts']['early_base_cal']['event']['raw']
            raw=np.array([source_raw[s] for s in core.ETFS]) if source_raw else None
            if raw is None:assert rec['raw'] is None
            else:close(rec['raw'],raw)
            for kind in ['context','score','intercept','context_iid']:
                heads+=check_estimate(panel,history,ids,at,raw,kind,rec['estimates'][kind])
            for name,kind,band in [('context_band','context',True),('context_point','context',False),
                ('score_band','score',True),('score_point','score',False),('intercept_band','intercept',True),('context_iid_band','context_iid',True)]:
                event=predictions[name][i];assert rec['events'][name]==event and pd.Timestamp(event['at'])==at
                info=rec['estimates'][kind];ready=info['ready'] and not expired;assert event['ready']==ready
                if ready:
                    values=info['lower' if band else 'point'];j=0 if values[0]>=values[1] else 1
                    action=core.ETFS[j] if values[j]>1e-12 else 'CASH';assert event['values']==values
                    near(event['buy_advantage'],values[j]);assert event['top_symbol']==core.ETFS[j]
                else:action='CASH';assert event['values'] is None
                assert event['action']==action and event['allocation']==float(action!='CASH');ticks+=1
            assert predictions['cash'][i]['action']=='CASH';history[at]=raw
            if (i+1)%50==0:print('verified through',row.date.date(),flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');settings=p[reg['phase']]
    assert frozen['predictions_sha256']==sha(run/'predictions.json') and not frozen['new_v26_capital_outcomes_seen']
    months=settings.get('months',settings.get('regression_months'));books=[]
    required=market.sessions[(market.sessions>=settings['evaluation_start'])&(market.sessions<=settings['evaluation_end'])]
    observed=list(features[features.date.between(settings['evaluation_start'],settings['evaluation_end'])].date)
    assert result['coverage']=={'complete':list(required)==observed,'expected_dates':len(required),'observed_dates':len(observed)}
    for name,stats in result['routes'].items():
        wanted=[e for e in predictions[name] if settings['evaluation_start']<=e['date'][:10]<=settings['evaluation_end']]
        assert frozen['routes'][name]==wanted
        books.append(book_check(run,name,market,features,stress if name=='double_cost' else base,wanted,
            2 if name=='double_cost' else 1,stats,months,require_full_cash=False))
    for n,old_n in [('source_early','early_base_cal'),('source_no_risk','early_no_risk')]:
        for suffix in ['equity','trades']:pd.testing.assert_frame_equal(pd.read_csv(run/(n+'_'+suffix+'.csv')),pd.read_csv(source/(old_n+'_'+suffix+'.csv')),atol=1e-12,rtol=1e-12)
    prefix_check=None
    if reg['phase']=='cases':
        dev=Path(reg['authorization']['run']);past=read(dev/'predictions.json')
        for name,events in past.items():assert predictions[name][:len(events)]==events
        for path in (dev/'training').glob('*.json'):assert read(path)==read(run/'training'/path.name)
        prefix_check={'preMay_records_exact':len(list((dev/'training').glob('*.json'))),'all_route_forecasts_exact':True}
    receipt={'passed':True,'phase':reg['phase'],'daily_membership_checks':len(panel),'joint_terminal_receipts':len(terminal),
        'independent_ridge_head_equations':heads,'calibration_and_action_checks':ticks,'capital_books':books,'two_source_books_exact':True,
        'case_prefix':prefix_check,'results_sha256':sha(run/'results.json'),'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
