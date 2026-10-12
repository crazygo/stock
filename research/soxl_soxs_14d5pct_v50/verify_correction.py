"""Independent direction-offset, joint bootstrap and full capital audit."""
import argparse
from threadpoolctl import threadpool_limits
from confidence_shrinkage import *
from audit_joint import daily_check,input_rows,predict as reference_predict
from audit_shrinkage import reference_q,input_rows as offset_rows
from verify_reliability import close
from verify_financial import check_action
from verify_continuation import near
from audit_books import book_check
from audit_statistics import audit_statistics
from audit_option_features import audit_option_features


def corrected(source,model,x):
    q=reference_q(model,x,source['q']);c=float(source['C']);m=float(source['M']);d=m*(2*q-1)
    return dict(C=c,M=m,D=d,q=q,SOXL=c+d,SOXS=c-d)


def check_estimate(info,point,draws):
    radius=max(0.,sorted(np.max(draws-point,axis=1))[179])
    close(info['point'],point);close(info['bootstrap_predictions'],draws)
    near(info['radius'],radius);close(info['lower'],point-radius)


def verify(run):
    run=Path(run);reg=read(run/'registration.json');p=reg['preregistration']['protocol'];manifest=read(run/'input_manifest.json')
    assert reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    root,want=verified_source(reg['phase'],p);assert root==Path(manifest['source']) and want==manifest['source_files_sha256']
    market=core.load_market();assert market.manifest==manifest['market']['full_market'];market=restrict_market(market,pd.Timestamp(manifest['market']['observation_cutoff_exclusive']))
    features=pd.read_parquet(root/'features.parquet');panel=pd.read_parquet(run/'option_panel.parquet');audit_option_features(run)
    records={pd.Timestamp(r['at']):r for f in sorted((root/'training').glob('*.json')) if (r:=read(f))}
    histories={k:{} for k in VARIANTS};predictions=read(run/'predictions.json');prev=None;daily=None
    calchecks=offsetchecks=actions=daychecks=querychecks=0
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            rec=read(run/'training'/(tag(at)+'.json'));source=records[at]
            assert rec['source_record_sha256']==sha(root/'training'/(tag(at)+'.json'))
            assert pd.Timestamp(rec['at'])==at and pd.Timestamp(rec['date'])==row.date and rec['clock']==row.clock and pd.Timestamp(rec['cutoff'])==day_cutoff(at)
            if row.date!=prev:
                daily=read(run/'daily'/(str(row.date.date())+'.json'))
                models,offsets,n,nq=daily_check(panel,market,records,histories,at,daily)
                calchecks+=n;offsetchecks+=nq;daychecks+=1;prev=row.date
                source_day=read(root/'daily'/(str(row.date.date())+'.json'))
                for field in ['parents','feedback','members','ready']:assert daily[field]==source_day[field]
                if daily['ready']:assert daily['variants']['baseline']==source_day['variants']['full']
            for kind in VARIANTS:
                original=source['fits']['full']['raw'];values={};raw=None;qx=None
                if original and panel.loc[at,'feature_ready']:
                    values=original.copy()
                    if kind=='neutral':
                        c,m=float(original['C']),float(original['M']);values=dict(C=c,M=m,D=0.,q=.5,SOXL=c,SOXS=c)
                    elif kind in CORRECTIONS and daily['ready']:
                        qx,_=offset_rows(panel,records,[at],correction_columns(kind))
                        values=corrected(original,offsets[kind]['point_model'],qx[0])
                    raw=np.array([values[s] for s in core.ETFS])
                assert set(rec['fits'][kind])=={'raw'}
                if raw is None:assert rec['raws'][kind] is None and rec['fits'][kind]['raw']=={}
                else:
                    close(rec['raws'][kind],raw)
                    assert set(values)==set(rec['fits'][kind]['raw'])
                    for k,x in values.items():near(rec['fits'][kind]['raw'][k],x)
                    querychecks+=1
                ready=raw is not None and daily['ready'];info=rec['estimates'][kind]
                assert info['ready']==bool(ready)
                if ready:
                    X=input_rows(panel,{at:raw},[at]);m=models[kind]
                    point=raw+reference_predict(m['point_model'],X)[0];samples=[];raw_samples=[]
                    for b,cal in enumerate(m['bootstrap_models']):
                        br=raw
                        if kind in CORRECTIONS:
                            bv=corrected(original,offsets[kind]['bootstrap_models'][b],qx[0]);br=np.array([bv[s] for s in core.ETFS])
                        bx=input_rows(panel,{at:br},[at]);raw_samples.append(br);samples.append(br+reference_predict(cal,bx)[0])
                    check_estimate(info,point,np.array(samples))
                    if kind=='temperature':
                        assert rec['direct_estimate']['ready'];check_estimate(rec['direct_estimate'],raw,np.array(raw_samples))
                else:
                    assert info=={'ready':False,'point':None,'lower':None}
                    if kind=='temperature':assert rec['direct_estimate']=={'ready':False,'point':None,'lower':None}
            assert rec['raws']['baseline']==source['raws']['full'] and rec['estimates']['baseline']==source['estimates']['full']
            for name,es in predictions.items():
                e=es[i];assert e==rec['events'][name] and pd.Timestamp(e['at'])==at and pd.Timestamp(e['date'])==row.date
                if name=='cash' or name=='premarket_only' and row.clock!=CLOCKS[0]:assert e['action']=='CASH' and e['allocation']==0
                else:
                    kind=name[:-5] if name in ['affine_band','options_band','neutral_band','baseline_band'] else 'temperature'
                    est=rec['direct_estimate'] if name=='direction_only_band' else rec['estimates'][kind]
                    check_action(est,daily['feedback']['expired_unresolved_prefix'],name!='temperature_point',e)
                if name=='baseline_band':assert e==source['events']['full_band']
                actions+=1
            for kind in VARIANTS:histories[kind][at]=rec['raws'][kind]
            if (i+1)%200==0:print('correction verified through',at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    settings=p[reg['phase']];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings.get('months',settings.get('regression_months'))
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(features[features.date.between(start,end)].index)==expected
    assert result['coverage']=={'complete':True,'expected_ticks':len(expected),'observed_ticks':len(expected),'dates':len(set(panel.loc[expected,'date']))};books=[]
    for name,stats in result['routes'].items():
        es=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert es==frozen['routes'][name]
        books.append(book_check(run,name,market,features,labels(root/('stress_labels.json' if name=='double_cost' else 'baseline_labels.json')),es,2 if name=='double_cost' else 1,stats,months))
    for suffix in ['equity','trades','orders','decisions']:
        pd.testing.assert_frame_equal(pd.read_csv(run/('baseline_band_'+suffix+'.csv')),pd.read_csv(root/('full_band_'+suffix+'.csv')),atol=1e-12,rtol=1e-12)
    if reg['phase']=='cases':
        dev=Path(reg['authorization']['run']);old_predictions=read(dev/'predictions.json')
        for name,es in predictions.items():assert es[:len(old_predictions[name])]==old_predictions[name]
        for folder in ['training','daily']:
            for f in (dev/folder).glob('*.json'):assert read(f)==read(run/folder/f.name)
    stats=audit_statistics(run);assert result['primary_name']=='temperature_band' and result['primary']==result['routes']['temperature_band']
    if reg['phase']=='development':assert result['development_gate']==development_gate(result['primary'],result['coverage']['complete'],p)
    else:assert result['case_acceptance']['passed']==bool(result['coverage']['complete'] and result['primary']['primary_pass'])
    receipt={'passed':True,'audit_revision':1,'phase':reg['phase'],'clock_ticks':len(features),'daily_fits':daychecks,
        'independent_bounded_KKT_certificates':offsetchecks,'independent_calibration_equations':calchecks,'query_raw_checks':querychecks,
        'action_checks':actions,'capital_books':books,'baseline_raw_calibration_actions_and_ledgers_exact':True,
        'independent_statistic_assertions':stats,'case_prefix_exact':reg['phase']=='cases','results_sha256':sha(run/'results.json'),
        'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
