"""Independent qualification membership, support bounds, fits and capital books."""
import argparse
from threadpoolctl import threadpool_limits
from supported import *
from verify_eligible import daily_audit,check_event,COLS
from verify_statistics import audit_statistics
from verify_selected import verify_panel,fit_audit,reference_quality,canonical,SCORE_COLUMNS,STATE_COLUMNS
from audit_shared import reference_pool
from verify_reliability import close
from verify_continuation import near
from audit_books import book_check

def verify(run):
    run=Path(run);reg=read(run/'registration.json');p=reg['preregistration']['protocol'];manifest=read(run/'input_manifest.json')
    assert reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    if reg['phase']=='development':
        root,records,want=load_source()
        assert root==Path(manifest['verified_source']) and manifest['source_files_sha256']==want
    else:
        authorization=authorize(reg['authorization']['run']);assert authorization==reg['authorization']
        root,records,want=case_source(manifest['verified_source'],authorization)
        assert manifest['source_files_sha256']==want
    prior,delivery=history_source()
    assert manifest['selected_history_delivery_sha256']==sha(core.ROOT/'research/soxl_soxs_14d5pct_v38/evidence/delivery_manifest.json')
    market=core.load_market();assert market.manifest==manifest['market']['full_market'];market=restrict_market(market,pd.Timestamp(manifest['market']['observation_cutoff_exclusive']))
    original=pd.read_parquet(root/'financial_panel.parquet');panel=pd.read_parquet(run/'selected_panel.parquet');features=pd.read_parquet(root/'features.parquet')
    records={pd.Timestamp(r['at']):r for f in sorted((root/'training').glob('*.json')) if (r:=read(f))}
    feature_checks=verify_panel(original,panel,records);predictions=read(run/'predictions.json');previous=None;daily=None;checks=actions=readydays=0
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            if row.date!=previous:
                daily=read(run/'daily'/(str(row.date.date())+'.json'));checks+=daily_audit(panel,market,records,at,daily);readydays+=int(daily['ready']);previous=row.date
            saved=read(run/'training'/(tag(at)+'.json'));record=records[at]
            assert saved['source_record_sha256']==sha(root/'training'/(tag(at)+'.json')) and pd.Timestamp(saved['at'])==at and saved['clock']==row.clock
            assert pd.Timestamp(saved['date'])==row.date and pd.Timestamp(saved['cutoff'])==pd.Timestamp(at.tz_convert(core.ET).date(),tz=core.ET).tz_convert('UTC')
            ready=bool(record['estimates']['full']['ready'] and panel.loc[at,'feature_ready'])
            for kind,v in saved['estimates'].items():
                assert v['ready']==ready and v['source_fallback']==(ready and not daily['ready'])
                if not ready or not daily['ready']:assert v['quality'] is None and v['lower_quality'] is None;continue
                x=panel.loc[at,COLS[kind]].to_numpy(float);info=daily['inputs'][kind]
                outside=[c for j,c in enumerate(COLS[kind]) if x[j]<info['minimum'][j]-1e-12 or x[j]>info['maximum'][j]+1e-12]
                assert v['outside_columns']==outside;point=reference_quality(daily['models'][kind],x);near(v['quality'],point)
                draws=[reference_quality(b[kind],x) if b[kind] is not None else None for b in daily['bootstrap']]
                assert v['bootstrap_complete']==all(z is not None for z in draws)
                for actual,wanted in zip(v['bootstrap_qualities'],draws):
                    if wanted is None:assert actual is None
                    else:near(actual,wanted)
                if v['bootstrap_complete']:
                    radius=max(0.,sorted(np.array(draws)-point)[179]);near(v['upward_radius'],radius);near(v['lower_quality'],point-radius)
                else:assert v['upward_radius'] is None and v['lower_quality'] is None
            for name,es in predictions.items():
                e=es[i];assert e==saved['events'][name] and pd.Timestamp(e['at'])==at and pd.Timestamp(e['date'])==row.date
                check_event('all_supported' if name=='double_cost' else name,e,record,daily,saved['estimates'],None);actions+=1
            if (i+1)%200==0:print('verified eligibility through',at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    settings=p[reg['phase']];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings.get('months',settings.get('regression_months'))
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(features[features.date.between(start,end)].index)==expected
    assert result['coverage']=={'complete':True,'expected_ticks':len(expected),'observed_ticks':len(expected),'dates':len(set(panel.loc[expected,'date']))};books=[]
    for name,stats in result['routes'].items():
        es=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert es==frozen['routes'][name]
        books.append(book_check(run,name,market,features,labels(root/('stress_labels.json' if name=='double_cost' else 'baseline_labels.json')),es,2 if name=='double_cost' else 1,stats,months))
    for suffix in ['equity','trades','decisions','orders']:
        pd.testing.assert_frame_equal(pd.read_csv(run/('source_v35_'+suffix+'.csv')),pd.read_csv(root/('full_band_'+suffix+'.csv')),atol=1e-12,rtol=1e-12)
        if reg['phase']=='development':
            pd.testing.assert_frame_equal(pd.read_csv(run/('all_supported_'+suffix+'.csv')),pd.read_csv(prior/('all_supported_'+suffix+'.csv')),atol=1e-12,rtol=1e-12)
    old=read(prior/'predictions.json')
    for name,es in predictions.items():
        before=old['all_supported' if name=='double_cost' else name];assert es[:len(before)]==before
    for f in (prior/'daily').glob('*.json'):assert read(f)==read(run/'daily'/f.name)
    for f in (prior/'training').glob('*.json'):
        before=read(f);after=read(run/'training'/f.name)
        for key in ['at','date','clock','cutoff','estimates','source_record_sha256']:assert before[key]==after[key]
    statistics=audit_statistics(run)
    gate=development_gate(result['primary'],result['coverage']['complete'],p) if reg['phase']=='development' else None
    assert result['primary_name']=='all_supported' and result['primary']==result['routes']['all_supported']
    if gate:assert gate==result['development_gate']
    else:assert result['case_acceptance']['passed']==bool(result['coverage']['complete'] and result['primary']['primary_pass'])
    receipt={'passed':True,'phase':reg['phase'],'clock_ticks':len(features),'daily_records':features.date.nunique(),'ready_fit_dates':readydays,
        'selected_feature_equations':feature_checks,'independent_logistic_equations':checks,'action_checks':actions,'capital_books':books,
        'source_v35_control_exact':True,'preMay_forecasts_exact':True,'registered_primary_verified':True,'primary_development_passed':gate['passed'] if gate else None,'independent_statistic_assertions':statistics,'results_sha256':sha(run/'results.json'),'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
