"""Independent full nonlinear model, calibration, source and fifteen-book audit."""
import argparse
import gzip
import json
from threadpoolctl import threadpool_limits
from block_learning import *
from audit_blocks import reference,compare,query
from verify_transfer import reference_parents,reference_estimate,check_events,canonical
from audit_dense_calibration import daily_check
from audit_capital import book_check
from audit_statistics import audit_statistics


def verify(run):
    run=Path(run);registration=read(run/'registration.json');prereg=registration['preregistration'];p=prereg['protocol']
    assert registration['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in prereg['code_sha256'].items():assert sha(ROOT/f)==h,f
    assert prereg['parent_delivery_sha256']==verify_parent()
    manifest=read(run/'input_manifest.json')
    for f,h in manifest['derived_files_sha256'].items():assert sha(run/f)==h==sha(PARENT/f)
    for f in ['input_manifest.json','input_audit.json']:assert sha(run/f)==sha(PARENT/f)
    market=load_market();assert market.manifest==manifest['market_manifest']
    panels={k:pd.read_parquet(run/(k+'_panel.parquet')) for k in PAIRS};panel=panels['semiconductor']
    predictions=read(run/'predictions.json');histories={k:{} for k in VARIANTS};previous=None;models_checked=calchecks=days=actions=0
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel.iterrows()):
            record=read(run/'training'/(tag(at)+'.json'));oldpath=PARENT/'training'/(tag(at)+'.json');old=read(oldpath)
            assert record['parent_source_sha256']==sha(oldpath)
            assert record['uniform_raw']==old['raws']['partial'] and record['uniform_estimate']==old['estimates']['partial']
            assert record['old_quadratic_raw']==old['old_quadratic_raw'] and record['old_quadratic_estimate']==old['old_quadratic_estimate']
            assert record['linear_raw']==old['linear_raw'] and record['linear_estimate']==old['linear_estimate']
            assert pd.Timestamp(record['at'])==at and pd.Timestamp(record['cutoff'])==day_cutoff(at)
            if row.date!=previous:
                filename=str(row.date.date());saved=read(run/'daily'/(filename+'.json'));parents,feedback=reference_parents(panels,market,at)
                assert saved['parents']==canonical(parents) and saved['feedback']==canonical(feedback)
                with gzip.open(run/'models'/(filename+'.json.gz'),'rt') as f:stored=json.load(f)
                refs={}
                for clock in CLOCKS:
                    ids=[t for d in parents for t in d['members'] if panel.loc[t,'clock']==clock] if len(parents)>=120 else []
                    assert saved['members'][clock]==canonical(ids)
                    if len(ids)>=120:
                        # Rebuild values and date weights without calling training_arrays.
                        n=len(ids);dw=np.array([2.**((j-n+1)/126) for j in range(n)]);dw*=n/dw.sum()
                        X=np.concatenate([panels[k].loc[ids,FIELDS].to_numpy(float) for k in PAIRS])
                        Y=np.concatenate([panels[k].loc[ids,['C','D']].to_numpy(float) for k in PAIRS])
                        names=np.repeat(list(PAIRS),n);weights=np.tile(dw/len(PAIRS),len(PAIRS))
                        refs[clock]=reference(X,Y,names,weights,n);compare(stored[clock],refs[clock])
                        np.testing.assert_array_equal(stored[clock]['common'],np.r_[np.full(18,100.),np.full(171,950.)])
                        np.testing.assert_array_equal(stored[clock]['deviation'],np.r_[np.full(18,300.),np.full(171,2850.)])
                        assert stored[clock]['normalization']=={'block_widths':[18,171],'reference_width':18,'factors':[1.]*18+[9.5]*171};models_checked+=3
                        # No sharing must collapse to a single-pair normal-equation problem.
                        ref=refs[clock];j=ref['names'].index('semiconductor');xx=panels['semiconductor'].loc[ids,FIELDS].to_numpy(float)
                        z=(xx-ref['original_center'][j])/ref['original_scale'][j]
                        poly=np.column_stack([z]+[z[:,a]*z[:,b] for a in range(len(FIELDS)) for b in range(a,len(FIELDS))])
                        z=(poly-ref['centers'][j])/ref['scales'][j];yy=panels['semiconductor'].loc[ids,['C','D']].to_numpy(float)
                        A=np.column_stack([np.ones(n),z]);P=np.diag([0.]+[100.]*18+[950.]*171)
                        solo=np.linalg.solve(A.T@(dw[:,None]*A)+P,A.T@(dw[:,None]*yy))
                        np.testing.assert_allclose(solo[0],ref['intercepts'][j],atol=1e-10,rtol=1e-9)
                        np.testing.assert_allclose(solo[1:],ref['coefficients']['independent'][j],atol=1e-10,rtol=1e-8)
                    else:refs[clock]=None;assert stored[clock] is None
                daily=saved['calibration'];calrefs={}
                for k in VARIANTS:calrefs[k],count=daily_check(panel,market,histories[k],at,daily[k]);calchecks+=count
                previous=row.date;days+=1
            estimates={}
            for k in VARIANTS:
                raw=None
                if refs[row.clock] is not None and row.feature_ready:
                    cd=query(refs[row.clock],row[FIELDS].to_numpy(float),'semiconductor',k);raw=np.array([cd[0]+cd[1],cd[0]-cd[1]])
                if raw is None:assert record['raws'][k] is None
                else:np.testing.assert_allclose(record['raws'][k],raw,atol=1e-9,rtol=1e-8)
                estimates[k]=reference_estimate(panel,at,raw,daily[k],calrefs[k]);got=record['estimates'][k]
                assert got['ready']==estimates[k]['ready']
                if got['ready']:
                    for f in ['point','lower','radius','bootstrap_predictions']:np.testing.assert_allclose(got[f],estimates[k][f],atol=1e-9,rtol=1e-8)
                else:assert got==estimates[k]
                histories[k][at]=raw
            original=check_events(at,row,estimates,daily,feedback)
            expected={'block_'+k+'_'+s:original[k+'_'+s] for k in VARIANTS for s in ['band','point']}
            for k in ['premarket_only','cash','double_cost']:expected[k]=original[k]
            for name,source in CONTROL_MAP.items():expected[name]=old['events'][source]
            for k,e in expected.items():
                got=record['events'][k];assert got==predictions[k][i]
                for f in ['at','date','clock','ready','action','allocation']:assert got[f]==canonical(e[f])
                for f in ['values','buy_advantage']:
                    if e.get(f) is not None:np.testing.assert_allclose(got[f],e[f],atol=1e-9,rtol=1e-8)
                actions+=1
            if (i+1)%400==0:print('Verified blocks',i+1,'/',len(panel),at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json') and not frozen['new_capital_outcomes_seen']
    settings=p['development'];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings['months']
    features=pd.read_parquet(run/'features.parquet');test=features[features.date.between(start,end)]
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(test.index)==expected and result['coverage']['complete']
    base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json');books=[]
    for k in ROUTES:
        es=[e for e in predictions[k] if start<=str(pd.Timestamp(e['date']).date())<=end];assert frozen['routes'][k]==es
        books.append(book_check(run,k,market,test,stress if k=='double_cost' else base,es,2 if k=='double_cost' else 1,result['routes'][k],months,require_full_cash=False))
    stats=audit_statistics(run);oldresult=read(PARENT/'results.json')
    for name,parent_name in CONTROL_MAP.items():
        assert result['routes'][name]==oldresult['routes'][parent_name]
        for suffix in ['equity.csv','trades.csv','decisions.csv']:assert sha(run/(name+'_'+suffix))==sha(PARENT/(parent_name+'_'+suffix))
    gate=development_gate(result['primary'],True,p)
    assert result['primary_name']==p['primary'] and result['primary']==result['routes'][p['primary']] and result['development_gate']==gate
    assert not result['goal_complete'] and not result['case_regression']['executed']
    receipt={'passed':True,'results_sha256':sha(run/'results.json'),'prediction_rows':len(panel),'dates':days,
             'base_models_normal_equations':models_checked,'calibration_equation_checks':calchecks,'route_actions':actions,
             'capital_books':books,'statistics_checks':stats,'six_old_controls_byte_identical':True,'independent_equals_solo':True,
             'source_sha256':sha(Path(__file__)),'development_gate':gate,'strategy_independent_validation':False,'goal_complete':False}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();verify(a.run)
