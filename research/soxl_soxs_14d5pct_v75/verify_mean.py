"""Independent causal mean, calibration, source and fourteen-book audit."""
import argparse
import gzip
import json
from threadpoolctl import threadpool_limits
from mean_learning import *
from verify_transfer import reference_parents,reference_estimate,canonical
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
            assert record['block_raw']==old['raws']['partial'] and record['block_estimate']==old['estimates']['partial']
            assert record['uniform_raw']==old['uniform_raw'] and record['uniform_estimate']==old['uniform_estimate']
            assert record['old_quadratic_raw']==old['old_quadratic_raw'] and record['old_quadratic_estimate']==old['old_quadratic_estimate']
            assert record['linear_raw']==old['linear_raw'] and record['linear_estimate']==old['linear_estimate']
            assert pd.Timestamp(record['at'])==at and pd.Timestamp(record['cutoff'])==day_cutoff(at)
            if row.date!=previous:
                filename=str(row.date.date());saved=read(run/'daily'/(filename+'.json'));parents,feedback=reference_parents(panels,market,at)
                assert saved['parents']==canonical(parents) and saved['feedback']==canonical(feedback)
                with gzip.open(run/'models'/(filename+'.json.gz'),'rt') as f:stored=json.load(f)
                with gzip.open(PARENT/'models'/(filename+'.json.gz'),'rt') as f:parent_models=json.load(f)
                refs={}
                for clock in CLOCKS:
                    ids=[t for d in parents for t in d['members'] if panel.loc[t,'clock']==clock] if len(parents)>=120 else []
                    assert saved['members'][clock]==canonical(ids)
                    if len(ids)>=120:
                        n=len(ids);w=np.array([2.**((j-n+1)/126) for j in range(n)]);w*=n/sum(w)
                        values=panel.loc[ids,['C','D']].to_numpy(float)
                        mean=np.array([sum(w[j]*values[j,k] for j in range(n))/sum(w) for k in range(2)])
                        assert stored[clock]['real_dates']==n
                        np.testing.assert_allclose(stored[clock]['weights'],w,atol=1e-12,rtol=1e-12)
                        np.testing.assert_allclose(stored[clock]['mean_cd'],mean,atol=1e-12,rtol=1e-12)
                        oldmodel=parent_models[clock];j=oldmodel['names'].index('semiconductor')
                        np.testing.assert_allclose(mean,oldmodel['intercepts'][j],atol=1e-12,rtol=1e-12)
                        refs[clock]=mean;models_checked+=1
                    else:refs[clock]=None;assert stored[clock] is None
                daily=saved['calibration'];calrefs={}
                for k in VARIANTS:calrefs[k],count=daily_check(panel,market,histories[k],at,daily[k]);calchecks+=count
                previous=row.date;days+=1
            estimates={}
            for k in VARIANTS:
                raw=None
                if refs[row.clock] is not None and row.feature_ready:
                    cd=refs[row.clock];raw=np.array([cd[0]+cd[1],cd[0]-cd[1]])
                if raw is None:assert record['raws'][k] is None
                else:np.testing.assert_allclose(record['raws'][k],raw,atol=1e-9,rtol=1e-8)
                estimates[k]=reference_estimate(panel,at,raw,daily[k],calrefs[k]);got=record['estimates'][k]
                assert got['ready']==estimates[k]['ready']
                if got['ready']:
                    for f in ['point','lower','radius','bootstrap_predictions']:np.testing.assert_allclose(got[f],estimates[k][f],atol=1e-9,rtol=1e-8)
                else:assert got==estimates[k]
                histories[k][at]=raw
            expected={}
            for suffix in ['band','point','raw']:
                info=estimates['mean'];e={'ready':False,'action':'CASH','allocation':0.,'values':None}
                if info['ready'] and not daily['mean']['feedback']['expired_unresolved_prefix'] and not feedback['expired_unresolved_prefix']:
                    values=histories['mean'][at] if suffix=='raw' else info['lower' if suffix=='band' else 'point']
                    values=np.array(values);j=0 if values[0]>=values[1] else 1;buy=values[j]>1e-12
                    e.update(ready=True,action=['SOXL','SOXS'][j] if buy else 'CASH',allocation=float(buy),values=values.tolist(),top_symbol=['SOXL','SOXS'][j],buy_advantage=float(values[j]))
                expected['mean_'+suffix]={'at':at,'date':row.date,'clock':row.clock,**e}
            expected['cash']={'at':at,'date':row.date,'clock':row.clock,'ready':True,'action':'CASH','allocation':0.,'values':None}
            expected['premarket_only']=dict(expected['mean_band']) if row.clock==CLOCKS[0] else dict(expected['cash'])
            expected['double_cost']=dict(expected['mean_band'])
            for name,source in CONTROL_MAP.items():expected[name]=old['events'][source]
            for k,e in expected.items():
                got=record['events'][k];assert got==predictions[k][i]
                for f in ['at','date','clock','ready','action','allocation']:assert got[f]==canonical(e[f])
                assert got.get('top_symbol')==e.get('top_symbol')
                for f in ['values','buy_advantage']:
                    if e.get(f) is not None:np.testing.assert_allclose(got[f],e[f],atol=1e-9,rtol=1e-8)
                    else:assert got.get(f) is None
                actions+=1
            if (i+1)%400==0:print('Verified means',i+1,'/',len(panel),at,flush=True)
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
             'causal_means_reconstructed':models_checked,'calibration_equation_checks':calchecks,'route_actions':actions,
             'capital_books':books,'statistics_checks':stats,'eight_old_controls_byte_identical':True,'own_mean_equals_parent_intercept':True,
             'source_sha256':sha(Path(__file__)),'development_gate':gate,'strategy_independent_validation':False,'goal_complete':False}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();verify(a.run)
