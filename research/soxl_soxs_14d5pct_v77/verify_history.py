"""Independent daily normal equations, calibration, decisions and full capital ledgers."""
import argparse
import gzip
import json
from threadpoolctl import threadpool_limits
from history_learning import *
from reference_fit import from_panels
from verify_transfer import reference_parents,reference_estimate,canonical
from audit_dense_calibration import daily_check
from audit_capital import book_check
from audit_statistics import audit_statistics


def verify(run):
    run=Path(run);registration=read(run/'registration.json');pre=registration['preregistration'];p=pre['protocol']
    assert registration['protocol_sha256']==sha(SOURCE/'protocol.json')==pre['protocol_sha256']
    for f,h in pre['code_sha256'].items():assert sha(ROOT/f)==h,f
    assert pre['parent_delivery_sha256']==verify_parents()
    manifest=read(run/'input_manifest.json')
    for f,h in manifest['derived_files_sha256'].items():assert sha(run/f)==h,f
    for name in ['input_manifest','input_audit']:assert pre[name+'_sha256']==sha(run/(name+'.json'))
    assert read(run/'input_audit.json')['passed']
    market=extended_market();ledger=load_market()
    assert market.manifest==manifest['expanded_market_manifest'] and ledger.manifest==manifest['ledger_market_manifest']
    panels={k:pd.read_parquet(run/(k+'_panel.parquet')) for k in PAIRS}
    shorts={k:pd.read_parquet(run/(k+'_short_panel.parquet')) for k in PAIRS};panel=panels['semiconductor']
    predictions=read(run/'predictions.json');histories={k:{} for k in VARIANTS};previous=None
    fits=calchecks=days=actions=0
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel.iterrows()):
            record=read(run/'training'/(tag(at)+'.json'));oldpath=PARENT/'training'/(tag(at)+'.json')
            parent=read(oldpath) if oldpath.exists() else None
            assert record['parent_source_sha256']==(sha(oldpath) if parent else None)
            assert record['short_full_raw']==(parent['linear_raw'] if parent else None)
            assert record['short_full_estimate']==(parent['linear_estimate'] if parent else None)
            assert record['short_mean_raw']==(parent['raws']['mean'] if parent else None)
            assert record['short_mean_estimate']==(parent['estimates']['mean'] if parent else None)
            assert pd.Timestamp(record['at'])==at and pd.Timestamp(record['cutoff'])==day_cutoff(at)
            if previous!=row.date:
                filename=str(row.date.date());saved=read(run/'daily'/(filename+'.json'))
                parents,lf=reference_parents(panels,market,at);sp,sf=reference_parents(shorts,market,at);feedback={'long':lf,'short':sf}
                assert saved['parents']==canonical({'long':parents,'short':sp}) and saved['feedback']==canonical(feedback)
                with gzip.open(run/'models'/(filename+'.json.gz'),'rt') as f:stored=json.load(f)
                refs={}
                for c in CLOCKS:
                    longids=[t for r in parents for t in r['members'] if panel.loc[t,'clock']==c] if len(parents)>=120 else []
                    shortids=[t for r in sp for t in r['members'] if shorts['semiconductor'].loc[t,'clock']==c] if len(sp)>=120 else []
                    reference=max(120,len(shortids));refs[c]={}
                    for k in VARIANTS:
                        ids=shortids if k=='short_price' else longids
                        assert saved['members'][c][k]==canonical(ids)
                        if len(ids)<120:assert k not in stored[c];continue
                        ref=from_panels(shorts if k=='short_price' else panels,ids,reference,k,FIELDS);refs[c][k]=ref;model=stored[c][k]
                        assert model['distinct_dates']==len(ids) and model['variant']==k
                        fields=['weights','mean_cd'] if k=='long_mean' else ['centers','scales','intercepts','coefficients','penalty','weights']
                        for f in fields:np.testing.assert_allclose(model[f],ref[f],atol=1e-10,rtol=1e-9)
                        w=np.array(ref['weights']);ess=w.sum()**2/(w@w)/(1 if k=='long_mean' else 3)
                        np.testing.assert_allclose(model['date_effective_sample_size'],ess,atol=1e-10,rtol=1e-10)
                        if k!='long_mean':
                            assert model['names']==ref['names'] and model['reference_mass']==reference
                            assert model['penalty_scale']==(1 if k=='long_native' else len(ids)/reference)
                            np.testing.assert_allclose(model['pair_mass'],[1/3]*3,atol=1e-12,rtol=1e-12)
                        fits+=1
                    assert set(stored[c])==set(refs[c])
                daily=saved['calibration'];calrefs={}
                for k in VARIANTS:calrefs[k],count=daily_check(panel,market,histories[k],at,daily[k]);calchecks+=count
                previous=row.date;days+=1
            estimates={}
            for k in VARIANTS:
                raw=None
                if k in refs[row.clock] and row.feature_ready:
                    model=refs[row.clock][k]
                    if k=='long_mean':cd=model['mean_cd']
                    else:
                        x=row[FIELDS].to_numpy(float).copy()
                        if k in ['short_price','long_price','long_native']:x[11:16]=0
                        j=model['names'].index('semiconductor');z=(x-model['centers'][j])/model['scales'][j]
                        cd=model['intercepts'][j]+z@model['coefficients'][j]
                    raw=np.array([cd[0]+cd[1],cd[0]-cd[1]])
                if raw is None:assert record['raws'][k] is None
                else:np.testing.assert_allclose(record['raws'][k],raw,atol=1e-10,rtol=1e-9)
                estimates[k]=reference_estimate(panel,at,raw,daily[k],calrefs[k]);got=record['estimates'][k]
                assert got['ready']==estimates[k]['ready']
                if got['ready']:
                    for f in ['point','lower','radius','bootstrap_predictions']:np.testing.assert_allclose(got[f],estimates[k][f],atol=1e-9,rtol=1e-8)
                else:assert got==estimates[k]
                histories[k][at]=raw
            cash={'at':at,'date':row.date,'clock':row.clock,'ready':True,'action':'CASH','allocation':0.,'values':None};expected={}
            for k in VARIANTS:
                for suffix in ['band','point']:
                    info=estimates[k];e={'ready':False,'action':'CASH','allocation':0.,'values':None}
                    fb=feedback['short' if k=='short_price' else 'long']
                    if info['ready'] and not daily[k]['feedback']['expired_unresolved_prefix'] and not fb['expired_unresolved_prefix']:
                        values=np.array(info['lower' if suffix=='band' else 'point']);j=int(values[1]>values[0]);buy=values[j]>1e-12
                        e.update(ready=True,action=['SOXL','SOXS'][j] if buy else 'CASH',allocation=float(buy),values=values.tolist(),
                            top_symbol=['SOXL','SOXS'][j],buy_advantage=float(values[j]))
                    expected[k+'_'+suffix]={'at':at,'date':row.date,'clock':row.clock,**e}
            expected['cash']=cash
            expected['premarket_only']=dict(expected['long_price_band']) if row.clock==CLOCKS[0] else dict(cash)
            expected['double_cost']=dict(expected['long_price_band'])
            for name,source in CONTROL_MAP.items():expected[name]=parent['events'][source] if parent else dict(cash)
            assert set(record['events'])==set(ROUTES)
            for k,e in expected.items():
                got=record['events'][k];assert got==predictions[k][i]
                for f in ['at','date','clock','ready','action','allocation']:assert got[f]==canonical(e[f])
                assert got.get('top_symbol')==e.get('top_symbol')
                for f in ['values','buy_advantage']:
                    if e.get(f) is not None:np.testing.assert_allclose(got[f],e[f],atol=1e-9,rtol=1e-8)
                    else:assert got.get(f) is None
                actions+=1
            if (i+1)%400==0:print('Verified history',i+1,'/',len(panel),at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json')
    assert frozen['predictions_sha256']==sha(run/'predictions.json') and not frozen['new_capital_outcomes_seen']
    settings=p['development'];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings['months']
    features=pd.read_parquet(run/'features.parquet');test=features[features.date.between(start,end)]
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in ledger.sessions[(ledger.sessions>=start)&(ledger.sessions<=end)]
        for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(ledger.calendar[d]['close_at'])]
    assert list(test.index)==expected and result['coverage']['complete']
    base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json');books=[]
    for k in ROUTES:
        es=[e for e in predictions[k] if start<=str(pd.Timestamp(e['date']).date())<=end];assert frozen['routes'][k]==es
        books.append(book_check(run,k,ledger,test,stress if k=='double_cost' else base,es,2 if k=='double_cost' else 1,result['routes'][k],months,require_full_cash=False))
    stats=audit_statistics(run);oldresult=read(PARENT/'results.json')
    for name,parent_name in CONTROL_MAP.items():
        assert result['routes'][name]==oldresult['routes'][parent_name]
        for suffix in ['equity.csv','trades.csv','decisions.csv']:assert sha(run/(name+'_'+suffix))==sha(PARENT/(parent_name+'_'+suffix))
    gate=development_gate(result['primary'],True,p)
    assert result['primary_name']==p['primary'] and result['primary']==result['routes'][p['primary']] and result['development_gate']==gate
    assert not result['goal_complete'] and not result['case_regression']['executed']
    receipt={'passed':True,'results_sha256':sha(run/'results.json'),'prediction_rows':len(panel),'dates':days,
        'base_models_normal_equations':fits,'calibration_equation_checks':calchecks,'route_actions':actions,
        'capital_books':books,'statistics_checks':stats,'four_old_controls_byte_identical':True,
        'source_sha256':sha(Path(__file__)),'development_gate':gate,'strategy_independent_validation':False,'goal_complete':False}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);verify(p.parse_args().run)
