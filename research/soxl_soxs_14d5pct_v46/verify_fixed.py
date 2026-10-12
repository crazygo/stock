"""Audit fixed candidate identity, daily continuation and six complete capital books."""
import argparse
from threadpoolctl import threadpool_limits
from fixed_candidate import *
from audit_fixed import daily_check,reference_predict
from audit_statistics import audit_statistics
from audit_macro_features import audit_macro_features
from audit_books import book_check
from verify_reliability import close
from verify_continuation import near
from verify_financial import check_action


def verify(run):
    run=Path(run);reg=read(run/'registration.json');p=reg['preregistration']['protocol'];manifest=read(run/'input_manifest.json')
    assert reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h,f
    root,hashes=verified_source(reg['phase'],p);assert root==Path(manifest['source']) and hashes==manifest['source_files_sha256']
    reference,refhash=verified_v45();assert str(reference)==manifest['v45_reference'] and refhash==manifest['v45_reference_sha256']
    market=core.load_market();assert market.manifest==manifest['market']['full_market'];market=restrict_market(market,pd.Timestamp(manifest['market']['observation_cutoff_exclusive']))
    features=pd.read_parquet(root/'features.parquet');panel=pd.read_parquet(run/'macro_panel.parquet');audit_macro_features(run)
    history={};predictions=read(run/'predictions.json');prev=None;calchecks=actions=days=prefix_queries=prefix_days=0
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            saved=read(run/'training'/(tag(at)+'.json'));source=read(root/'training'/(tag(at)+'.json'));assert saved['source_record_sha256']==sha(root/'training'/(tag(at)+'.json')) and saved['raw']==source['raws']['full']
            assert pd.Timestamp(saved['at'])==at and pd.Timestamp(saved['date'])==row.date and saved['clock']==row.clock and pd.Timestamp(saved['cutoff'])==day_cutoff(at)
            if row.date!=prev:
                daily=read(run/'daily'/(str(row.date.date())+'.json'));models,c=daily_check(panel,market,history,at,daily);calchecks+=c;days+=1;prev=row.date
                source_day=read(root/'daily'/(str(row.date.date())+'.json'))
                for f in ['parents','members','feedback','ready']:assert daily[f]==source_day[f],f
                if daily['ready']:assert daily['models']['plain']==source_day['variants']['full']
                if row.date<pd.Timestamp('2026-05-01'):
                    old_day=read(reference/'daily'/(str(row.date.date())+'.json'))
                    for f in ['parents','members','feedback','ready']:assert daily[f]==old_day[f],f
                    for k in CANDIDATES:
                        if daily['ready']:assert daily['models'][k]==old_day['models'][k]
                    prefix_days+=1
            raw=saved['raw'];ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready']
            for k in CANDIDATES:
                info=saved['estimates'][k];assert info['ready']==bool(ready)
                if ready:
                    x=np.r_[raw,[panel.loc[at,c] for c in MACRO],CLOCKS.index(row.clock)];xx=x[[0,1,10]] if k=='plain' else x
                    point=np.array(raw)+reference_predict(models[k]['point_model'],xx)[0];draws=np.array([raw+reference_predict(m,xx)[0] for m in models[k]['bootstrap_models']]);radius=max(0.,sorted(np.max(draws-point,axis=1))[179])
                    close(info['point'],point);close(info['bootstrap_predictions'],draws);near(info['radius'],radius);close(info['lower'],point-radius)
                else:assert info['point'] is None and info['lower'] is None
            assert saved['estimates']['plain']==source['estimates']['full']
            if row.date<pd.Timestamp('2026-05-01'):
                old=read(reference/'training'/(tag(at)+'.json'))
                for k in CANDIDATES:assert saved['estimates'][k]==old['estimates'][k]
                for name in ['strong_band','plain_band']:assert saved['events'][name]==old['events'][name]
                prefix_queries+=1
            for name,es in predictions.items():
                e=es[i];assert e==saved['events'][name] and pd.Timestamp(e['at'])==at and pd.Timestamp(e['date'])==row.date
                if name=='cash' or name=='premarket_only' and row.clock!=CLOCKS[0]:assert e['action']=='CASH' and e['allocation']==0
                else:check_action(saved['estimates']['plain' if name=='plain_band' else 'strong'],daily['feedback']['expired_unresolved_prefix'],name!='strong_point',e)
                actions+=1
            history[at]=raw
            if (i+1)%200==0:print('fixed candidate verified through',at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    settings=p[reg['phase']];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings.get('months',settings.get('regression_months'))
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(features[features.date.between(start,end)].index)==expected;assert result['coverage']=={'complete':True,'expected_ticks':len(expected),'observed_ticks':len(expected),'dates':len(set(panel.loc[expected,'date']))}
    books=[]
    for name,stats in result['routes'].items():
        es=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert es==frozen['routes'][name]
        books.append(book_check(run,name,market,features,labels(root/('stress_labels.json' if name=='double_cost' else 'baseline_labels.json')),es,2 if name=='double_cost' else 1,stats,months))
    for suffix in ['equity','trades','orders','decisions']:
        pd.testing.assert_frame_equal(pd.read_csv(run/('plain_band_'+suffix+'.csv')),pd.read_csv(root/('full_band_'+suffix+'.csv')),atol=1e-12,rtol=1e-12)
        if reg['phase']=='development':
            for name in ['strong_band','plain_band']:pd.testing.assert_frame_equal(pd.read_csv(run/(name+'_'+suffix+'.csv')),pd.read_csv(reference/(name+'_'+suffix+'.csv')),atol=1e-12,rtol=1e-12)
    if reg['phase']=='development':assert result['routes']['strong_band']==read(reference/'results.json')['routes']['strong_band']
    if reg['phase']=='cases':
        dev=Path(reg['authorization']['run']);old=read(dev/'predictions.json')
        for name,es in predictions.items():assert es[:len(old[name])]==old[name]
        for folder in ['daily','training']:
            for f in (dev/folder).glob('*.json'):assert f.read_bytes()==(run/folder/f.name).read_bytes(),(folder,f.name)
    stats=audit_statistics(run);assert result['primary_name']=='strong_band' and result['primary']==result['routes']['strong_band']
    if reg['phase']=='development':assert result['development_gate']==development_gate(result['primary'],result['coverage']['complete'],p)
    else:
        from run_fixed import authorize_cases
        assert reg['authorization']==authorize_cases(reg['authorization']['run'],p)
        assert result['case_acceptance']['passed']==bool(result['coverage']['complete'] and result['primary']['primary_pass'])
    receipt={'passed':True,'phase':reg['phase'],'clock_ticks':len(features),'daily_fits':days,'independent_calibration_equations':calchecks,'action_checks':actions,'capital_books':books,'independent_statistics':stats,
        'v45_prefix_exact_queries':prefix_queries,'v45_prefix_exact_days':prefix_days,'own_development_prefix_bytes_exact':reg['phase']=='cases','base_models_reused_daily_as_issued_not_retrained_by_v46':True,'results_sha256':sha(run/'results.json'),'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
