"""Independent risk anchors, event competition, daily training and cash ledgers."""
import argparse,gzip,pickle
from threadpoolctl import threadpool_limits
from overnight import *
from audit_paths import reference_forecasts,reference_label,compare
from verify_reliability import check_estimate,close
from verify_clock_reliability_v2 import compare_reencoded_labels
from verify_continuation import near
from audit_models import model_equations
from audit_books import book_check
from audit_inputs import input_check
from clock_reliability import load_panel,PANEL_ROOT


def compare_records(actual,expected):
    # dump() writes timestamps using JSON default=str; compare the same wire type.
    import json
    from run_planning import json_ready
    from audit_paths import compare_records as numeric_compare
    a=json.loads(json.dumps(json_ready(actual),default=str))
    b=json.loads(json.dumps(json_ready(expected),default=str))
    numeric_compare(a,b)


def prior_members(panel,at):
    day=panel.loc[at,'date'];cutoff=day_cutoff(at);prior=list(panel[panel.feature_ready&(panel.date<day)&(panel.index<cutoff)].index)
    prefix=[]
    for t in prior:
        if pd.isna(panel.loc[t,'terminal_release']) or panel.loc[t,'terminal_release']>=cutoff:break
        prefix.append(t)
    pending=prior[len(prefix)] if len(prefix)<len(prior) else None
    expired=bool(pending is not None and panel.loc[pending,'date']+pd.Timedelta(days=14)<=day)
    return prior,prefix,expired


def verify(run):
    run=Path(run);reg=read(run/'registration.json');p=reg['preregistration']['protocol'];manifest=read(run/'input_manifest.json')
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    assert reg['protocol_sha256']==sha(Path(__file__).parent/'protocol.json')
    source=Path(manifest['source_run'])
    for f,h in manifest['source_files_sha256'].items():assert sha(source/f)==h
    market=core.load_market();assert market.manifest==manifest['full_market'];cutoff=pd.Timestamp(manifest['observation_cutoff_exclusive'])
    market=restrict_market(market,cutoff);features=pd.read_parquet(run/'features.parquet');original=pd.read_parquet(run/'source_panel.parquet');panel=pd.read_parquet(run/'panel.parquet')
    if reg['phase']=='development':
        pd.testing.assert_frame_equal(features,pd.read_parquet(source/'features.parquet'));old=pd.read_parquet(source/'panel.parquet')
        pd.testing.assert_frame_equal(original[old.columns],old)
        for f in ['baseline_labels.json','stress_labels.json']:compare_reencoded_labels(read(run/f),read(source/f))
        input_audit={'exact_verified_v23_features_and_fixed_labels':True}
    else:
        fs=Path(manifest['new_inputs']['source']);assert sha(fs/'features.parquet')==manifest['new_inputs']['features_sha256']
        old=pd.read_parquet(fs/'features.parquet');pd.testing.assert_frame_equal(features,old[old.index<cutoff])
        leaders=load_panel(market.calendar,PANEL_ROOT);assert leaders[2]==manifest['new_inputs']['panel']
        input_audit=input_check(run,market,leaders[0],features,original,cutoff)
    base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json')
    gaprows,gaps,anchors=reference_forecasts(market,features);compare_records(read(run/'gap_records.json'),gaprows)
    stored={(pd.Timestamp(x['date']),x['symbol']):x for x in read(run/'gap_forecasts.json')};assert set(stored)==set(gaps)
    for (d,s),f in gaps.items():compare_records(stored[(d,s)],{'symbol':s,**f})
    sets={};label_checks=review_checks=0
    for name,source_lab,mode,multiple in [('buffer',base,'buffer',1),('buffer_stress',stress,'buffer',2),('flat',base,'flat',1)]:
        managed=read_managed(run/(name+'_labels.json'));reviews=[];assert set(managed)==set(source_lab)
        for k,b in source_lab.items():
            event,rs=reference_label(market,b,gaps,mode,multiple);compare(managed[k],event);reviews.extend(rs);label_checks+=1
        compare_records(read(run/(name+'_reviews.json')),reviews);review_checks+=len(reviews);sets[name]=managed
        print('verified raw managed paths',name,label_checks,flush=True)
    unchanged=[c for c in original if c not in ['C','D','M','terminal_release']];pd.testing.assert_frame_equal(panel[unchanged],original[unchanged])
    for at in panel.index:
        pair=[sets['buffer'][key(at,s)] for s in core.ETFS];values=np.log1p(np.array([r.get('net_return') for r in pair],dtype=float))
        close(panel.loc[at,['C','D','M']].to_numpy(dtype=float),[(values[0]+values[1])/2,(values[0]-values[1])/2,abs((values[0]-values[1])/2)])
        releases=[r.get('terminal_release') for r in pair]
        if all(t is not None for t in releases):assert panel.loc[at,'terminal_release']==max(releases)
        else:assert pd.isna(panel.loc[at,'terminal_release'])
    histories={c:{} for c in CLOCKS};old_histories={c:{} for c in CLOCKS};predictions=read(run/'predictions.json')
    oldroot=core.ROOT/p['control_source'];delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v29/evidence/delivery_manifest.json')
    for f in ['predictions.json','regime_band_equity.csv','regime_band_trades.csv']:assert sha(oldroot/f)==delivery['cache_outputs_sha256'][f]
    oldevents={pd.Timestamp(e['at']):e for e in read(oldroot/'predictions.json')['regime_band']}
    if reg['phase']=='cases':
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'soxl_soxs_14d5pct_v29'))
        from audit_regime import reference_states
        from regime_learning import issue as old_issue,decide as old_decide
        oldstates=reference_states(features,market.sessions)
    raw_checks=error_checks=actions=0
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            c=row.clock;same=panel[panel.clock==c];history=histories[c];rec=read(run/'training'/(tag(at)+'.json'));raw=rec['raw']
            assert pd.Timestamp(rec['at'])==at and pd.Timestamp(rec['cutoff'])==day_cutoff(at) and rec['clock']==c
            prior,prefix,expired=prior_members(same,at);info=rec['fit'];assert list(pd.to_datetime(info['members'],utc=True))==prefix
            for name in ['feedback','base_feedback']:
                fb=rec[name];assert fb['expired_unresolved_prefix']==expired and fb['prior_dates']==len(prior) and fb['selected_early_dates']==len(prefix)
            mp=run/'models'/(tag(at)+'.pkl.gz');models=pickle.load(gzip.open(mp,'rb')) if mp.exists() else {'managed':None,'old':None}
            want,n=model_equations(same,prefix,at,models['managed'],info,columns(c,'context'));raw_checks+=n
            if want:close(raw,[want[s] for s in core.ETFS])
            else:assert raw is None
            ids=[t for t in history if t in set(prefix) and history[t] is not None][-120:]
            error_checks+=check_estimate(same,history,ids,at,np.array(raw) if raw else None,'score',rec['estimate'])
            for name,band in [('band',True),('point',False)]:
                e=rec[name];ready=rec['estimate']['ready'] and not expired;assert e['ready']==ready
                if ready:
                    values=rec['estimate']['lower' if band else 'point'];j=0 if values[0]>=values[1] else 1
                    action=core.ETFS[j] if values[j]>1e-12 else 'CASH';assert e['values']==values
                else:action='CASH';assert e['values'] is None
                assert e['action']==action and e['allocation']==float(action!='CASH')
            origin=rec['old_origin'];oldraw=rec['old_raw']
            if at in oldevents:
                f=source/'training'/(tag(at)+'.json');assert sha(f)==manifest['source_training_sha256'][f.name]==origin['sha256']
                want=read(f)['variants']['context']['forecasts']['early']['event']['raw'];assert oldraw==([want[s] for s in core.ETFS] if want else None)
                oldev=oldevents[at]
            else:
                olds=original[original.clock==c];_,oldids,_=prior_members(olds,at)
                want,n=model_equations(olds,oldids,at,models['old'],origin['fit'],columns(c,'context'));raw_checks+=n
                if want:close(oldraw,[want[s] for s in core.ETFS])
                else:assert oldraw is None
                dec=old_issue(original,old_histories[c],oldstates,at,np.array(oldraw) if oldraw else None)
                oldev=old_decide(dec['estimates']['regime'],dec['feedback'],True)
            for name,events in predictions.items():
                e=events[i];assert rec['events'][name]==e and pd.Timestamp(e['at'])==at
                ref=oldev if name in ['source_v29','source_v29_buffer'] else rec['point' if name=='buffer_point' else 'band']
                if name=='cash':assert e['action']=='CASH' and e['allocation']==0
                else:
                    for k,v in ref.items():assert e[k]==v
                actions+=1
            history[at]=raw;old_histories[c][at]=oldraw
            if (i+1)%200==0:print('verified daily models through',at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    settings=p[reg['phase']];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings.get('months',settings.get('regression_months'))
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(features[features.date.between(start,end)].index)==expected and result['coverage']['complete'];books=[]
    for name,stats in result['routes'].items():
        wanted=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert frozen['routes'][name]==wanted
        books.append(book_check(run,name,market,features,labels_for(name,sets,base),wanted,2 if name=='double_cost' else 1,stats,months))
    prefix=None
    if reg['phase']=='development':
        for suffix in ['equity','trades']:
            old=pd.read_csv(oldroot/('regime_band_'+suffix+'.csv'));new=pd.read_csv(run/('source_v29_'+suffix+'.csv'))
            pd.testing.assert_frame_equal(new[old.columns],old,atol=1e-12,rtol=1e-12)
    else:
        dev=Path(reg['authorization']['run']);old=read(dev/'predictions.json')
        for name,events in old.items():assert predictions[name][:len(events)]==events
        for f in (dev/'training').glob('*.json'):assert read(f)==read(run/'training'/f.name)
        prefix={'all_preMay_daily_training_and_actions_exact':True}
    receipt={'passed':True,'phase':reg['phase'],'clock_ticks':len(features),'gap_anchor_pairs':anchors,'gap_forecasts':len(gaps),
        'independent_managed_label_checks':label_checks,'independent_holding_reviews':review_checks,
        'independent_base_model_equations':raw_checks,'independent_error_model_head_equations':error_checks,'action_checks':actions,
        'capital_books':books,'input_audit':input_audit,'case_prefix':prefix,'source_v29_development_exact':reg['phase']=='development',
        'results_sha256':sha(run/'results.json'),'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
