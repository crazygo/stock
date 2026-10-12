"""Clock isolation, source audit, all bootstrap equations and sequential capital."""
import argparse,gzip,pickle
from pathlib import Path
from threadpoolctl import threadpool_limits
from clock_reliability import *
from verify_reliability import check_estimate,close
from verify_continuation import key,near,raw_exit
from verify_feedback_v2 import terminal_audit,verify_asof_replays
from audit_models import model_equations
from audit_inputs import input_check
from audit_books import book_check


def verify(run):
    run=Path(run);reg=read(run/'registration.json');m=read(run/'input_manifest.json');p=reg['preregistration']['protocol']
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    assert reg['protocol_sha256']==sha(Path(__file__).parent/'protocol.json')
    source=Path(m['source_run'])
    for f,h in m['source_files_sha256'].items():assert sha(source/f)==h
    market=core.load_market();assert market.manifest==m['full_market'];cutoff=pd.Timestamp(m['observation_cutoff_exclusive'])
    market=restrict_market(market,cutoff);features=pd.read_parquet(run/'features.parquet');panel=pd.read_parquet(run/'panel.parquet')
    assert panel.clock.equals(features.clock)
    if reg['phase']=='development':
        pd.testing.assert_frame_equal(features,pd.read_parquet(source/'features.parquet'))
        old=pd.read_parquet(source/'panel.parquet');pd.testing.assert_frame_equal(panel[old.columns],old)
        for f in ['baseline_labels.json','stress_labels.json','terminal_receipts.json']:
            assert read(run/f)==read(source/f)
        input_audit={'reused_exact_verified_v23_inputs':True}
    else:
        fs=Path(m['new_inputs']['source']);assert sha(fs/'features.parquet')==m['new_inputs']['features_sha256']
        old=pd.read_parquet(fs/'features.parquet');pd.testing.assert_frame_equal(features,old[old.index<cutoff])
        leaders=load_panel(market.calendar,PANEL_ROOT);assert leaders[2]==m['new_inputs']['panel']
        input_audit=input_check(run,market,leaders[0],features,panel,cutoff)
        oldpanel=pd.read_parquet(source/'panel.parquet');oldfeatures=pd.read_parquet(source/'features.parquet')
        pd.testing.assert_frame_equal(features.loc[oldfeatures.index],oldfeatures)
        for c in CLOCKS:
            ids=oldfeatures[oldfeatures.clock==c].index
            close(panel.loc[ids,columns(c,'context')],oldpanel.loc[ids,columns(c,'context')])
    base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json');terminal={}
    for at,row in panel[panel.feature_ready].iterrows():
        pair=[base[key(at,s)] for s in core.ETFS];rel=[terminal_audit(market,b) for b in pair]
        if all(t is not None for t in rel):terminal[at]=max(rel);assert row.terminal_release==terminal[at]
        else:assert pd.isna(row.terminal_release)
    predictions=read(run/'predictions.json');history={c:{} for c in CLOCKS};heads=actions=base_checks=0
    v26=core.ROOT/p['premarket_development_control'];old26=read(v26/'predictions.json')['score_band'];old26map={pd.Timestamp(e['at']):e for e in old26}
    v26delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v26/evidence/delivery_manifest.json')
    for f in ['predictions.json','score_band_equity.csv','score_band_trades.csv']:
        assert sha(v26/f)==v26delivery['cache_outputs_sha256'][f]
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            c=row.clock;same=panel[panel.clock==c];dc=day_cutoff(at);record=read(run/'training'/(tag(at)+'.json'))
            assert record['clock']==c and pd.Timestamp(record['cutoff'])==dc and pd.Timestamp(record['at'])==at
            prior=list(same[same.feature_ready&(same.date<row.date)&(same.index<dc)].index);early=[]
            for t in prior:
                if t not in terminal or terminal[t]>=dc:break
                early.append(t)
            expired=len(early)<len(prior) and same.loc[prior[len(early)],'date']+pd.Timedelta(days=14)<=row.date
            fb=record['feedback'];assert fb['expired_unresolved_prefix']==bool(expired)
            assert fb['prior_dates']==len(prior) and fb['selected_early_dates']==len(early)
            assert fb['known_but_waiting']==sum(t in terminal and terminal[t]<dc for t in prior[len(early)+1:])
            origin=record['origin']
            if row.date<pd.Timestamp('2026-05-01'):
                filename=tag(at)+'.json';path=source/'training'/filename
                assert origin['kind']=='frozen_v23' and sha(path)==origin['sha256']==m['source_training_sha256'][filename]
                saved=read(path)['variants']['context'];raw=saved['forecasts']['early']['event']['raw'];assert origin['raw']==raw
            else:
                assert origin['kind']=='new_daily_raw';info=origin['fit'];assert list(pd.to_datetime(info['members'],utc=True))==early
                modelpath=run/'models'/(tag(at)+'.pkl.gz')
                if info['ready']:
                    with gzip.open(modelpath,'rb') as stream:model=pickle.load(stream)
                else:model=None;assert not modelpath.exists()
                raw,n=model_equations(same,early,at,model,info,columns(c,'context'));base_checks+=n
                assert set(raw)==set(origin['raw'])
                for k in raw:near(raw[k],origin['raw'][k])
                raw=origin['raw']
            v=np.array([raw[s] for s in core.ETFS]) if raw else None
            if v is None:assert record['raw'] is None
            else:close(record['raw'],v)
            ids=[t for t in history[c] if t in set(early) and history[c][t] is not None][-120:]
            heads+=check_estimate(same,history[c],ids,at,v,'score',record['estimate'])
            info=record['estimate'];ready=info['ready'] and not expired
            for variant,band in [('band',True),('point',False)]:
                ev=record[variant];assert ev['ready']==ready
                if ready:
                    vals=info['lower' if band else 'point'];j=0 if vals[0]>=vals[1] else 1
                    action=core.ETFS[j] if vals[j]>1e-12 else 'CASH';assert ev['values']==vals;near(ev['buy_advantage'],vals[j])
                else:action='CASH';assert ev['values'] is None
                assert ev['action']==action and ev['allocation']==float(action!='CASH')
            only={'premarket_only':CLOCKS[0],'score_1005':CLOCKS[1],'score_1035':CLOCKS[2],'score_1305':CLOCKS[3]}
            for name,events in predictions.items():
                e=events[i];assert record['events'][name]==e and pd.Timestamp(e['at'])==at
                active=name!='cash' and not(name=='score_after_open' and c==CLOCKS[0]) and (name not in only or only[name]==c)
                if active:
                    want=record['point' if name=='point_sequential' else 'band']
                    for k in want:assert e[k]==want[k]
                else:assert e['action']=='CASH' and e['allocation']==0.
                actions+=1
            if at in old26map:
                assert predictions['premarket_only'][i]==old26map[at]
                path=v26/'training'/(str(row.date.date())+'.json');assert sha(path)==v26delivery['daily_records_sha256'][path.name]
                assert record['estimate']==read(path)['estimates']['score']
            history[c][at]=v
            if (i+1)%200==0:print('verified through',at,flush=True)
    exits=0
    if reg['phase']=='cases':
        for lab,multiple in [(base,1),(stress,2)]:
            for label in lab.values():
                if 'entry_fill_economic' in label:raw_exit(market,label,None,multiple);exits+=1
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    settings=p[reg['phase']];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings.get('months',settings.get('regression_months'));books=[]
    expected=[]
    for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)]:
        for c in CLOCKS:
            at=pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC')
            if at<pd.Timestamp(market.calendar[d]['close_at']):expected.append(at)
    observed=features[features.date.between(start,end)]
    assert result['coverage']=={'complete':list(observed.index)==expected,'expected_ticks':len(expected),'observed_ticks':len(observed),
        'dates':observed.date.nunique(),'same_day_ticks_are_not_independent_dates':True}
    for name,stats in result['routes'].items():
        wanted=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert wanted==frozen['routes'][name]
        books.append(book_check(run,name,market,features,stress if name=='double_cost' else base,wanted,2 if name=='double_cost' else 1,stats,months))
    if reg['phase']=='development':
        for suffix in ['equity','trades']:
            old=pd.read_csv(v26/('score_band_'+suffix+'.csv'));new=pd.read_csv(run/('premarket_only_'+suffix+'.csv'))
            pd.testing.assert_frame_equal(new[old.columns],old,atol=1e-12,rtol=1e-12)
    prefix=None
    if reg['phase']=='cases':
        dev=Path(reg['authorization']['run']);old=read(dev/'predictions.json')
        for name,events in old.items():assert predictions[name][:len(events)]==events
        for path in (dev/'training').glob('*.json'):assert read(path)==read(run/'training'/path.name)
        prefix={'all_preMay_records_and_actions_exact':True,'records':len(list((dev/'training').glob('*.json')))}
    receipt={'passed':True,'phase':reg['phase'],'clock_ticks':len(features),'joint_terminal_receipts':len(terminal),
        'independent_error_model_heads':heads,'new_base_model_equations':base_checks,'action_checks':actions,
        'new_raw_exit_checks':exits,'input_audit':input_audit,'capital_books':books,'premarket_matches_v26':True,
        'case_prefix':prefix,'results_sha256':sha(run/'results.json'),'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
