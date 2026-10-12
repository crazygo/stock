"""Independent state recurrence, mature memberships, model equations and books."""
import argparse,gzip,pickle
from threadpoolctl import threadpool_limits
from regime_learning import *
from audit_regime import reference_states,check_estimate,close,near
from verify_clock_reliability_v2 import compare_reencoded_labels
from verify_continuation import key,raw_exit
from verify_feedback_v2 import terminal_audit
from audit_books import book_check
from audit_models import model_equations
from audit_inputs import input_check
from clock_reliability import columns,training_dates,load_panel,PANEL_ROOT


def verify(run):
    run=Path(run);reg=read(run/'registration.json');manifest=read(run/'input_manifest.json');p=reg['preregistration']['protocol']
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    assert reg['protocol_sha256']==sha(Path(__file__).parent/'protocol.json')
    source=Path(manifest['source_run'])
    for f,h in manifest['source_files_sha256'].items():assert sha(source/f)==h
    market=core.load_market();assert market.manifest==manifest['full_market'];cutoff=pd.Timestamp(manifest['observation_cutoff_exclusive'])
    market=restrict_market(market,cutoff);features=pd.read_parquet(run/'features.parquet');panel=pd.read_parquet(run/'panel.parquet')
    assert panel.clock.equals(features.clock)
    if reg['phase']=='development':
        pd.testing.assert_frame_equal(features,pd.read_parquet(source/'features.parquet'))
        old=pd.read_parquet(source/'panel.parquet');pd.testing.assert_frame_equal(panel[old.columns],old)
        for f in ['baseline_labels.json','stress_labels.json']:compare_reencoded_labels(read(run/f),read(source/f))
        input_audit={'exact_verified_v23_inputs':True}
    else:
        fs=Path(manifest['new_inputs']['source']);assert sha(fs/'features.parquet')==manifest['new_inputs']['features_sha256']
        old=pd.read_parquet(fs/'features.parquet');pd.testing.assert_frame_equal(features,old[old.index<cutoff])
        leaders=load_panel(market.calendar,PANEL_ROOT);assert leaders[2]==manifest['new_inputs']['panel']
        input_audit=input_check(run,market,leaders[0],features,panel,cutoff)
    states=reference_states(features,market.sessions);stored=read(run/'states.json');assert len(stored)==len(states)
    for (day,s),saved in zip(states.items(),stored):
        for k in ['at','date']:assert pd.Timestamp(saved[k])==s[k]
        for k in ['index','consecutive_observations','ready']:assert saved[k]==s[k]
        for k in ['coordinates','posterior','last_five_change_probability','expected_run_length']:
            if s[k] is None:assert saved[k] is None
            else:close(np.asarray(saved[k],dtype=float),np.asarray(s[k],dtype=float))
    base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json');terminals={}
    for at,r in panel[panel.feature_ready].iterrows():
        releases=[terminal_audit(market,base[key(at,s)]) for s in core.ETFS]
        if all(t is not None for t in releases):terminals[at]=max(releases);assert r.terminal_release==terminals[at]
        else:assert pd.isna(r.terminal_release)
    histories={c:{} for c in CLOCKS};predictions=read(run/'predictions.json');heads=actions=raw_heads=0
    oldroot=core.ROOT/p['control_source'];delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v27/evidence/delivery_manifest.json')
    for f in ['predictions.json','score_sequential_equity.csv','score_sequential_trades.csv']:assert sha(oldroot/f)==delivery['cache_outputs_sha256'][f]
    oldevents={pd.Timestamp(e['at']):e for e in read(oldroot/'predictions.json')['score_sequential']}
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            rec=read(run/'training'/(tag(at)+'.json'));c=row.clock;history=histories[c];dc=day_cutoff(at)
            assert pd.Timestamp(rec['at'])==at and rec['clock']==c and pd.Timestamp(rec['cutoff'])==dc and pd.Timestamp(rec['state_date'])==row.date
            same=panel[panel.clock==c];prior=list(same[same.feature_ready&(same.date<row.date)&(same.index<dc)].index);prefix=[]
            for t in prior:
                if t not in terminals or terminals[t]>=dc:break
                prefix.append(t)
            pending=prior[len(prefix)] if len(prefix)<len(prior) else None
            expired=bool(pending is not None and same.loc[pending,'date']+pd.Timedelta(days=14)<=row.date)
            fb=rec['feedback'];assert fb['expired_unresolved_prefix']==expired and fb['selected_early_dates']==len(prefix) and fb['prior_dates']==len(prior)
            assert fb['known_but_waiting']==sum(t in terminals and terminals[t]<dc for t in prior[len(prefix)+1:])
            raw=rec['raw'];origin=rec['origin']
            if row.date<pd.Timestamp('2026-05-01'):
                file=source/'training'/(tag(at)+'.json');assert sha(file)==manifest['source_training_sha256'][file.name]==origin['sha256']
                want=read(file)['variants']['context']['forecasts']['early']['event']['raw']
                assert raw==([want[s] for s in core.ETFS] if want else None)
            else:
                dates,_=training_dates(same,dc);info=origin['fit'];mp=run/'models'/(tag(at)+'.pkl.gz')
                model=pickle.load(gzip.open(mp,'rb')) if info['ready'] else None
                want,n=model_equations(same,dates['early'],at,model,info,columns(c,'context'));raw_heads+=n
                if want:close(raw,[want[s] for s in core.ETFS])
                else:assert raw is None
            ids=[t for t in history if t in set(prefix) and history[t] is not None][-120:]
            for mode in ['regime','hazard']:heads+=check_estimate(same,history,states,ids,at,raw,mode,rec['estimates'][mode])
            for name,events in predictions.items():
                e=events[i];assert rec['events'][name]==e and pd.Timestamp(e['at'])==at
                if name=='source_v27':
                    if at in oldevents:assert e==oldevents[at]
                    else:
                        from clock_reliability import issue as old_issue
                        ref=old_issue(panel,history,at,np.array(raw) if raw else None)['band']
                        for k,v in ref.items():assert e[k]==v
                elif name=='cash' or (name=='premarket_only' and c!=CLOCKS[0]):assert e['action']=='CASH' and e['allocation']==0.
                else:
                    info=rec['estimates']['hazard' if name=='hazard_only_band' else 'regime'];ready=info['ready'] and not expired
                    assert e['ready']==ready
                    if ready:
                        values=info['point' if name=='regime_point' else 'lower'];j=0 if values[0]>=values[1] else 1
                        action=core.ETFS[j] if values[j]>1e-12 else 'CASH';assert e['values']==values
                    else:action='CASH';assert e['values'] is None
                    assert e['action']==action and e['allocation']==float(action!='CASH')
                actions+=1
            history[at]=raw
            if (i+1)%200==0:print('verified states and models through',at,flush=True)
    exits=0
    if reg['phase']=='cases':
        for lab,multiple in [(base,1),(stress,2)]:
            for label in lab.values():
                if 'entry_fill_economic' in label:raw_exit(market,label,None,multiple);exits+=1
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    settings=p[reg['phase']];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings.get('months',settings.get('regression_months'))
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(features[features.date.between(start,end)].index)==expected and result['coverage']['complete'];books=[]
    for name,s in result['routes'].items():
        wanted=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert frozen['routes'][name]==wanted
        books.append(book_check(run,name,market,features,stress if name=='double_cost' else base,wanted,2 if name=='double_cost' else 1,s,months))
    prefix=None
    if reg['phase']=='development':
        for suffix in ['equity','trades']:
            old=pd.read_csv(oldroot/('score_sequential_'+suffix+'.csv'));new=pd.read_csv(run/('source_v27_'+suffix+'.csv'))
            pd.testing.assert_frame_equal(new[old.columns],old,atol=1e-12,rtol=1e-12)
    else:
        dev=Path(reg['authorization']['run']);old=read(dev/'predictions.json')
        for name,events in old.items():assert predictions[name][:len(events)]==events
        for f in (dev/'training').glob('*.json'):assert read(f)==read(run/'training'/f.name)
        oldstates=read(dev/'states.json');assert read(run/'states.json')[:len(oldstates)]==oldstates
        prefix={'all_preMay_emissions_and_states_exact':True}
    receipt={'passed':True,'phase':reg['phase'],'state_dates':len(states),'clock_ticks':len(features),'joint_terminal_receipts':len(terminals),
        'independent_error_model_head_equations':heads,'independent_new_raw_model_equations':raw_heads,'action_checks':actions,
        'new_case_raw_exit_checks':exits,'capital_books':books,'input_audit':input_audit,'case_prefix':prefix,
        'source_v27_development_exact':reg['phase']=='development','results_sha256':sha(run/'results.json'),'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
