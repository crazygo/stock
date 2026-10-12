"""Independent date-cluster fit, causal membership, all actions and capital books."""
import argparse,gzip,pickle,json
from threadpoolctl import threadpool_limits
from coherence_learning import *
from audit_calibration import daily_check,input_rows,predict as reference_predict
from audit_hierarchical import daily_check as source_daily_check
from verify_reliability import check_estimate,close
from verify_clock_reliability_v2 import compare_reencoded_labels
from audit_features import audit_features
from verify_continuation import key,near,raw_exit
from verify_feedback_v2 import terminal_audit
from audit_models import model_equations
from audit_books import book_check
from audit_inputs import input_check
from clock_reliability import load_panel,PANEL_ROOT


def verify(run):
    run=Path(run);reg=read(run/'registration.json');p=reg['preregistration']['protocol'];manifest=read(run/'input_manifest.json')
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    assert reg['protocol_sha256']==sha(Path(__file__).parent/'protocol.json')
    source=Path(manifest['source_run'])
    for f,h in manifest['source_files_sha256'].items():assert sha(source/f)==h
    market=core.load_market();assert market.manifest==manifest['full_market'];cutoff=pd.Timestamp(manifest['observation_cutoff_exclusive'])
    market=restrict_market(market,cutoff);features=pd.read_parquet(run/'features.parquet');panel=pd.read_parquet(run/'panel.parquet')
    if reg['phase']=='development':
        pd.testing.assert_frame_equal(features,pd.read_parquet(source/'features.parquet'));old=pd.read_parquet(source/'panel.parquet')
        pd.testing.assert_frame_equal(panel[old.columns],old)
        for f in ['baseline_labels.json','stress_labels.json']:compare_reencoded_labels(read(run/f),read(source/f))
        input_audit={'exact_verified_v23_features_and_fixed_labels':True}
    else:
        fs=Path(manifest['new_inputs']['source']);assert sha(fs/'features.parquet')==manifest['new_inputs']['features_sha256']
        old=pd.read_parquet(fs/'features.parquet');pd.testing.assert_frame_equal(features,old[old.index<cutoff])
        leaders=load_panel(market.calendar,PANEL_ROOT);assert leaders[2]==manifest['new_inputs']['panel']
        input_audit=input_check(run,market,leaders[0],features,panel,cutoff)
    original_panel=panel.copy();leaders=load_panel(market.calendar,PANEL_ROOT);assert leaders[2]==manifest['coherence_panel_source']
    new=pd.read_parquet(run/'coherence_features.parquet');feature_audit=audit_features(features,new,read(run/'feature_receipts.json'),leaders,market.sessions)
    panel=pd.read_parquet(run/'coherence_panel.parquet')
    pd.testing.assert_frame_equal(panel[original_panel.columns].drop(columns='feature_ready'),original_panel.drop(columns='feature_ready'))
    pd.testing.assert_frame_equal(panel[new.columns],new)
    assert panel.feature_ready.equals(original_panel.feature_ready & new.coherence_ready)
    base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json');terminals={}
    for at,row in original_panel[original_panel.feature_ready].iterrows():
        values=[terminal_audit(market,base[key(at,s)]) for s in core.ETFS]
        if all(t is not None for t in values):terminals[at]=max(values);assert row.terminal_release==max(values)
        else:assert pd.isna(row.terminal_release)
    histories={k:{} for k in VARIANTS};source_history={};predictions=read(run/'predictions.json');oldroot=core.ROOT/p['control_source']
    delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v32/evidence/delivery_manifest.json')
    for f in ['predictions.json','partial_band_equity.csv','partial_band_trades.csv']:assert sha(oldroot/f)==delivery['cache_outputs_sha256'][f]
    oldevents={pd.Timestamp(e['at']):e for e in read(oldroot/'predictions.json')['partial_band']}
    prev=None;models=None;daily=None;source_day=None;source_models=None;heads=baseheads=actions=daychecks=0
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            c=row.clock;rec=read(run/'training'/(tag(at)+'.json'));raw=rec['source_raw'];origin=rec['source_origin']
            assert pd.Timestamp(rec['at'])==at and pd.Timestamp(rec['date'])==row.date and rec['clock']==c and pd.Timestamp(rec['cutoff'])==day_cutoff(at)
            same=original_panel[original_panel.clock==c];prior=list(same[same.feature_ready&(same.date<row.date)].index);prefix=[]
            for t in prior:
                if t not in terminals or terminals[t]>=day_cutoff(at):break
                prefix.append(t)
            if at in oldevents:
                file=source/'training'/(tag(at)+'.json');assert sha(file)==manifest['source_training_sha256'][file.name]==origin['sha256']
                v=read(file)['variants']['context']['forecasts']['early']['event']['raw'];want=[v[s] for s in core.ETFS] if v else None
                assert raw==want;old=oldevents[at];assert rec['old_decision'] is None
            else:
                info=origin['fit'];assert list(pd.to_datetime(info['members'],utc=True))==prefix
                mp=run/'models'/('source_'+tag(at)+'.pkl.gz');model=pickle.load(gzip.open(mp,'rb')) if mp.exists() else None
                v,n=model_equations(same,prefix,at,model,info,old_columns(c,'context'));baseheads+=n
                if v:close(raw,[v[s] for s in core.ETFS])
                else:assert raw is None
                if row.date!=prev:
                    source_day=read(run/'source_daily'/(str(row.date.date())+'.json'))
                    source_models,n=source_daily_check(original_panel,market,source_history,at,source_day);heads+=n
                info=rec['old_decision']['partial'];ready=raw is not None and original_panel.loc[at,'feature_ready'] and source_day['ready']
                assert info['ready']==bool(ready)
                if ready:
                    x=input_rows(panel,{at:raw},[at]);m=source_models['partial']
                    point=np.array(raw)+reference_predict(m['point_model'],x)[0]
                    samples=np.array([np.array(raw)+reference_predict(z,x)[0] for z in m['bootstrap_models']])
                    radius=max(0.,sorted(np.max(samples-point,axis=1))[179]);close(info['point'],point);close(info['bootstrap_predictions'],samples)
                    near(info['radius'],radius);close(info['lower'],point-radius)
                old=check_action(info,source_day['feedback']['expired_unresolved_prefix'],True,rec['old'])
            assert rec['old']==old
            same_new=panel[panel.clock==c];prior_new=list(same_new[same_new.feature_ready&(same_new.date<row.date)].index);ids=[]
            for t in prior_new:
                if t not in terminals or terminals[t]>=day_cutoff(at):break
                ids.append(t)
            for kind in VARIANTS:
                info=rec['fits'][kind]['fit'];assert info['columns']==columns(c,kind) and list(pd.to_datetime(info['members'],utc=True))==ids
                path=run/'models'/(kind+'_'+tag(at)+'.pkl.gz');model=pickle.load(gzip.open(path,'rb')) if path.exists() else None
                values,n=model_equations(same_new,ids,at,model,info,columns(c,kind));baseheads+=n
                expected_raw=rec['fits'][kind]['raw']
                if values:
                    for k,v in values.items():near(expected_raw[k],v)
                    close(rec['raws'][kind],[values[s] for s in core.ETFS])
                else:assert expected_raw=={} and rec['raws'][kind] is None
                fb=rec['fits'][kind]['feedback'];assert fb['selected_early_dates']==len(ids)
                expired_base=len(ids)<len(prior_new) and same_new.loc[prior_new[len(ids)],'date']+pd.Timedelta(days=14)<=row.date
                assert fb['expired_unresolved_prefix']==bool(expired_base)
            if row.date!=prev:
                daily=read(run/'daily'/rec['daily_record']);models,n=daily_check(panel,market,histories,at,daily);heads+=n;daychecks+=1;prev=row.date
            assert rec['daily_record']==str(row.date.date())+'.json' and pd.Timestamp(daily['date'])==row.date
            for kind in VARIANTS:
                raw=rec['raws'][kind];ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready']
                info=rec['estimates'][kind];assert info['ready']==bool(ready)
                if ready:
                    x=input_rows(panel,{at:raw},[at]);m=models[kind];point=np.array(raw)+reference_predict(m['point_model'],x)[0]
                    samples=np.array([np.array(raw)+reference_predict(z,x)[0] for z in m['bootstrap_models']]);close(info['point'],point);close(info['bootstrap_predictions'],samples)
                    radius=max(0.,sorted(np.max(samples-point,axis=1))[179]);near(info['radius'],radius);close(info['lower'],point-radius)
                else:assert info['point'] is None and info['lower'] is None
            expired=daily['feedback']['expired_unresolved_prefix']
            for name,es in predictions.items():
                e=es[i];assert e==rec['events'][name] and pd.Timestamp(e['at'])==at and pd.Timestamp(e['date'])==row.date
                if name=='source_v32':
                    for k,v in old.items():
                        if k not in ['at','date']:assert e[k]==v
                elif name=='cash' or name=='premarket_only' and c!=CLOCKS[0]:assert e['action']=='CASH' and e['allocation']==0
                else:
                    kind='participation' if name=='participation_band' else 'baseline' if name=='baseline_band' else 'full'
                    check_action(rec['estimates'][kind],expired,name!='full_point',e)
                actions+=1
            for kind in VARIANTS:histories[kind][at]=rec['raws'][kind]
            source_history[at]=rec['source_raw']
            if (i+1)%200==0:print('verified through',at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    settings=p[reg['phase']];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings.get('months',settings.get('regression_months'))
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(features[features.date.between(start,end)].index)==expected and result['coverage']['complete'];books=[]
    for name,stats in result['routes'].items():
        wanted=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert frozen['routes'][name]==wanted
        books.append(book_check(run,name,market,features,stress if name=='double_cost' else base,wanted,2 if name=='double_cost' else 1,stats,months))
    if reg['phase']=='development':
        for suffix in ['equity','trades']:
            old=pd.read_csv(oldroot/('partial_band_'+suffix+'.csv'));new=pd.read_csv(run/('source_v32_'+suffix+'.csv'))
            pd.testing.assert_frame_equal(new[old.columns],old,atol=1e-12,rtol=1e-12)
    else:
        dev=Path(reg['authorization']['run']);old=read(dev/'predictions.json')
        for name,events in old.items():assert predictions[name][:len(events)]==events
        for folder in ['training','daily','source_daily']:
            for f in (dev/folder).glob('*.json'):assert read(f)==read(run/folder/f.name)
    receipt={'passed':True,'phase':reg['phase'],'clock_ticks':len(features),'daily_fits':daychecks,
        'independent_model_head_equations':heads,'new_base_model_equations':baseheads,'action_checks':actions,
        'terminal_pairs_checked':len(terminals),'coherence_features':feature_audit,'capital_books':books,'input_audit':input_audit,
        'source_v32_development_exact':reg['phase']=='development','case_prefix_exact':reg['phase']=='cases',
        'results_sha256':sha(run/'results.json'),'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


def check_action(info,expired,band,e):
    ready=info['ready'] and not expired;assert e['ready']==ready
    if ready:
        values=info['lower' if band else 'point'];j=0 if values[0]>=values[1] else 1;action=core.ETFS[j] if values[j]>1e-12 else 'CASH'
        assert e['values']==values and e['top_symbol']==core.ETFS[j];near(e['buy_advantage'],values[j])
    else:action='CASH';assert e['values'] is None
    assert e['action']==action and e['allocation']==float(action!='CASH')
    return e


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
