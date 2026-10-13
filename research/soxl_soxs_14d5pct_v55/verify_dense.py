"""Independently verify every new model, causal date membership and capital book."""
import argparse,gzip,pickle
from threadpoolctl import threadpool_limits
from dense_learning import *
from audit_dense_calibration import daily_check,input_rows,predict as reference_predict
from audit_models import model_equations
from audit_capital import book_check
from audit_statistics import audit_statistics
from verify_financial import check_action
from verify_reliability import close
from verify_continuation import near,key
from verify_feedback_v2 import terminal_audit


def verify(run):
    run=Path(run);reg=read(run/'registration.json');p=reg['preregistration']['protocol'];manifest=read(run/'input_manifest.json')
    assert reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    for f,h in manifest['derived_files_sha256'].items():assert sha(run/f)==h
    dense_source,dense_sha=verified_dense_source();assert reg['dense_source_manifest_sha256']==dense_sha
    old_audit=source_module('audit_dense_calibration.py','v54_frozen_audit');source_checks=0
    source,hashes=verified_source(reg['phase'],p);assert manifest['source_run']==str(source) and manifest['source_files_sha256']==hashes
    if reg['phase']=='development':
        assert sha(run/'input_audit.json')==reg['preregistration']['input_audit_sha256']
        input_audit=read(run/'input_audit.json');assert input_audit['passed'] and input_audit['input_manifest_sha256']==sha(run/'input_manifest.json')==reg['preregistration']['development_inputs_sha256']
    else:
        import subprocess
        subprocess.run([sys.executable,str(core.ROOT/'research/soxl_soxs_14d5pct_v54/preflight_inputs.py'),'--inputs',str(run)],check=True);input_audit=read(run/'input_audit.json')
    market=core.load_market();assert market.manifest==manifest['full_market'];market=restrict_market(market,pd.Timestamp(manifest['observation_cutoff_exclusive']))
    features=pd.read_parquet(run/'features.parquet');panel=pd.read_parquet(run/'financial_panel.parquet');base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json');terminals={}
    for at,row in panel[panel.feature_ready].iterrows():
        release=[terminal_audit(market,base[key(at,s)]) for s in core.ETFS]
        if all(t is not None for t in release):terminals[at]=max(release);assert row.terminal_release==max(release)
        else:assert pd.isna(row.terminal_release)
    predictions=read(run/'predictions.json');assert set(predictions)==set(ROUTES);h={};prev=None;models=None;daily=None;cal_checks=base_checks=actions=old_count=days=0
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            rec=read(run/'training'/(tag(at)+'.json'));assert pd.Timestamp(rec['at'])==at and pd.Timestamp(rec['date'])==row.date and rec['clock']==row.clock and pd.Timestamp(rec['cutoff'])==day_cutoff(at)
            if row.date!=prev:
                daily=read(run/'daily'/(str(row.date.date())+'.json'));models,n=daily_check(panel,market,h,at,daily);cal_checks+=n;days+=1;prev=row.date
                old_day=read(run/'source_daily'/(str(row.date.date())+'.json'));original=dense_source/'daily'/(str(row.date.date())+'.json')
                if original.exists():
                    assert old_day==read(original);old_models=None
                    assert {k:v for k,v in daily.items() if k!='model'}=={k:v for k,v in old_day.items() if k!='model'}
                else:old_models,n=old_audit.daily_check(panel,market,h,at,old_day);source_checks+=n
            if row.clock in OLD_CLOCKS:
                old_file=source/'training'/(tag(at)+'.json');old=read(old_file);assert rec['origin']=={'kind':'frozen_old_clock','source_record_sha256':sha(old_file)}
                assert rec['fit']==old['fits']['full'] and rec['raw']==old['raws']['full'];assert rec['events']['old_four_band']==old['events']['full_band'];old_count+=1
            else:
                assert rec['origin']=={'kind':'new_clock_daily_refit'};same=panel[panel.clock==row.clock];prior=list(same[same.feature_ready&(same.date<row.date)].index);ids=[]
                for t in prior:
                    if t not in terminals or terminals[t]>=day_cutoff(at):break
                    ids.append(t)
                info=rec['fit']['fit'];assert list(pd.to_datetime(info['members'],utc=True))==ids and info['columns']==columns(row.clock,'full')
                model_file=run/'models'/(tag(at)+'.pkl.gz');model=pickle.load(gzip.open(model_file,'rb')) if model_file.exists() else None
                raw,n=model_equations(same,ids,at,model,info,columns(row.clock,'full'));base_checks+=n
                if raw:
                    for k,v in raw.items():near(rec['fit']['raw'][k],v)
                    close(rec['raw'],[raw[s] for s in core.ETFS])
                else:assert rec['raw'] is None and rec['fit']['raw']=={}
                fb=rec['fit']['feedback'];assert fb['prior_dates']==len(prior) and fb['selected_early_dates']==len(ids)
                expired=len(ids)<len(prior) and same.loc[prior[len(ids)],'date']+pd.Timedelta(days=14)<=row.date;assert fb['expired_unresolved_prefix']==bool(expired)
            raw=np.array(rec['raw']) if rec['raw'] is not None else None
            original_record=dense_source/'training'/(tag(at)+'.json')
            if original_record.exists():
                old_record=read(original_record);assert rec['fit']==old_record['fit'] and rec['raw']==old_record['raw'];dense_event=old_record['events']['dense_band']
            else:
                old_ready=raw is not None and panel.loc[at,'feature_ready'] and old_day['ready']
                old_info={'ready':bool(old_ready),'point':None,'lower':None}
                if old_ready:
                    x=input_rows(panel,{at:raw},[at],CLOCKS);pt=raw+old_audit.predict(old_models['point_model'],x)[0];draws=np.array([raw+old_audit.predict(z,x)[0] for z in old_models['bootstrap_models']]);rad=max(0.,sorted(np.max(draws-pt,axis=1))[179]);old_info.update(point=pt.tolist(),lower=(pt-rad).tolist())
                dense_event=rec['events']['source_dense'];ready_old=old_info['ready'] and not old_day['feedback']['expired_unresolved_prefix'];assert dense_event['ready']==ready_old
                if ready_old:
                    values=old_info['lower'];j=0 if values[0]>=values[1] else 1;want_action=core.ETFS[j] if values[j]>1e-12 else 'CASH'
                    close(dense_event['values'],values);assert dense_event['top_symbol']==core.ETFS[j];near(dense_event['buy_advantage'],values[j])
                else:want_action='CASH';assert dense_event['values'] is None
                assert dense_event['action']==want_action and dense_event['allocation']==float(want_action!='CASH')
            info=rec['estimate'];ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready'];assert info['ready']==bool(ready)
            if ready:
                x=input_rows(panel,{at:raw},[at],CLOCKS);point=raw+reference_predict(models['point_model'],x)[0];samples=np.array([raw+reference_predict(z,x)[0] for z in models['bootstrap_models']])
                radius=max(0.,sorted(np.max(samples-point,axis=1))[179]);close(info['point'],point);close(info['bootstrap_predictions'],samples);near(info['radius'],radius);close(info['lower'],point-radius)
            else:assert info=={'ready':False,'point':None,'lower':None}
            for name,es in predictions.items():
                e=es[i];assert e==rec['events'][name] and pd.Timestamp(e['at'])==at and pd.Timestamp(e['date'])==row.date
                if name=='source_dense':assert e==dense_event
                elif name=='old_four_band':
                    if row.clock not in OLD_CLOCKS:assert e['action']=='CASH' and e['allocation']==0
                elif name=='cash' or name=='dense_four_only' and row.clock not in OLD_CLOCKS or name=='added_only' and row.clock in OLD_CLOCKS or name=='premarket_only' and row.clock!=CLOCKS[0]:assert e['action']=='CASH' and e['allocation']==0
                else:check_action(info,daily['feedback']['expired_unresolved_prefix'],name!='dense_point',e)
                actions+=1
            h[at]=raw
            if (i+1)%400==0:print('dense independently verified through',at,flush=True)
    result=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json');settings=p[reg['phase']];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings.get('months',settings.get('regression_months'))
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(features[features.date.between(start,end)].index)==expected
    assert result['coverage']=={'complete':True,'expected_ticks':len(expected),'observed_ticks':len(expected),'dates':len(set(panel.loc[expected,'date'])),'same_day_ticks_are_not_independent_dates':True};books=[]
    for name,stats in result['routes'].items():
        es=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert es==frozen['routes'][name];books.append(book_check(run,name,market,features,stress if name=='double_cost' else base,es,2 if name=='double_cost' else 1,stats,months,require_full_cash=False))
    for suffix in ['equity','trades','orders','decisions']:
        old=pd.read_csv(source/('full_band_'+suffix+'.csv'));new=pd.read_csv(run/('old_four_band_'+suffix+'.csv'))
        if suffix=='decisions':new=new[new.clock.isin(OLD_CLOCKS)].reset_index(drop=True)
        pd.testing.assert_frame_equal(new,old,atol=1e-12,rtol=1e-12)
    if reg['phase']=='development':
        for suffix in ['equity','trades','orders','decisions']:
            assert sha(run/('source_dense_'+suffix+'.csv'))==sha(dense_source/('dense_band_'+suffix+'.csv'))
    if reg['phase']=='cases':
        dev=Path(reg['authorization']['run']);old=read(dev/'predictions.json')
        for name,es in old.items():assert predictions[name][:len(es)]==es
        for folder in ['training','daily','source_daily']:
            for f in (dev/folder).glob('*.json'):assert read(f)==read(run/folder/f.name)
    n=audit_statistics(run);assert result['primary_name']=='dense_band' and result['primary']==result['routes']['dense_band']
    if reg['phase']=='development':assert result['development_gate']==development_gate(result['primary'],True,p)
    else:assert result['case_acceptance']['passed']==bool(result['primary']['primary_pass'])
    receipt={'passed':True,'phase':reg['phase'],'clock_ticks':len(features),'daily_fits':days,'independent_calibration_equations':cal_checks,'new_base_head_equations':base_checks,'action_checks':actions,'source_extension_equations':source_checks,'v54_raws_exact':True,'source_dense_capital_exact':reg['phase']=='development',
        'terminal_pairs':len(terminals),'old_clock_frozen_raws_and_actions_exact':old_count,'old_four_capital_exact':True,'capital_books':books,'statistic_assertions':n,'input_audit':input_audit,
        'case_prefix_exact':reg['phase']=='cases','results_sha256':sha(run/'results.json'),'engineering_verification_is_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
