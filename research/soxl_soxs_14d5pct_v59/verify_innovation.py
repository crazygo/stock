"""Independent state-ready training members, original-horizon equations and capital."""
import argparse,gzip,pickle
from threadpoolctl import threadpool_limits
from innovation_learning import *
from audit_state_features import audit as audit_inputs
from audit_dense_calibration import daily_check,input_rows,predict as ref_predict
from audit_models import model_equations
from audit_capital import book_check
from audit_statistics import audit_statistics
from verify_financial import check_action
from verify_reliability import close
from verify_continuation import near,key
from verify_feedback_v2 import terminal_audit


def check_base(panel,at,fit,model,cols):
    same=panel[panel.clock==panel.loc[at,'clock']]
    prior_ids=list(same[same.feature_ready&(same.date<panel.loc[at,'date'])].index);ids=[]
    for t in prior_ids:
        if pd.isna(same.loc[t,'terminal_release']) or same.loc[t,'terminal_release']>=day_cutoff(at):break
        ids.append(t)
    info=fit['fit'];assert list(pd.to_datetime(info['members'],utc=True))==ids and info['columns']==cols
    raw,n=model_equations(same,ids,at,model,info,cols);fb=fit['feedback']
    assert fb['prior_dates']==len(prior_ids) and fb['selected_early_dates']==len(ids)
    expired=len(ids)<len(prior_ids) and same.loc[prior_ids[len(ids)],'date']+pd.Timedelta(days=14)<=panel.loc[at,'date']
    assert fb['expired_unresolved_prefix']==bool(expired)
    for k,v in raw.items():near(fit['raw'][k],v)
    if not raw:assert fit['raw']=={}
    return np.array([raw[s] for s in core.ETFS]) if raw else None,n


def expected_info(panel,at,raw,daily,models):
    ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready'];out={'ready':bool(ready),'point':None,'lower':None}
    if ready:
        x=input_rows(panel,{at:raw},[at],CLOCKS);point=raw+ref_predict(models['point_model'],x)[0]
        draws=np.array([raw+ref_predict(z,x)[0] for z in models['bootstrap_models']]);radius=max(0.,sorted(np.max(draws-point,axis=1))[179])
        out.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=draws.tolist())
    return out


def check_info(saved,want):
    assert saved['ready']==want['ready']
    if want['ready']:
        for f in ['point','lower','radius','bootstrap_predictions']:close(saved[f],want[f])
    else:assert saved==want


def verify(run):
    run=Path(run);registration=read(run/'registration.json');p=registration['preregistration']['protocol'];manifest=read(run/'input_manifest.json')
    assert registration['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in registration['source_sha256'].items():assert sha(core.ROOT/f)==h
    for f,h in manifest['derived_files_sha256'].items():assert sha(run/f)==h
    source,source_sha,prior_sha=verified_source();assert source_sha==registration['source_v56_manifest_sha256']
    if registration['phase']=='development':
        audit=read(run/'state_audit.json');assert audit['passed']
        assert sha(run/'state_audit.json')==registration['preregistration']['state_audit_sha256']
        assert audit['input_manifest_sha256']==sha(run/'input_manifest.json')==registration['preregistration']['development_inputs_sha256']
    else:audit_inputs(run);audit=read(run/'state_audit.json')
    market=core.load_market();assert market.manifest==manifest['full_market'];market=restrict_market(market,pd.Timestamp(manifest['observation_cutoff_exclusive']))
    features=pd.read_parquet(run/'features.parquet');panel=pd.read_parquet(run/'innovation_panel.parquet');source_panel=pd.read_parquet(run/'activity_panel.parquet');original=pd.read_parquet(run/'financial_panel.parquet')
    base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json');terminals={}
    for at,row in panel[panel.feature_ready].iterrows():
        t=[terminal_audit(market,base[key(at,s)]) for s in core.ETFS]
        if all(x is not None for x in t):terminals[at]=max(t);assert row.terminal_release==max(t)
        else:assert pd.isna(row.terminal_release)
    predictions=read(run/'predictions.json');assert set(predictions)==set(ROUTES)
    histories={k:{} for k in VARIANTS};basehist={k:{} for k in ['v56','v55']};previous=None
    cal_checks=base_checks=actions=days=source_equations=0
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel.iterrows()):
            rec=read(run/'training'/(tag(at)+'.json'))
            assert pd.Timestamp(rec['at'])==at and pd.Timestamp(rec['date'])==row.date and pd.Timestamp(rec['cutoff'])==day_cutoff(at) and rec['clock']==row.clock
            if row.date!=previous:
                filename=str(row.date.date())+'.json';daily=read(run/'daily'/filename);references={}
                for k in VARIANTS:references[k],n=daily_check(panel,market,histories[k],at,daily[k]);cal_checks+=n
                source_day=read(run/'source_daily'/filename);oldfile=source/'daily'/filename;source_models={}
                if oldfile.exists():assert source_day=={'v56':read(oldfile)['full'],'v55':read(source/'source_daily'/filename)}
                else:
                    for k,table in [('v56',source_panel),('v55',original)]:source_models[k],n=daily_check(table,market,basehist[k],at,source_day[k]);source_equations+=n
                days+=1;previous=row.date
            members=[]
            for k in VARIANTS:
                path=run/'models'/(k+'_'+tag(at)+'.pkl.gz');model=pickle.load(gzip.open(path,'rb')) if path.exists() else None
                raw,n=check_base(panel,at,rec['fits'][k],model,columns(row.clock,k));base_checks+=n;members.append(rec['fits'][k]['fit']['members'])
                if raw is None:assert rec['raws'][k] is None
                else:close(rec['raws'][k],raw)
                check_info(rec['estimates'][k],expected_info(panel,at,raw,daily[k],references[k]));histories[k][at]=raw
            assert all(m==members[0] for m in members)
            oldfile=source/'training'/(tag(at)+'.json')
            if oldfile.exists():
                old=read(oldfile);wanted={'v56':{'raw':old['raws']['full'],'fit':old['fits']['full'],'estimate':old['estimates']['full']},'v55':{k:old['baseline'][k] for k in ['raw','fit','estimate']}}
                for k in wanted:wanted[k]['origin']={'kind':'frozen_v56','sha256':sha(oldfile)}
                assert rec['baselines']==wanted
                assert rec['events']['source_v56']==old['events']['activity_band'] and rec['events']['source_v55']==old['events']['source_v55']
            else:
                for k,table,cols in [('v56',source_panel,columns(row.clock)),('v55',original,old55_columns(row.clock))]:
                    saved=rec['baselines'][k];path=run/'baseline_models'/(k+'_'+tag(at)+'.pkl.gz');model=pickle.load(gzip.open(path,'rb')) if path.exists() else None
                    raw,n=check_base(table,at,saved['fit'],model,cols);source_equations+=n
                    assert saved['origin']=={'kind':'same_formula_extension'}
                    if raw is None:assert saved['raw'] is None
                    else:close(saved['raw'],raw)
                    check_info(saved['estimate'],expected_info(table,at,raw,source_day[k],source_models[k]))
                    check_action(saved['estimate'],source_day[k]['feedback']['expired_unresolved_prefix'],True,rec['events']['source_'+k])
            for k in basehist:
                raw=rec['baselines'][k]['raw'];basehist[k][at]=np.array(raw) if raw is not None else None
            for name,es in predictions.items():
                e=es[i];assert e==rec['events'][name] and pd.Timestamp(e['at'])==at and pd.Timestamp(e['date'])==row.date
                if name in ['source_v56','source_v55']:pass
                elif name=='cash' or name=='premarket_only' and row.clock!=CLOCKS[0]:assert e['action']=='CASH' and e['allocation']==0
                else:
                    k='difference' if name=='difference_band' else 'matched' if name=='matched_band' else 'innovation';expired=bool(daily[k]['feedback']['expired_unresolved_prefix'] or rec['fits'][k]['feedback']['expired_unresolved_prefix'])
                    check_action(rec['estimates'][k],expired,name!='innovation_point',e)
                actions+=1
            if (i+1)%400==0:print('innovation independently verified through',at,flush=True)
    r=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json')
    settings=p[registration['phase']];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings.get('months',settings.get('regression_months'))
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(features[features.date.between(start,end)].index)==expected
    assert r['coverage']=={'complete':True,'expected_ticks':len(expected),'observed_ticks':len(expected),'dates':len(set(panel.loc[expected,'date'])),'same_day_ticks_are_not_independent_dates':True}
    books=[]
    for name,stats in r['routes'].items():
        es=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert es==frozen['routes'][name]
        books.append(book_check(run,name,market,features,stress if name=='double_cost' else base,es,2 if name=='double_cost' else 1,stats,months,require_full_cash=False))
    if registration['phase']=='development':
        for name,oldname in [('source_v56','activity_band'),('source_v55','source_v55')]:
            for suffix in ['equity','trades','orders','decisions']:assert sha(run/(name+'_'+suffix+'.csv'))==sha(source/(oldname+'_'+suffix+'.csv'))
    else:
        dev=Path(registration['authorization']['run']);old=read(dev/'predictions.json')
        for name,es in old.items():assert predictions[name][:len(es)]==es
        oldpanel=pd.read_parquet(dev/'innovation_panel.parquet');pd.testing.assert_frame_equal(panel.loc[oldpanel.index],oldpanel)
        oldstates=read(dev/'state_records.json');assert read(run/'state_records.json')[:len(oldstates)]==oldstates
        for folder in ['training','daily','source_daily']:
            for f in (dev/folder).glob('*.json'):assert read(f)==read(run/folder/f.name)
    n=audit_statistics(run);assert r['primary_name']=='innovation_band' and r['primary']==r['routes']['innovation_band']
    if registration['phase']=='development':assert r['development_gate']==development_gate(r['primary'],True,p)
    else:assert r['case_acceptance']['passed']==bool(r['primary']['primary_pass'])
    receipt={'passed':True,'phase':registration['phase'],'queries':len(panel),'daily_records':days,'variant_calibration_equations':cal_checks,'base_head_equations':base_checks,'action_checks':actions,'source_extension_equations':source_equations,'original_terminal_pairs':len(terminals),'capital_books':books,'statistics':n,'input_audit':audit,'source_v56_v55_exact':registration['phase']=='development','case_prefix_exact':registration['phase']=='cases','results_sha256':sha(run/'results.json'),'engineering_checks_are_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
