"""Independent features, all three variant fits, causal action and cash audits."""
import argparse,gzip,pickle
from threadpoolctl import threadpool_limits
from activity_learning import *
from audit_activity_features import audit as audit_inputs
from audit_dense_calibration import daily_check,input_rows,predict as ref_predict
from audit_models import model_equations
from audit_capital import book_check
from audit_statistics import audit_statistics
from verify_financial import check_action
from verify_reliability import close
from verify_continuation import near,key
from verify_feedback_v2 import terminal_audit


def check_base(panel,at,fit,model,cols,terminals):
    same=panel[panel.clock==panel.loc[at,'clock']];prior_ids=list(same[same.feature_ready&(same.date<panel.loc[at,'date'])].index);ids=[]
    for t in prior_ids:
        if t not in terminals or terminals[t]>=day_cutoff(at):break
        ids.append(t)
    info=fit['fit'];assert list(pd.to_datetime(info['members'],utc=True))==ids and info['columns']==cols
    raw,n=model_equations(same,ids,at,model,info,cols);fb=fit['feedback'];assert fb['prior_dates']==len(prior_ids) and fb['selected_early_dates']==len(ids)
    expired=len(ids)<len(prior_ids) and same.loc[prior_ids[len(ids)],'date']+pd.Timedelta(days=14)<=panel.loc[at,'date'];assert fb['expired_unresolved_prefix']==bool(expired)
    for k,v in raw.items():near(fit['raw'][k],v)
    if not raw:assert fit['raw']=={}
    return np.array([raw[s] for s in core.ETFS]) if raw else None,n


def expected_info(panel,at,raw,daily,models):
    ready=raw is not None and panel.loc[at,'feature_ready'] and daily['ready'];out={'ready':bool(ready),'point':None,'lower':None}
    if ready:
        x=input_rows(panel,{at:raw},[at],CLOCKS);point=raw+ref_predict(models['point_model'],x)[0];draws=np.array([raw+ref_predict(z,x)[0] for z in models['bootstrap_models']]);radius=max(0.,sorted(np.max(draws-point,axis=1))[179]);out.update(point=point.tolist(),lower=(point-radius).tolist(),radius=radius,bootstrap_predictions=draws.tolist())
    return out


def verify(run):
    run=Path(run);reg=read(run/'registration.json');p=reg['preregistration']['protocol'];manifest=read(run/'input_manifest.json');assert reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in reg['source_sha256'].items():assert sha(core.ROOT/f)==h
    for f,h in manifest['derived_files_sha256'].items():assert sha(run/f)==h
    source,source_sha=verified_source();assert source_sha==reg['source_v55_manifest_sha256']
    if reg['phase']=='development':
        input_audit=read(run/'activity_audit.json');assert input_audit['passed'] and sha(run/'activity_audit.json')==reg['preregistration']['activity_audit_sha256'];assert input_audit['input_manifest_sha256']==sha(run/'input_manifest.json')==reg['preregistration']['development_inputs_sha256']
    else:audit_inputs(run);input_audit=read(run/'activity_audit.json')
    market=core.load_market();assert market.manifest==manifest['full_market'];market=restrict_market(market,pd.Timestamp(manifest['observation_cutoff_exclusive']))
    features=pd.read_parquet(run/'features.parquet');original=pd.read_parquet(run/'financial_panel.parquet');panel=pd.read_parquet(run/'activity_panel.parquet');base=labels(run/'baseline_labels.json');stress=labels(run/'stress_labels.json');terminals={}
    for at,row in original[original.feature_ready].iterrows():
        t=[terminal_audit(market,base[key(at,s)]) for s in core.ETFS]
        if all(x is not None for x in t):terminals[at]=max(t);assert row.terminal_release==max(t)
        else:assert pd.isna(row.terminal_release)
    predictions=read(run/'predictions.json');assert set(predictions)==set(ROUTES);histories={k:{} for k in VARIANTS};oldhistory={};previous=None;cal_checks=base_checks=actions=days=source_equations=0
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel.iterrows()):
            rec=read(run/'training'/(tag(at)+'.json'));assert pd.Timestamp(rec['at'])==at and pd.Timestamp(rec['date'])==row.date and pd.Timestamp(rec['cutoff'])==day_cutoff(at) and rec['clock']==row.clock
            if row.date!=previous:
                daily=read(run/'daily'/(str(row.date.date())+'.json'));references={}
                for k in VARIANTS:references[k],n=daily_check(panel,market,histories[k],at,daily[k]);cal_checks+=n
                old_day=read(run/'source_daily'/(str(row.date.date())+'.json'));oldfile=source/'daily'/(str(row.date.date())+'.json')
                if oldfile.exists():assert old_day==read(oldfile);oldmodels=None
                else:oldmodels,n=daily_check(original,market,oldhistory,at,old_day);source_equations+=n
                days+=1;previous=row.date
            for k in VARIANTS:
                path=run/'models'/(k+'_'+tag(at)+'.pkl.gz');model=pickle.load(gzip.open(path,'rb')) if path.exists() else None
                raw,n=check_base(panel,at,rec['fits'][k],model,columns(row.clock,k),terminals);base_checks+=n
                if raw is None:assert rec['raws'][k] is None
                else:close(rec['raws'][k],raw)
                info=expected_info(panel,at,raw,daily[k],references[k]);saved=rec['estimates'][k];assert saved['ready']==info['ready']
                if info['ready']:
                    for field in ['point','lower','radius','bootstrap_predictions']:close(saved[field],info[field])
                else:assert saved==info
                histories[k][at]=raw
            oldfile=source/'training'/(tag(at)+'.json');saved=rec['baseline']
            if oldfile.exists():
                old=read(oldfile);assert saved=={'raw':old['raw'],'fit':old['fit'],'estimate':old['estimate'],'origin':{'kind':'frozen_v55','sha256':sha(oldfile)}};baseline=old['events']['dense_band']
            else:
                path=run/'baseline_models'/(tag(at)+'.pkl.gz');model=pickle.load(gzip.open(path,'rb')) if path.exists() else None
                raw,n=check_base(original,at,saved['fit'],model,prior.columns(row.clock,'full'),terminals);source_equations+=n
                assert saved['origin']=={'kind':'same_formula_extension'}
                if raw is None:assert saved['raw'] is None
                else:close(saved['raw'],raw)
                info=expected_info(original,at,raw,old_day,oldmodels);assert saved['estimate']['ready']==info['ready']
                if info['ready']:
                    for field in ['point','lower','radius','bootstrap_predictions']:close(saved['estimate'][field],info[field])
                else:assert saved['estimate']==info
                baseline=rec['events']['source_v55'];ready=info['ready'] and not old_day['feedback']['expired_unresolved_prefix'];assert baseline['ready']==ready
                if ready:
                    values=info['lower'];j=0 if values[0]>=values[1] else 1;action=core.ETFS[j] if values[j]>1e-12 else 'CASH';close(baseline['values'],values);assert baseline['top_symbol']==core.ETFS[j];near(baseline['buy_advantage'],values[j])
                else:action='CASH';assert baseline['values'] is None
                assert baseline['action']==action and baseline['allocation']==float(action!='CASH')
            oldhistory[at]=np.array(saved['raw']) if saved['raw'] is not None else None
            for name,es in predictions.items():
                e=es[i];assert e==rec['events'][name] and pd.Timestamp(e['at'])==at and pd.Timestamp(e['date'])==row.date
                if name=='source_v55':assert e==baseline
                elif name=='cash' or name=='four_only' and row.clock not in OLD_CLOCKS or name=='premarket_only' and row.clock!=CLOCKS[0]:assert e['action']=='CASH' and e['allocation']==0
                else:
                    kind='pair' if name=='pair_band' else 'sector' if name=='sector_band' else 'full';check_action(rec['estimates'][kind],daily[kind]['feedback']['expired_unresolved_prefix'],name!='activity_point',e)
                actions+=1
            if (i+1)%400==0:print('activity independently verified through',at,flush=True)
    r=read(run/'results.json');frozen=read(run/'decisions_frozen.json');assert frozen['predictions_sha256']==sha(run/'predictions.json');settings=p[reg['phase']];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings.get('months',settings.get('regression_months'))
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    assert list(features[features.date.between(start,end)].index)==expected;assert r['coverage']=={'complete':True,'expected_ticks':len(expected),'observed_ticks':len(expected),'dates':len(set(panel.loc[expected,'date'])),'same_day_ticks_are_not_independent_dates':True};books=[]
    for name,stats in r['routes'].items():
        es=[e for e in predictions[name] if start<=e['date'][:10]<=end];assert es==frozen['routes'][name];books.append(book_check(run,name,market,features,stress if name=='double_cost' else base,es,2 if name=='double_cost' else 1,stats,months,require_full_cash=False))
    if reg['phase']=='development':
        for suffix in ['equity','trades','orders','decisions']:assert sha(run/('source_v55_'+suffix+'.csv'))==sha(source/('dense_band_'+suffix+'.csv'))
    else:
        dev=Path(reg['authorization']['run']);old=read(dev/'predictions.json')
        for name,es in old.items():assert predictions[name][:len(es)]==es
        for folder in ['training','daily','source_daily']:
            for f in (dev/folder).glob('*.json'):assert read(f)==read(run/folder/f.name)
    n=audit_statistics(run);assert r['primary_name']=='activity_band' and r['primary']==r['routes']['activity_band']
    if reg['phase']=='development':assert r['development_gate']==development_gate(r['primary'],True,p)
    else:assert r['case_acceptance']['passed']==bool(r['primary']['primary_pass'])
    receipt={'passed':True,'phase':reg['phase'],'queries':len(panel),'daily_records':days,'variant_calibration_equations':cal_checks,'base_head_equations':base_checks,'action_checks':actions,'source_extension_equations':source_equations,'terminal_pairs':len(terminals),'capital_books':books,'statistics':n,'input_audit':input_audit,'source_v55_exact':reg['phase']=='development','case_prefix_exact':reg['phase']=='cases','results_sha256':sha(run/'results.json'),'engineering_checks_are_not_return_acceptance':True}
    dump(run/'verification.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
