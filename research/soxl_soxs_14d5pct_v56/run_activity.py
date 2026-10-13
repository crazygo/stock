"""Register inputs, fit chronological variant histories, freeze actions, then replay."""
import argparse,gzip,pickle,shutil
from datetime import datetime,timezone
from threadpoolctl import threadpool_limits
from activity_learning import *
from cycle_ledger import replay
from run_intraday import score
from run_breadth import paired_uncertainty
from study import save_replay


def authorize_cases(path,p):
    run=Path(path);reg=read(run/'registration.json');r=read(run/'results.json');v=read(run/'verification.json')
    assert reg['phase']=='development' and reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    assert r['primary_name']==p['primary'] and r['primary']==r['routes'][p['primary']]
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    gate=development_gate(r['primary'],r['coverage']['complete'],p)
    if not gate['passed']:raise ValueError('Development primary failed; new case replay is forbidden')
    return {'run':str(run.resolve()),'results_sha256':sha(run/'results.json'),'verification_sha256':sha(run/'verification.json'),'gate':gate}


def run(output,case_after=None,inputs=None):
    p=read(SOURCE/'protocol.json');prereg=read(SOURCE/'evidence/preregistration.json');assert prereg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in prereg['code_sha256'].items():assert sha(core.ROOT/f)==h
    auth=authorize_cases(case_after,p) if case_after else None;phase='cases' if auth else 'development';output=Path(output);assert not output.exists()
    inputs=Path(inputs) if inputs else core.ROOT/(prereg['development_inputs'] if phase=='development' else '.cache/soxl_soxs_14d5pct_v56/inputs/cases_20261013')
    if not inputs.exists():
        assert auth
        from prepare_activity import prepare_inputs
        from audit_activity_features import audit
        prepare_inputs(inputs,phase);audit(inputs)
    manifest=read(inputs/'input_manifest.json');assert manifest['phase']==phase
    assert read(inputs/'input_registration.json')['protocol_sha256']==sha(SOURCE/'protocol.json')
    if phase=='development':assert sha(inputs/'input_manifest.json')==prereg['development_inputs_sha256'] and sha(inputs/'activity_audit.json')==prereg['activity_audit_sha256']
    for f,h in manifest['derived_files_sha256'].items():assert sha(inputs/f)==h
    assert read(inputs/'activity_audit.json')['passed'];source,source_sha=verified_source();assert source_sha==prereg['source_v55_manifest_sha256']==manifest['source_v55_manifest_sha256']
    market=core.load_market();assert market.manifest==manifest['full_market'];market=restrict_market(market,pd.Timestamp(manifest['observation_cutoff_exclusive']))
    for folder in ['','training','daily','source_daily','models','baseline_models','executed_sources']:(output/folder).mkdir(parents=True,exist_ok=True)
    for f in inputs.iterdir():
        if f.is_file():shutil.copyfile(f,output/f.name)
    dump(output/'registration.json',{'phase':phase,'started_at':datetime.now(timezone.utc).isoformat(),'preregistration':prereg,'protocol_sha256':sha(SOURCE/'protocol.json'),'authorization':auth,'source_sha256':prereg['code_sha256'],'source_v55_manifest_sha256':source_sha,'input_manifest_sha256':sha(inputs/'input_manifest.json')})
    for f in prereg['code_sha256']:
        dest=output/'executed_sources'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((core.ROOT/f).read_bytes())
    panel=pd.read_parquet(output/'activity_panel.parquet');original=pd.read_parquet(output/'financial_panel.parquet');features=pd.read_parquet(output/'features.parquet')
    reg=registry(panel,market);oldreg=registry(original,market);histories={k:{} for k in VARIANTS};oldhistory={};predictions={k:[] for k in ROUTES};previous=None
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel.iterrows()):
            dayfile=str(row.date.date())+'.json'
            if row.date!=previous:
                daily={k:fit_day(panel,histories[k],reg,at) for k in VARIANTS};dump(output/'daily'/dayfile,daily);previous=row.date
                oldfile=source/'daily'/dayfile;source_day=read(oldfile) if oldfile.exists() else prior.fit_day(original,oldhistory,oldreg,at);dump(output/'source_daily'/dayfile,source_day)
            models,fits,raws=fit_bases(panel,at)
            for kind,model in models.items():
                if model is not None:
                    with gzip.open(output/'models'/(kind+'_'+tag(at)+'.pkl.gz'),'wb') as f:pickle.dump(model,f)
            oldfile=source/'training'/(tag(at)+'.json')
            if oldfile.exists():
                old=read(oldfile);oldraw=np.array(old['raw']) if old['raw'] is not None else None;oldfit=old['fit'];oldinfo=old['estimate'];baseline=old['events']['dense_band'];oldorigin={'kind':'frozen_v55','sha256':sha(oldfile)}
            else:
                model,oldfit,oldraw=prior.fit_base(original,at);oldinfo=prior.issue(original,at,oldraw,source_day);baseline=decide(oldinfo,source_day['feedback'],True);oldorigin={'kind':'same_formula_extension'}
                if model is not None:
                    with gzip.open(output/'baseline_models'/(tag(at)+'.pkl.gz'),'wb') as f:pickle.dump(model,f)
            estimates={k:issue(panel,at,raws[k],daily[k]) for k in VARIANTS};issued=events(at,row.date,row.clock,estimates,daily,baseline)
            dump(output/'training'/(tag(at)+'.json'),{'at':at,'date':row.date,'clock':row.clock,'cutoff':day_cutoff(at),'fits':fits,'raws':raws,'estimates':estimates,'baseline':{'raw':oldraw,'fit':oldfit,'estimate':oldinfo,'origin':oldorigin},'events':issued})
            for k,e in issued.items():predictions[k].append(e)
            for k in VARIANTS:histories[k][at]=raws[k]
            oldhistory[at]=oldraw
            if (i+1)%400==0:print('activity forecasts through',at,flush=True)
    dump(output/'predictions.json',predictions);settings=p[phase];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings.get('months',settings.get('regression_months'))
    actions={k:[e for e in es if start<=str(e['date'].date())<=end] for k,es in predictions.items()};dump(output/'decisions_frozen.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'new_capital_outcomes_seen':False,'prior_exposed_history':True,'predictions_sha256':sha(output/'predictions.json'),'routes':actions})
    test=features[features.date.between(start,end)];base=labels(output/'baseline_labels.json');stress=labels(output/'stress_labels.json');books={};result={'phase':phase,'primary_name':p['primary'],'routes':{}}
    for name,es in actions.items():
        multiple=2 if name=='double_cost' else 1;book=replay(market,test,list((stress if multiple==2 else base).values()),{e['at']:e['action'] for e in es},start,end,cost_multiple=multiple,allocations={e['at']:e['allocation'] for e in es},require_full_cash=False)
        books[name]=book;save_replay(output,name,book);result['routes'][name]=score(book,months)
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    complete=list(test.index)==expected;result['coverage']={'complete':complete,'expected_ticks':len(expected),'observed_ticks':len(test),'dates':test.date.nunique(),'same_day_ticks_are_not_independent_dates':True};result['primary']=result['routes'][p['primary']]
    result['paired_increments']={k:paired_uncertainty(books[p['primary']][0],books[k][0]) for k in ROUTES if k not in [p['primary'],'cash','double_cost']}
    if phase=='development':
        gate=development_gate(result['primary'],complete,p);result['development_gate']=gate;result['case_regression']={'executed':False,'eligible_after_verification':gate['passed']}
    else:result['case_acceptance']={'passed':bool(complete and result['primary']['primary_pass'])}
    result.update(goal_complete=False,new_buy_eligible=False,future_nonnegative_month_guarantee=False);dump(output/'results.json',result)
    print({k:{'mean':s['primary_average'],'total':s['continuous_return']} for k,s in result['routes'].items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--case-after',type=Path);p.add_argument('--inputs',type=Path);a=p.parse_args();run(a.output,a.case_after,a.inputs)
