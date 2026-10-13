"""Freeze eight-clock as-issued actions before any new capital outcome."""
import argparse,gzip,pickle,shutil,subprocess
from datetime import datetime,timezone
from threadpoolctl import threadpool_limits
from dense_learning import *
from cycle_ledger import replay
from run_intraday import score
from run_breadth import paired_uncertainty
from study import save_replay


def authorize_cases(path,p):
    root=Path(path);reg=read(root/'registration.json');r=read(root/'results.json');v=read(root/'verification.json')
    assert reg['phase']=='development' and reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    assert r['primary_name']==p['primary'] and r['primary']==r['routes'][p['primary']]
    assert v['passed'] and v['results_sha256']==sha(root/'results.json');gate=development_gate(r['primary'],r['coverage']['complete'],p)
    if not gate['passed']:raise ValueError('Development primary failed; new case replay is forbidden')
    return {'run':str(root.resolve()),'results_sha256':sha(root/'results.json'),'verification_sha256':sha(root/'verification.json'),'gate':gate}


def run(output,case_after=None,inputs=None):
    p=read(SOURCE/'protocol.json');prereg=read(SOURCE/'evidence/preregistration.json');assert prereg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in prereg['code_sha256'].items():assert sha(core.ROOT/f)==h
    auth=authorize_cases(case_after,p) if case_after else None;phase='cases' if auth else 'development';output=Path(output)
    if output.exists():raise ValueError('Use a new output directory; preserve every attempt')
    dense_source,dense_sha=verified_dense_source();assert dense_sha==prereg['dense_source_manifest_sha256']
    old_engine=source_module('dense_learning.py','v54_frozen_learning')
    inputs=Path(inputs) if inputs else (dense_source if phase=='development' else core.ROOT/'.cache/soxl_soxs_14d5pct_v55/inputs/cases_20261013')
    if not inputs.exists():
        assert phase=='cases'
        subprocess.run([sys.executable,str(core.ROOT/'research/soxl_soxs_14d5pct_v54/prepare_dense.py'),'--output',str(inputs),'--phase',phase],check=True)
    manifest=read(inputs/'input_manifest.json');assert manifest['phase']==phase
    if phase=='development':assert sha(inputs/'input_manifest.json')==prereg['development_inputs_sha256']
    assert read(inputs/'input_registration.json')['protocol_sha256']==sha(core.ROOT/'research/soxl_soxs_14d5pct_v54/protocol.json')
    for f,h in manifest['derived_files_sha256'].items():assert sha(inputs/f)==h
    source,hashes=verified_source(phase,p);assert manifest['source_run']==str(source) and hashes==manifest['source_files_sha256']
    market=core.load_market();assert market.manifest==manifest['full_market'];market=restrict_market(market,pd.Timestamp(manifest['observation_cutoff_exclusive']))
    for folder in ['','training','daily','source_daily','models','executed_sources']:(output/folder).mkdir(parents=True,exist_ok=True)
    for f in inputs.iterdir():
        if f.is_file() and (f.name in manifest['derived_files_sha256'] or f.name in ['input_manifest.json','input_audit.json','coverage.json']):shutil.copyfile(f,output/f.name)
    dump(output/'registration.json',{'phase':phase,'started_at':datetime.now(timezone.utc).isoformat(),'preregistration':prereg,'protocol_sha256':sha(SOURCE/'protocol.json'),
        'authorization':auth,'source_sha256':prereg['code_sha256'],'prepared_inputs':str(inputs.resolve()),'input_manifest_sha256':sha(inputs/'input_manifest.json'),'dense_source_manifest_sha256':dense_sha})
    for f in prereg['code_sha256']:
        dest=output/'executed_sources'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((core.ROOT/f).read_bytes())
    panel=pd.read_parquet(output/'financial_panel.parquet');features=pd.read_parquet(output/'features.parquet');reg=registry(panel,market);history={};predictions={k:[] for k in ROUTES};previous=None;daily=None
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            if row.date!=previous:
                daily=fit_day(panel,history,reg,at);dayfile=str(row.date.date())+'.json';dump(output/'daily'/dayfile,daily);previous=row.date
                old_day_file=dense_source/'daily'/dayfile
                source_daily=read(old_day_file) if old_day_file.exists() else old_engine.fit_day(panel,history,reg,at)
                dump(output/'source_daily'/dayfile,source_daily)
            if row.clock in OLD_CLOCKS:
                f=source/'training'/(tag(at)+'.json');old=read(f);fit=old['fits']['full'];raw=old['raws']['full'];raw=np.array(raw) if raw is not None else None;model=None
                origin={'kind':'frozen_old_clock','source_record_sha256':sha(f)};old_event=old['events']['full_band']
            else:
                model,fit,raw=fit_base(panel,at);origin={'kind':'new_clock_daily_refit'};old_event=None
                if model is not None:
                    with gzip.open(output/'models'/(tag(at)+'.pkl.gz'),'wb') as stream:pickle.dump(model,stream)
            source_record=dense_source/'training'/(tag(at)+'.json')
            if source_record.exists():
                old_dense=read(source_record)
                from run_planning import json_ready
                import json
                assert json.loads(json.dumps(json_ready(fit),default=str))==old_dense['fit']
                np.testing.assert_allclose(raw,old_dense['raw'],rtol=0,atol=0) if raw is not None else None
                assert (raw is None)==(old_dense['raw'] is None)
                dense_event=old_dense['events']['dense_band']
            else:
                old_info=old_engine.issue(panel,at,raw,source_daily)
                dense_event={**decide(old_info,source_daily['feedback'],True),'at':at,'date':row.date}
            info=issue(panel,at,raw,daily);issued=events(at,row.date,row.clock,info,daily['feedback'],old_event,dense_event)
            record={'at':at,'date':row.date,'clock':row.clock,'cutoff':day_cutoff(at),'fit':fit,'raw':raw,'origin':origin,'estimate':info,'events':issued}
            dump(output/'training'/(tag(at)+'.json'),record)
            for k,e in issued.items():predictions[k].append(e)
            history[at]=raw
            if (i+1)%400==0:print('dense forecasts trained through',at,flush=True)
    dump(output/'predictions.json',predictions);settings=p[phase];start,end=settings['evaluation_start'],settings['evaluation_end'];months=settings.get('months',settings.get('regression_months'))
    actions={k:[e for e in es if start<=str(e['date'].date())<=end] for k,es in predictions.items()}
    dump(output/'decisions_frozen.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'new_v55_capital_outcomes_seen':False,'prior_versions_exposed_history':True,'predictions_sha256':sha(output/'predictions.json'),'routes':actions})
    test=features[features.date.between(start,end)];base=labels(output/'baseline_labels.json');stress=labels(output/'stress_labels.json');books={};result={'phase':phase,'primary_name':p['primary'],'routes':{}}
    for k,es in actions.items():
        multiple=2 if k=='double_cost' else 1;book=replay(market,test,list((stress if multiple==2 else base).values()),{e['at']:e['action'] for e in es},start,end,cost_multiple=multiple,allocations={e['at']:e['allocation'] for e in es},require_full_cash=False)
        books[k]=book;save_replay(output,k,book);result['routes'][k]=score(book,months)
    expected=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for d in market.sessions[(market.sessions>=start)&(market.sessions<=end)] for c in CLOCKS if pd.Timestamp(str(d.date())+' '+c,tz=core.ET)<pd.Timestamp(market.calendar[d]['close_at'])]
    complete=list(test.index)==expected;result['coverage']={'complete':complete,'expected_ticks':len(expected),'observed_ticks':len(test),'dates':test.date.nunique(),'same_day_ticks_are_not_independent_dates':True}
    result['primary']=result['routes'][p['primary']];result['paired_increments']={k:paired_uncertainty(books[p['primary']][0],books[k][0]) for k in ROUTES if k not in [p['primary'],'cash','double_cost']}
    if phase=='development':
        gate=development_gate(result['primary'],complete,p);result['development_gate']=gate;result['case_regression']={'executed':False,'eligible_after_verification':gate['passed']}
    else:result['case_acceptance']={'passed':bool(complete and result['primary']['primary_pass'])}
    result.update(goal_complete=False,new_buy_eligible=False,future_nonnegative_month_guarantee=False);dump(output/'results.json',result)
    print({k:{'mean':s['primary_average'],'total':s['continuous_return']} for k,s in result['routes'].items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--case-after',type=Path);p.add_argument('--inputs',type=Path);a=p.parse_args();run(a.output,a.case_after,a.inputs)
