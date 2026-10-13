"""Chronological immutable state features, supervised fits, calibration and capital."""
import argparse,gzip,pickle,shutil,subprocess
from datetime import datetime,timezone
from threadpoolctl import threadpool_limits
from path_learning import *
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



def case_inputs(output):
    from prepare_activity import prepare_inputs as prepare_activity
    from audit_activity_features import audit as audit_activity
    from prepare_paths import prepare_inputs
    from audit_path_bank import audit
    base=core.ROOT/'.cache/soxl_soxs_14d5pct_v60/inputs/activity_cases_20261013'
    if not base.exists():
        prepare_activity(base,'cases')
        ancestor=Path(read(base/'input_manifest.json')['base_inputs'])
        subprocess.run([sys.executable,str(core.ROOT/'research/soxl_soxs_14d5pct_v54/preflight_inputs.py'),'--inputs',str(ancestor)],check=True)
        audit_activity(base)
    assert read(base/'activity_audit.json')['passed']
    prepare_inputs(output,'cases',base);audit(output)
    from audit_activity_inputs import audit as activity_inputs_audit
    activity_inputs_audit(output)
    from audit_freshness_inputs import audit as freshness_inputs_audit
    freshness_inputs_audit(output)
    from audit_volatility import audit as volatility_inputs_audit
    volatility_inputs_audit(output)


def run(output,case_after=None,inputs=None):
    p=read(SOURCE/'protocol.json');prereg=read(SOURCE/'evidence/preregistration.json');assert prereg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in prereg['code_sha256'].items():assert sha(core.ROOT/f)==h
    auth=authorize_cases(case_after,p) if case_after else None;phase='cases' if auth else 'development';output=Path(output);assert not output.exists()
    inputs=Path(inputs) if inputs else core.ROOT/(prereg['development_inputs'] if phase=='development' else '.cache/soxl_soxs_14d5pct_v68/inputs/cases_20261013')
    if not inputs.exists():
        assert auth
        case_inputs(inputs)
    manifest=read(inputs/'input_manifest.json');assert manifest['phase']==phase
    assert read(inputs/'input_registration.json')['protocol_sha256']==sha(SOURCE/'protocol.json')
    if phase=='development':assert sha(inputs/'input_manifest.json')==prereg['development_inputs_sha256'] and sha(inputs/'bank_audit.json')==prereg['bank_audit_sha256']
    for f,h in manifest['derived_files_sha256'].items():assert sha(inputs/f)==h
    assert read(inputs/'activity_audit.json')['passed']
    if phase=='development':assert sha(inputs/'activity_audit.json')==prereg['activity_audit_sha256']
    assert read(inputs/'freshness_audit.json')['passed'] and read(inputs/'freshness_audit.json')['input_manifest_sha256']==sha(inputs/'input_manifest.json')
    if phase=='development':assert sha(inputs/'freshness_audit.json')==prereg['freshness_audit_sha256']
    assert read(inputs/'volatility_audit.json')['passed'] and read(inputs/'volatility_audit.json')['input_manifest_sha256']==sha(inputs/'input_manifest.json')
    if phase=='development':assert sha(inputs/'volatility_audit.json')==prereg['volatility_audit_sha256']
    assert read(inputs/'bank_audit.json')['passed'];source,source_sha,prior_sha=verified_source();assert source_sha==prereg['source_v56_manifest_sha256']==manifest['source_v56_manifest_sha256']
    assert prior_sha==prereg['prior_v59_manifest_sha256']==manifest['prior_v59_manifest_sha256']
    market=core.load_market();assert market.manifest==manifest['full_market'];market=restrict_market(market,pd.Timestamp(manifest['observation_cutoff_exclusive']))
    for folder in ['','training','daily','source_daily','models','baseline_models','executed_sources']:(output/folder).mkdir(parents=True,exist_ok=True)
    for f in inputs.iterdir():
        if f.is_file():shutil.copyfile(f,output/f.name)
    dump(output/'registration.json',{'phase':phase,'started_at':datetime.now(timezone.utc).isoformat(),'preregistration':prereg,'protocol_sha256':sha(SOURCE/'protocol.json'),'authorization':auth,'source_sha256':prereg['code_sha256'],'source_v56_manifest_sha256':source_sha,'input_manifest_sha256':sha(inputs/'input_manifest.json')})
    for f in prereg['code_sha256']:
        dest=output/'executed_sources'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((core.ROOT/f).read_bytes())
    panel=pd.read_parquet(output/'activity_panel.parquet');source_panel=pd.read_parquet(output/'activity_panel.parquet');original=pd.read_parquet(output/'financial_panel.parquet');features=pd.read_parquet(output/'features.parquet')
    bank=dict(np.load(output/'path_bank.npz'))
    reg=registry(panel,market);oldreg=registry(original,market);sourcereg=registry(source_panel,market);histories={k:{} for k in VARIANTS};basehist={k:{} for k in ['v56','v55']};predictions={k:[] for k in ROUTES};previous=None
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel.iterrows()):
            dayfile=str(row.date.date())+'.json'
            if row.date!=previous:
                daily={k:fit_day(panel,histories[k],reg,at) for k in VARIANTS};dump(output/'daily'/dayfile,daily);previous=row.date
                oldfile=source/'daily'/dayfile
                source_day={'v56':read(oldfile)['full'],'v55':read(source/'source_daily'/dayfile)} if oldfile.exists() else {'v56':fit_day(source_panel,basehist['v56'],sourcereg,at),'v55':fit_day(original,basehist['v55'],oldreg,at)}
                dump(output/'source_daily'/dayfile,source_day)
            values={k:forecast(bank,market,panel,at,k) for k in VARIANTS};models={k:v[0] for k,v in values.items()};fits={k:v[1] for k,v in values.items()};raws={k:v[2] for k,v in values.items()}
            for kind,model in models.items():
                if model is not None:
                    with gzip.open(output/'models'/(kind+'_'+tag(at)+'.pkl.gz'),'wb') as f:pickle.dump(model,f)
            oldfile=source/'training'/(tag(at)+'.json');baselines={};baseevents={}
            if oldfile.exists():
                old=read(oldfile)
                baselines={'v56':{'raw':old['raws']['full'],'fit':old['fits']['full'],'estimate':old['estimates']['full']},'v55':{k:old['baseline'][k] for k in ['raw','fit','estimate']}}
                baseevents={'v56':old['events']['activity_band'],'v55':old['events']['source_v55']}
                for k in baselines:baselines[k]['origin']={'kind':'frozen_v56','sha256':sha(oldfile)}
            else:
                for k,table,cols in [('v56',source_panel,columns(row.clock)),('v55',original,old55_columns(row.clock))]:
                    model,fit,raw=original_fit(table,at,cols);info=issue(table,at,raw,source_day[k]);event=decide(info,source_day[k]['feedback'],True)
                    baselines[k]={'raw':raw,'fit':fit,'estimate':info,'origin':{'kind':'same_formula_extension'}};baseevents[k]=event
                    if model is not None:
                        with gzip.open(output/'baseline_models'/(k+'_'+tag(at)+'.pkl.gz'),'wb') as f:pickle.dump(model,f)
            estimates={k:issue(panel,at,raws[k],daily[k]) for k in VARIANTS};issued=events(at,row.date,row.clock,estimates,daily,baseevents['v56'],baseevents['v55'])
            dump(output/'training'/(tag(at)+'.json'),{'at':at,'date':row.date,'clock':row.clock,'cutoff':day_cutoff(at),'fits':fits,'raws':raws,'estimates':estimates,'baselines':baselines,'events':issued})
            for k,e in issued.items():predictions[k].append(e)
            for k in VARIANTS:histories[k][at]=raws[k]
            for k in basehist:basehist[k][at]=np.array(baselines[k]['raw']) if baselines[k]['raw'] is not None else None
            if (i+1)%400==0:print('paired path forecasts through',at,flush=True)
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
