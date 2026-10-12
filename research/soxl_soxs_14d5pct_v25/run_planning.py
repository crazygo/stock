"""Freeze as-issued buy/wait values before replaying complete capital cycles."""
import argparse,gzip,hashlib,pickle
from datetime import datetime,timezone
from pathlib import Path
from threadpoolctl import threadpool_limits
from planning import *
from verify_continuation import read,labels
from cycle_ledger import replay
from run_intraday import score
from run_breadth import paired_uncertainty
from study import save_replay
SOURCE=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def json_ready(value):
    if isinstance(value,np.ndarray):return json_ready(value.tolist())
    if isinstance(value,np.generic):return json_ready(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:json_ready(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_ready(v) for v in value]
    return value


def dump(path,value):core.dump(path,json_ready(value))


def read_bank(path):
    bank=read(path)
    for r in bank:
        for k in ['at','date','ready_at','expired_after']:
            if r[k] is not None:r[k]=pd.Timestamp(r[k])
        for k in ['next_at','next_date']:r[k]=[pd.Timestamp(t) if t is not None else None for t in r[k]]
        r['reward']=[float(x) if x is not None else np.nan for x in r['reward']]
    return bank


def authorize_cases(path,p):
    run=Path(path);reg=read(run/'registration.json');r=read(run/'results.json');v=read(run/'verification.json')
    assert reg['phase']=='development' and reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    assert v['passed'] and v['results_sha256']==sha(run/'results.json')
    gate=development_gate(r['primary'],r['coverage']['complete'],p)
    if not gate['passed']:raise ValueError('Development primary failed; new case replay is forbidden')
    return {'run':str(run.resolve()),'results_sha256':sha(run/'results.json'),'gate':gate}


def run(output,case_after=None):
    p=read(SOURCE/'protocol.json');prereg=read(SOURCE/'evidence/preregistration.json')
    assert prereg['protocol_sha256']==sha(SOURCE/'protocol.json')
    for f,h in prereg['code_sha256'].items():assert sha(core.ROOT/f)==h
    authorization=authorize_cases(case_after,p) if case_after else None
    phase='cases' if case_after else 'development';settings=p[phase];start,end=settings['evaluation_start'],settings['evaluation_end']
    cutoff=pd.Timestamp(end,tz=core.ET)+pd.Timedelta(days=1);source=core.ROOT/p['source_runs'][phase]
    delivery=read(core.ROOT/'research/soxl_soxs_14d5pct_v21/evidence/delivery_manifest.json')['cache_runs'][phase]
    needed=['panel.parquet','features.parquet','terminal_receipts.json','predictions.json','results.json','verification.json','input_manifest.json']
    for f in needed:assert sha(source/f)==delivery['outputs_sha256'][f]
    sm=read(source/'input_manifest.json');assert pd.Timestamp(sm['observation_cutoff_exclusive'])==cutoff
    label_source=Path(sm['source_run'])
    for n in ['baseline_labels.json','stress_labels.json']:assert sha(label_source/n)==sm['source_files_sha256'][n]
    assert read(source/'verification.json')['passed']
    original=core.load_market();assert original.manifest==sm['full_market'];market=restrict_market(original,cutoff)
    features=pd.read_parquet(source/'features.parquet');panel=pd.read_parquet(source/'panel.parquet')
    base=labels(label_source/'baseline_labels.json');stress=labels(label_source/'stress_labels.json')
    old=read(source/'predictions.json')['early_base_cal'];assert len(old)==len(features)
    bank=transition_bank(market,panel,base)
    output=Path(output)
    if output.exists():raise ValueError('Use a new output directory; preserve every attempt')
    for sub in ['','models','training','executed_sources']:(output/sub).mkdir(parents=True,exist_ok=True)
    dump(output/'registration.json',{'phase':phase,'started_at':datetime.now(timezone.utc).isoformat(),'preregistration':prereg,
        'protocol_sha256':sha(SOURCE/'protocol.json'),'authorization':authorization,'source_sha256':prereg['code_sha256']})
    for f in prereg['code_sha256']:
        a=core.ROOT/f;dest=output/'executed_sources'/a.parent.name/a.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(a.read_bytes())
    features.to_parquet(output/'features.parquet');panel.to_parquet(output/'panel.parquet');dump(output/'transitions.json',bank)
    dump(output/'input_manifest.json',{'source_run':str(source.resolve()),'source_files_sha256':{f:sha(source/f) for f in needed},
        'label_source':str(label_source.resolve()),'label_files_sha256':{f:sha(label_source/f) for f in ['baseline_labels.json','stress_labels.json']},
        'full_market':original.manifest,'observation_cutoff_exclusive':cutoff})
    predictions={n:[] for n in ROUTES};history={}
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(panel.iterrows()):
            ids,feedback=admitted(bank,day_cutoff(at));model,fit_info=fit(panel,bank,ids);raw=raw_prediction(model,row)
            record={'at':at,'date':row.date,'cutoff':day_cutoff(at),'feedback':feedback,'fit':fit_info,'raw':raw,'calibration':{},'events':{}}
            for mode in ['planning','myopic']:
                cal=calibrate(bank,ids,history,mode);record['calibration'][mode]=cal
                for uncal in [False,True]:
                    name=mode+('_raw' if uncal else '_calibrated');e={'at':at,'date':row.date,**decide(raw,cal,mode,feedback,uncal)}
                    predictions[name].append(e);record['events'][name]=e
            assert pd.Timestamp(old[i]['at'])==at
            predictions['source_early'].append(old[i]);predictions['source_full_cash'].append(old[i])
            predictions['cash'].append({'at':at,'date':row.date,'ready':True,'action':'CASH','allocation':0.})
            history[at]=raw;dump(output/'training'/(str(row.date.date())+'.json'),record)
            with gzip.open(output/'models'/(str(row.date.date())+'.pkl.gz'),'wb') as f:pickle.dump(model,f)
            if (i+1)%75==0:print('trained through',row.date.date(),flush=True)
    predictions['double_cost']=predictions[p['primary']];predictions['same_action_regular_cash']=predictions[p['primary']]
    dump(output/'predictions.json',predictions)
    frozen={n:[e for e in h if start<=str(pd.Timestamp(e['date']).date())<=end] for n,h in predictions.items()}
    dump(output/'decisions_frozen.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'new_v25_capital_outcomes_seen':False,
        'prior_versions_exposed_history':True,'predictions_sha256':sha(output/'predictions.json'),'routes':frozen})
    test=features[features.date.between(start,end)];months=settings.get('months',settings.get('regression_months'));books={}
    result={'phase':phase,'primary_name':p['primary'],'routes':{}}
    for n,h in frozen.items():
        m=2 if n=='double_cost' else 1
        book=replay(market,test,list((stress if m==2 else base).values()),{pd.Timestamp(e['at']):e['action'] for e in h},start,end,
            cost_multiple=m,allocations={pd.Timestamp(e['at']):e['allocation'] for e in h},require_full_cash=n not in ['source_early','same_action_regular_cash'])
        books[n]=book;save_replay(output,n,book);result['routes'][n]=score(book,months)
    required=market.sessions[(market.sessions>=start)&(market.sessions<=end)];covered=list(test.date)==list(required)
    result['coverage']={'complete':covered,'expected_dates':len(required),'observed_dates':len(test)}
    result['primary']=result['routes'][p['primary']]
    result['paired_increments']={n:paired_uncertainty(books[p['primary']][0],books[n][0]) for n in ROUTES if n not in [p['primary'],'cash','double_cost']}
    if phase=='development':
        gate=development_gate(result['primary'],covered,p);result['development_gate']=gate
        result['case_regression']={'executed':False,'eligible_after_verification':gate['passed']}
    else:result['case_acceptance']={'passed':bool(covered and result['primary']['primary_pass'])}
    result.update(goal_complete=False,new_buy_eligible=False,future_nonnegative_month_guarantee=False)
    dump(output/'results.json',result)
    print({n:{'mean':s['primary_average'],'total':s['continuous_return']} for n,s in result['routes'].items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--case-after',type=Path)
    a=p.parse_args();run(a.output,a.case_after)
