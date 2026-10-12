"""Freeze causal allocation choices, then replay a continuous capital account."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from allocation import core,ROUTES,bank,prior_pool,choose,day_cutoff,restrict_market,development_gate
from allocation_ledger import replay
from verify_continuation import labels,read
from run_intraday import score
from run_breadth import paired_uncertainty
from study import save_replay

SOURCE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authorize_cases(run,p):
    r=Path(run);reg=read(r/'registration.json');verification=read(r/'verification.json');result=read(r/'results.json')
    assert reg['phase']=='development' and reg['protocol_sha256']==sha(SOURCE/'protocol.json')
    assert verification['passed'] and verification['results_sha256']==sha(r/'results.json')
    gate=development_gate(result['primary'],result['coverage']['complete'],p)
    if not gate['passed']:raise ValueError('Pre-case development gate failed; case replay is forbidden')
    return {'run':str(r.resolve()),'results_sha256':sha(r/'results.json'),'gate':gate}


def run(output,case_after=None,case_source=None):
    p=read(SOURCE/'protocol.json');prereg=read(SOURCE/'evidence/preregistration.json')
    assert prereg['protocol_sha256']==sha(SOURCE/'protocol.json')
    authorization=authorize_cases(case_after,p) if case_after else None
    phase='cases' if case_after else 'development';settings=p[phase];start,end=settings['evaluation_start'],settings['evaluation_end']
    cutoff=pd.Timestamp(end,tz=core.ET)+pd.Timedelta(days=1)
    if phase=='cases' and not case_source:raise ValueError('Verified case-horizon paired forecasts are required after development passes')
    source=Path(case_source) if phase=='cases' else core.ROOT/p['source_run']
    src_manifest=read(source/'input_manifest.json');src_verification=read(source/'verification.json')
    assert src_verification['passed'] and src_verification['results_sha256']==sha(source/'results.json')
    assert pd.Timestamp(src_manifest['observation_cutoff_exclusive'])==cutoff
    src_registration=read(source/'registration.json')
    assert src_registration['protocol_sha256']==sha(SOURCE.parent/'soxl_soxs_14d5pct_v14/protocol.json')
    # Bind all reused market, predictions and model files to the original audited delivery.
    delivery=read(SOURCE.parent/'soxl_soxs_14d5pct_v14/evidence/delivery_manifest.json') if phase=='development' else read(source/'delivery_manifest.json')
    needed=['features.parquet','panel.parquet','predictions.json','baseline_labels.json','stress_labels.json',
            'results.json','verification.json','input_manifest.json','registration.json','training_registry.json']
    for f in needed:assert sha(source/f)==delivery['cache_outputs_sha256'][f]
    original=core.load_market();assert original.manifest==src_manifest['full_market'];market=restrict_market(original,cutoff)
    features=pd.read_parquet(source/'features.parquet');panel=pd.read_parquet(source/'panel.parquet')
    assert (features.index<cutoff).all() and features.clock.eq('09:25:30').all()
    events=read(source/'predictions.json')['regret_regime'];frame=bank(panel,events)
    output=Path(output)
    if output.exists():raise ValueError('Use a new output directory; preserve every attempt')
    output.mkdir(parents=True);(output/'training').mkdir();(output/'executed_sources').mkdir()
    files=[*SOURCE.glob('*.py'),SOURCE/'protocol.json']
    files += [SOURCE.parent/f'soxl_soxs_14d5pct_{v}'/n for v,n in [('v2','core.py'),('v2','study.py'),('v3','risk_ledger.py'),
        ('v3','risk.py'),('v4','daily_study.py'),('v6','run_intraday.py'),('v7','run_breadth.py'),('v11','uncertainty.py'),('v14','paired_entry.py')]]
    core.dump(output/'registration.json',{'phase':phase,'started_at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha(SOURCE/'protocol.json'),'preregistration':prereg,'authorization':authorization,
        'source_sha256':{str(f.relative_to(core.ROOT)):sha(f) for f in files}})
    for f in files:
        dest=output/'executed_sources'/f.parent.name/f.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(f.read_bytes())
    frame.to_parquet(output/'distribution_bank.parquet',compression='zstd',compression_level=7)
    features.to_parquet(output/'features.parquet',compression='zstd',compression_level=7)
    core.dump(output/'input_manifest.json',{'source_run':str(source.resolve()),'source_files_sha256':{f:sha(source/f) for f in needed},
        'source_delivery_sha256':sha(SOURCE.parent/'soxl_soxs_14d5pct_v14/evidence/delivery_manifest.json') if phase=='development' else sha(source/'delivery_manifest.json'),
        'full_market':original.manifest,'observation_cutoff_exclusive':cutoff,'bank_sha256':sha(output/'distribution_bank.parquet'),
        'features_sha256':sha(output/'features.parquet'),'source_model_retraining':'Reuse immutable verified daily v14 fits; allocation distribution rebuilt daily as of cutoff'})
    predictions={n:[] for n in ROUTES};summary=[]
    for e in events:
        at=pd.Timestamp(e['at']);date=pd.Timestamp(e['date']);cutoff_day=day_cutoff(at)
        record={'at':at,'date':date,'cutoff':cutoff_day,'source_ready':e['ready'],'source_top':e['top_symbol'],'routes':{}}
        for name,(mode,gamma,fractions) in ROUTES.items():
            if e['ready']:
                current=frame.loc[(at,e['top_symbol'])];pool=prior_pool(frame,cutoff_day,e['top_symbol'])
                picked=choose(pool,current,mode,gamma,fractions)
            else:picked={'ready':False,'members':[],'dates':[],'allocation':0.,'reason':'source_unready','distribution':None,'utility':None}
            action=e['top_symbol'] if picked['allocation']>0 else 'CASH'
            event={'at':at,'date':date,'source_top':e['top_symbol'],'action':action,'allocation':picked['allocation'],
                   'ready':picked['ready'],'reason':picked['reason']}
            record['routes'][name]={'event':event,'fit':picked};predictions[name].append(event)
        core.dump(output/'training'/(str(date.date())+'.json'),record)
        summary.append({'date':date,'cutoff':cutoff_day,'routes':{n:{'history_dates':len(v['fit']['members']),
            'effective_dates':v['fit']['distribution']['effective_dates'] if v['fit']['distribution'] else None,
            'action':v['event']['action'],'allocation':v['event']['allocation'],'reason':v['event']['reason']} for n,v in record['routes'].items()}})
    for name,fraction,direction_only in [('source_full',1.,False),('source_half',.5,False),('source_direction',1.,True)]:
        predictions[name]=[{'at':pd.Timestamp(e['at']),'date':pd.Timestamp(e['date']),'ready':e['ready'],
            'action':e['top_symbol'] if e['ready'] and (direction_only or e['action']!='CASH') else 'CASH',
            'allocation':fraction if e['ready'] and (direction_only or e['action']!='CASH') else 0.} for e in events]
    predictions['cash']=[{'at':pd.Timestamp(e['at']),'date':pd.Timestamp(e['date']),'ready':True,'action':'CASH','allocation':0.} for e in events]
    predictions['double_cost']=predictions[p['primary']]
    core.dump(output/'predictions.json',predictions);core.dump(output/'training_registry.json',summary)
    test=features[(features.date>=start)&(features.date<=end)];frozen={}
    for name,history in predictions.items():
        frozen[name]=[e for e in history if e['at'] in test.index]
    core.dump(output/'decisions_frozen.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'phase':phase,
        'new_v15_capital_outcomes_seen':False,'development_previously_exposed':True,'predictions_sha256':sha(output/'predictions.json'),'routes':frozen})
    base=labels(source/'baseline_labels.json');stress=labels(source/'stress_labels.json');books={};results={'phase':phase,'primary_name':p['primary'],'routes':{}}
    months=settings.get('months',settings.get('regression_months'))
    for name,history in frozen.items():
        actions={e['at']:e['action'] for e in history};allocations={e['at']:e['allocation'] for e in history};multiple=2 if name=='double_cost' else 1
        b=replay(market,test,list((stress if multiple==2 else base).values()),actions,start,end,cost_multiple=multiple,allocations=allocations)
        books[name]=b;save_replay(output,name,b);results['routes'][name]=score(b,months)
    required=market.sessions[(market.sessions>=start)&(market.sessions<=end)];covered=pd.DatetimeIndex(test.date).equals(required)
    results['coverage']={'complete':covered,'expected_days':len(required),'observed_days':len(test)};results['primary']=results['routes'][p['primary']]
    results['paired_increments']={n:paired_uncertainty(books[p['primary']][0],books[n][0]) for n in predictions if n not in [p['primary'],'cash','double_cost']}
    if phase=='development':
        gate=development_gate(results['primary'],covered,p);results['development_gate']=gate
        results['case_regression']={'executed':False,'eligible_after_verification':gate['passed']}
    else:results['case_acceptance']={'passed':bool(covered and results['primary']['primary_pass'])}
    results.update(goal_complete=False,new_buy_eligible=False,future_nonnegative_month_guarantee=False)
    core.dump(output/'results.json',results)
    print(json.dumps({n:{'mean':v['primary_average'],'return':v['continuous_return'],'pass':v['primary_pass']} for n,v in results['routes'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--case-after',type=Path);p.add_argument('--case-source',type=Path)
    a=p.parse_args();run(a.output,a.case_after,a.case_source)
