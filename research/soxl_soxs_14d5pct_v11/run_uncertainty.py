"""Pre-case screening, with a mandatory verified development gate before case replay."""
import argparse
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from uncertainty import (core,MANAGER_COLUMNS,context,prepare_reviews,apply_exit,alive,label_key,
    baseline_labels,joint_releases,restrict_market,build_bank,pool_for_day,decision,development_gate)
from path_engine import day_cutoff
from risk_ledger import replay
from run_intraday import score
from study import save_replay
from run_breadth import paired_uncertainty

SOURCE = Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authorize_cases(development,protocol):
    development = Path(development)
    registration = json.loads((development/'registration.json').read_text())
    assert registration['phase']=='development'
    assert registration['protocol_sha256']==sha(SOURCE/'protocol.json')
    verification = json.loads((development/'verification.json').read_text())
    assert verification['passed'] and verification['results_sha256']==sha(development/'results.json')
    results = json.loads((development/'results.json').read_text())
    gate = development_gate(results['primary'],results['coverage']['complete'],protocol)
    if not gate['passed']:raise ValueError('Pre-case development gate failed; case replay is forbidden')
    return {'development_run':str(development.resolve()),'results_sha256':sha(development/'results.json'),
            'verification_sha256':sha(development/'verification.json'),'gate':gate}


def run(output,case_after=None):
    protocol = json.loads((SOURCE/'protocol.json').read_text());prereg = json.loads((SOURCE/'evidence/preregistration.json').read_text())
    assert prereg['protocol_sha256']==sha(SOURCE/'protocol.json')
    authorization = authorize_cases(case_after,protocol) if case_after else None
    phase = 'cases' if case_after else 'development';settings = protocol[phase]
    start,end = settings['evaluation_start'],settings['evaluation_end']
    cutoff = pd.Timestamp(end,tz=core.ET)+pd.Timedelta(days=1)
    if phase=='development':assert cutoff==pd.Timestamp(settings['observation_cutoff_exclusive_et'],tz=core.ET)
    output = Path(output)
    if output.exists():raise ValueError('Output must be new; preserve every attempt')
    for sub in ['','training','executed_sources']:(output/sub).mkdir(parents=True,exist_ok=True)
    source = core.ROOT/protocol['source_run'];source_manifest = json.loads((source/'input_manifest.json').read_text())
    assert json.loads((source/'verification.json').read_text())['passed']
    full_market = core.load_market();assert full_market.manifest==source_manifest['market']
    market = restrict_market(full_market,cutoff)
    features = pd.read_parquet(source/'features.parquet');features = features[features.index<cutoff]
    features.to_parquet(output/'features.parquet',compression='zstd',compression_level=7)
    files = list(SOURCE.glob('*.py'))+[SOURCE/'protocol.json']
    files += [SOURCE.parent/f'soxl_soxs_14d5pct_{v}'/name for v,name in [
        ('v2','core.py'),('v2','study.py'),('v3','risk.py'),('v3','risk_ledger.py'),('v4','walkforward.py'),
        ('v4','daily_study.py'),('v6','path_engine.py'),('v6','run_intraday.py'),('v7','breadth.py'),
        ('v7','run_breadth.py'),('v9','continuation.py'),('v10','micro_manager.py')]]
    core.dump(output/'registration.json',{'started_at':datetime.now(timezone.utc).isoformat(),'phase':phase,
        'protocol_sha256':sha(SOURCE/'protocol.json'),'preregistration':prereg,'case_authorization':authorization,
        'source_sha256':{str(p.relative_to(core.ROOT)):sha(p) for p in files}})
    for p in files:
        dest = output/'executed_sources'/p.parent.name/p.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes())
    baseline = baseline_labels(market,features);stress_base = baseline_labels(market,features,2)
    releases = joint_releases(features,baseline);prepared = context(market)
    reviews = prepare_reviews(market,features,baseline,prepared,protocol)
    stress_reviews = prepare_reviews(market,features,stress_base,prepared,protocol,2)
    by_date = {name:{} for name in ['base','stress']}
    for name,rows in [('base',reviews),('stress',stress_reviews)]:
        for r in rows:by_date[name].setdefault(r['at'].tz_convert(core.ET).tz_localize(None).normalize(),[]).append(r)
    model_hashes = {};source_training = []
    # Evaluate each frozen historical model only on states available on its own date.
    with threadpool_limits(limits=1):
        for date in features.date:
            path = source/'models'/(str(date.date())+'.pkl.gz');assert date.tz_localize(core.ET)<cutoff
            info = json.loads((source/'training'/(str(date.date())+'.json')).read_text())
            assert pd.Timestamp(info['cutoff'])==date.tz_localize(core.ET).tz_convert('UTC')
            if info['latest_label_release'] is not None:assert pd.Timestamp(info['latest_label_release'])<pd.Timestamp(info['cutoff'])
            source_training.append({'date':date,'cutoff':info['cutoff'],'training_dates':info['fits']['manager']['dates'],
                'latest_label_release':info['latest_label_release'],'registration_sha256':sha(source/'training'/(str(date.date())+'.json'))})
            model_hashes[path.name]=sha(path)
            with gzip.open(path,'rb') as f:model = pickle.load(f)['manager']
            for name in by_date:
                group = by_date[name].get(date,[])
                for r in group:r['raw']=None
                good = [r for r in group if r['state'] is not None]
                if model is not None and good:
                    values = model.predict(np.array([[r['state'][c] for c in MANAGER_COLUMNS] for r in good]))
                    for r,value in zip(good,values):r['raw']=float(value)
    core.dump(output/'source_training.json',source_training)
    core.dump(output/'review_forecasts.json',reviews);core.dump(output/'stress_review_forecasts.json',stress_reviews)
    core.dump(output/'baseline_labels.json',list(baseline.values()));core.dump(output/'stress_baseline_labels.json',list(stress_base.values()))
    manifest = {'source_run':str(source),'source_files_sha256':{n:sha(source/n) for n in [
        'features.parquet','entry_predictions.json','manager_predictions.json','input_manifest.json','verification.json']},
        'source_model_sha256':model_hashes,'full_source_market':full_market.manifest,'observation_cutoff_exclusive':cutoff,
        'restricted_market':{s:{'rows':len(f),'last_bar_end':f.end_at.max(),'last_available_at':f.available_at.max(),
        'daily_last':market.daily[s].index.max()} for s,f in market.bars.items()},'feature_sha256':sha(output/'features.parquet')}
    core.dump(output/'input_manifest.json',manifest)
    bank = build_bank(reviews,releases,protocol['local_columns']);bank.to_parquet(output/'calibration_bank.parquet',compression='zstd',compression_level=7)
    policies = {protocol['primary']:dict(baseline),protocol['diagnostic']:dict(baseline),'double_cost':dict(stress_base)}
    events = [];registry = []
    print('prepared',len(features),'dates;',len(reviews),'normal forecasts;',len(stress_reviews),'stress forecasts',flush=True)
    for i,(at,row) in enumerate(features.iterrows()):
        pool,local = pool_for_day(bank,day_cutoff(at),protocol)
        record = {'date':row.date,'pool':pool,'events':[]}
        for name,labels in policies.items():
            group = by_date['stress' if name=='double_cost' else 'base'].get(row.date,[])
            method = protocol['primary'] if name=='double_cost' else name
            for r in sorted(group,key=lambda r:(r['at'],r['key'])):
                if not alive(labels[r['key']],r['at']):continue
                event = {'id':r['id'],'key':r['key'],'policy':name,'at':r['at'],'entry_date':r['entry_date'],
                         **decision(r['raw'],r['state'],pool,local,protocol,method)}
                if event['action']=='EXIT':
                    sale = r['sale'];event['execution']='unavailable' if sale is None else 'cancelled_by_barrier_during_latency' if sale['barrier_during_latency'] else 'scheduled_market_exit'
                    labels[r['key']] = apply_exit(labels[r['key']],sale,r['at'],'uncertainty_exit')
                events.append(event);record['events'].append(event)
        core.dump(output/'training'/(str(row.date.date())+'.json'),record)
        registry.append({'date':row.date,'calibration_parents':len(pool['quantile_parents']),'local_parents':len(pool['eligible_parents']),
            'residual_upper':pool['residual_upper'],'events':len(record['events'])})
        if (i+1)%50==0:print('bounded through',row.date.date(),flush=True)
    # Reconstruct the old point policy using ONLY its recorded intents before this cutoff.
    original = dict(baseline);lookup = {r['id']:r for r in reviews}
    old_events = [e for e in json.loads((source/'manager_predictions.json').read_text()) if e['cost_path']=='base' and pd.Timestamp(e['at'])<cutoff]
    for e in old_events:
        r = lookup[e['id']]
        if e['raw'] is not None:assert np.isclose(e['raw'],r['raw'],rtol=1e-11,atol=1e-13)
        if e['action']=='EXIT':original[r['key']]=apply_exit(original[r['key']],r['sale'],r['at'],'intraday_value_exit')
    for name,labels in {**policies,'original_v10_point':original}.items():core.dump(output/(name+'_labels.json'),list(labels.values()))
    core.dump(output/'manager_predictions.json',events);core.dump(output/'training_registry.json',registry)
    entry = [e for e in json.loads((source/'entry_predictions.json').read_text()) if pd.Timestamp(e['at'])<cutoff]
    actions = {pd.Timestamp(e['at']):e['action'] for e in entry};core.dump(output/'entry_predictions.json',entry)
    test = features[(features.date>=start)&(features.date<=end)]
    core.dump(output/'decisions_frozen.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'phase':phase,
        'new_policy_results_seen_before_decisions_frozen':False,'manager_sha256':sha(output/'manager_predictions.json'),
        'entry':[e for e in entry if pd.Timestamp(e['at']) in test.index]})
    months = settings['months'] if phase=='development' else settings['regression_months'];results = {'phase':phase,'primary_name':protocol['primary'],'routes':{}};books = {}
    definitions = [*[(n,labels,2 if n=='double_cost' else 1) for n,labels in policies.items()],
                   ('fixed_contract',baseline,1),('original_v10_point',original,1),('double_cost_fixed',stress_base,2)]
    for name,labels,multiple in definitions:
        book = replay(market,test,list(labels.values()),actions,start,end,cost_multiple=multiple)
        save_replay(output,name,book);books[name]=book;stats = score(book,months)
        done = book[1]
        if len(done):done=done[pd.to_datetime(done.exit_at,utc=True)<cutoff]
        stats['uncertainty_exits']=int(done.reason.eq('uncertainty_exit').sum()) if len(done) else 0
        results['routes'][name]=stats
    primary = protocol['primary'];results['primary']=results['routes'][primary]
    required = market.sessions[(market.sessions>=start)&(market.sessions<=end)]
    covered = pd.DatetimeIndex(test.date).equals(required)
    results['coverage']={'complete':covered,'expected_days':len(required),'observed_days':len(test)}
    results['paired_increment']=paired_uncertainty(books[primary][0],books['fixed_contract'][0])
    if phase=='development':
        gate=development_gate(results['primary'],covered,protocol)
        results['development_gate']=gate;results['case_regression']={'executed':False,'eligible_after_verification':gate['passed'],
            'reason':'await_verified_development' if gate['passed'] else 'development_gate_failed_no_case_replay'}
    else:results['case_acceptance']={'passed':bool(covered and results['primary']['primary_pass']),'development_authorization':authorization}
    results['goal_complete']=False;results['new_buy_eligible']=False;results['future_nonnegative_month_guarantee']=False
    core.dump(output/'results.json',results);print(json.dumps(results,indent=2),flush=True)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case-after',type=Path,help='Only a verified, passing development run authorizes case months')
    args=parser.parse_args();run(args.output,args.case_after)
