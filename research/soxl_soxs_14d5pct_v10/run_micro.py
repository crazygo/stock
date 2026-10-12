"""Daily learning with early intraday management and matched exit controls."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import platform
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from micro_manager import (core, ENTRY_COLUMNS, MANAGER_COLUMNS, entry_state, label_key, alive,
    context, prepare_reviews, apply_exit, simple_exit, joint_releases, eligible_dates,
    fit, calibration, predict, choose_entry, day_cutoff)
from run_continuation import training_entries, selected_history
from risk_ledger import replay
from run_intraday import score
from study import save_replay
from run_breadth import paired_uncertainty

SOURCE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_labels(path):
    result = {}
    for r in json.loads(Path(path).read_text()):
        for name in ['decision_at','entry_at','entry_date','entry_bar_end','expiry_at','expiry_date',
                     'exit_at','exit_bar_start','label_available_at','manager_decision_at']:
            if r.get(name) is not None: r[name] = pd.Timestamp(r[name])
        for d in r.get('dividends',[]):
            d['ex'] = pd.Timestamp(d['ex']); d['pay'] = pd.Timestamp(d['pay'])
        result[label_key(r['decision_at'],r['symbol'])] = r
    return result


def run(output):
    output = Path(output)
    if output.exists(): raise ValueError('Use a new output directory, preserve all attempts')
    for sub in ['','models','training','executed_sources']: (output/sub).mkdir(parents=True,exist_ok=True)
    protocol = json.loads((SOURCE/'protocol.json').read_text()); prereg = json.loads((SOURCE/'evidence/preregistration.json').read_text())
    assert sha(SOURCE/'protocol.json') == prereg['protocol_sha256']
    previous = core.ROOT/protocol['premarket_reference']; old = json.loads((previous/'input_manifest.json').read_text())
    assert json.loads((previous/'verification.json').read_text())['passed']
    market = core.load_market(); assert market.manifest == old['market']
    source = core.ROOT/protocol['source_features']; assert sha(source) == old['source_feature_sha256']
    features = pd.read_parquet(source); features = features[features.clock.eq(protocol['entry_clock_et'])]
    features.to_parquet(output/'features.parquet',compression='zstd',compression_level=7)
    inputs = ['baseline_labels.json','stress_baseline_labels.json','managed_labels.json','managed_entry_predictions.json','results.json','verification.json']
    core.dump(output/'input_manifest.json',{'market':market.manifest,'previous_run':str(previous),
        'previous_files_sha256':{n:sha(previous/n) for n in inputs},'source_features':str(source),
        'source_features_sha256':sha(source),'feature_sha256':sha(output/'features.parquet'),
        'entry_columns':ENTRY_COLUMNS,'manager_columns':MANAGER_COLUMNS})
    files = list(SOURCE.glob('*.py'))+[SOURCE/'protocol.json']
    files += [SOURCE.parent/f'soxl_soxs_14d5pct_{v}'/n for v,n in [
        ('v2','core.py'),('v2','study.py'),('v3','risk.py'),('v3','risk_ledger.py'),('v4','walkforward.py'),
        ('v4','daily_study.py'),('v6','path_engine.py'),('v6','run_intraday.py'),('v7','breadth.py'),
        ('v7','run_breadth.py'),('v9','continuation.py'),('v9','run_continuation.py')]]
    core.dump(output/'registration.json',{'started_at':datetime.now(timezone.utc).isoformat(),'preregistration':prereg,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
        'source_sha256':{str(p.relative_to(core.ROOT)):sha(p) for p in files}})
    for p in files:
        dest = output/'executed_sources'/p.parent.name/p.name; dest.parent.mkdir(exist_ok=True);dest.write_bytes(p.read_bytes())
    baseline = read_labels(previous/'baseline_labels.json'); stress_base = read_labels(previous/'stress_baseline_labels.json')
    releases = joint_releases(features,baseline); prepared = context(market)
    reviews = prepare_reviews(market,features,baseline,prepared,protocol)
    stress_reviews = prepare_reviews(market,features,stress_base,prepared,protocol,2)
    core.dump(output/'review_states.json',reviews);core.dump(output/'stress_review_states.json',stress_reviews)
    core.dump(output/'baseline_labels.json',list(baseline.values()));core.dump(output/'stress_baseline_labels.json',list(stress_base.values()))
    print('prepared',len(features),'dates,',len(reviews),'normal reviews,',len(stress_reviews),'stress reviews',flush=True)
    managed = dict(baseline); stressed = dict(stress_base); rule_labels = dict(baseline)
    by_date = {name:{} for name in ['base','stress']}
    for name,rows in [('base',reviews),('stress',stress_reviews)]:
        for r in rows: by_date[name].setdefault(r['at'].tz_convert(core.ET).tz_localize(None).normalize(),[]).append(r)
    history = []; manager_history = []; all_events = []; registry = []
    with threadpool_limits(limits=1):
        for i,(at,row) in enumerate(features.iterrows()):
            cutoff = day_cutoff(at); dates = eligible_dates(releases,cutoff,protocol); date_set = set(dates)
            entry_model,entry_info = fit(training_entries(features,managed,dates),ENTRY_COLUMNS,protocol)
            training_reviews = [r for r in reviews if r['entry_date'] in date_set and r['state'] is not None and np.isfinite(r['target'])]
            manager_model,manager_info = fit(training_reviews,MANAGER_COLUMNS,protocol)
            entry_cal = calibration(selected_history(history,managed),date_set,protocol)
            manager_cal = calibration(manager_history,date_set,protocol)
            record = {'date':row.date,'at':at,'cutoff':cutoff,'eligible_dates':dates,
                'latest_label_release':max((releases[d] for d in dates),default=None),
                'fits':{'entry':entry_info,'manager':manager_info},'calibrations':{'entry':entry_cal,'manager':manager_cal},'manager':[]}
            with gzip.open(output/'models'/(str(row.date.date())+'.pkl.gz'),'wb') as f:
                pickle.dump({'entry':entry_model,'manager':manager_model},f)
            raw = {s:predict(entry_model,entry_state(row,s),ENTRY_COLUMNS) for s in core.ETFS} if entry_model else {}
            event = {'at':at,'entry_date':row.date,**choose_entry(raw,entry_cal)}
            history.append(event);record['entry'] = event
            # Models and calibrations stay fixed through this entire review day.
            for name,labels in [('base',managed),('stress',stressed)]:
                day_reviews = sorted(by_date[name].get(row.date,[]),key=lambda r:(r['at'],r['key']))
                for r in day_reviews:
                    if not alive(labels[r['key']],r['at']): continue
                    raw = predict(manager_model,r['state'],MANAGER_COLUMNS) if manager_model is not None and r['state'] is not None else None
                    adjusted = raw+manager_cal['correction'] if raw is not None and manager_cal['correction'] is not None else None
                    ready = raw is not None and manager_cal['ready']
                    event = {'id':r['id'],'key':r['key'],'cost_path':name,'at':r['at'],'entry_date':r['entry_date'],
                        'raw':raw,'adjusted':adjusted,'ready':ready,'action':'EXIT' if ready and adjusted<=0 else 'HOLD'}
                    if event['action']=='EXIT':
                        event['execution'] = ('unavailable' if r['sale'] is None else
                            'cancelled_by_barrier_during_latency' if r['sale']['barrier_during_latency'] else 'scheduled_market_exit')
                        labels[r['key']] = apply_exit(labels[r['key']],r['sale'],r['at'])
                    record['manager'].append(event);all_events.append(event)
                    if name=='base' and raw is not None:manager_history.append({**event,'target':r['target']})
            # Independent, fixed structural control, never used for training/selection.
            for r in sorted(by_date['base'].get(row.date,[]),key=lambda r:(r['at'],r['key'])):
                if alive(rule_labels[r['key']],r['at']) and simple_exit(r['state']):
                    rule_labels[r['key']] = apply_exit(rule_labels[r['key']],r['sale'],r['at'],'simple_trend_exit')
            core.dump(output/'training'/(str(row.date.date())+'.json'),record)
            registry.append({'date':row.date,'cutoff':cutoff,'latest_label_release':record['latest_label_release'],
                'training_dates':{n:len(v['dates']) for n,v in record['fits'].items()},
                'calibration_dates':{n:len(v['dates']) for n,v in record['calibrations'].items()},
                'entry_action':record['entry']['action'],'manager_reviews':len(record['manager'])})
            if (i+1)%25==0:print('trained through',row.date.date(),flush=True)
    for name,labels in [('managed',managed),('stress_managed',stressed),('simple_trend',rule_labels)]:
        core.dump(output/(name+'_labels.json'),list(labels.values()))
    core.dump(output/'training_registry.json',registry);core.dump(output/'entry_predictions.json',history)
    core.dump(output/'manager_predictions.json',all_events)
    start,end = protocol['evaluation_start'],protocol['evaluation_end'];test = features[(features.date>=start)&(features.date<=end)]
    actions = {r['at']:r['action'] for r in history}
    old_labels = read_labels(previous/'managed_labels.json')
    old_actions = {pd.Timestamp(r['at']):r['action'] for r in json.loads((previous/'managed_entry_predictions.json').read_text())}
    core.dump(output/'decisions_frozen.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),
        'new_strategy_case_results_seen_before_decisions_frozen':False,'previous_versions_same_history_exposed':True,
        'entry':[r for r in history if r['at'] in test.index],'manager_sha256':sha(output/'manager_predictions.json')})
    primary = protocol['primary'];controls = protocol['controls'];books = {};results = {'primary_name':primary,'routes':{}}
    for name,labels,selected,multiple in [(primary,managed,actions,1),(controls[0],baseline,actions,1),
        (controls[1],old_labels,actions,1),(controls[2],rule_labels,actions,1),
        (controls[3],old_labels,old_actions,1),('double_cost',stressed,actions,2)]:
        book = replay(market,test,list(labels.values()),selected,start,end,cost_multiple=multiple)
        save_replay(output,name,book);books[name] = book;stats = score(book,protocol['regression_months'])
        done = book[1]
        if len(done):done = done[pd.to_datetime(done.exit_at,utc=True)<pd.Timestamp(end,tz=core.ET)+pd.Timedelta(days=1)]
        stats['intraday_value_exits'] = int(done.reason.eq('intraday_value_exit').sum()) if len(done) else 0
        stats['simple_trend_exits'] = int(done.reason.eq('simple_trend_exit').sum()) if len(done) else 0
        results['routes'][name] = stats
    old_results = json.loads((previous/'results.json').read_text())['primary']
    for k,v in old_results.items():
        if k!='continuation_exits': assert results['routes'][controls[3]][k]==v,(k,results['routes'][controls[3]][k],v)
    results['primary'] = results['routes'][primary]
    results['paired_increment'] = {n:paired_uncertainty(books[primary][0],books[n][0]) for n in controls[:3]}
    required = market.sessions[(market.sessions>=start)&(market.sessions<=end)]
    covered = pd.DatetimeIndex(test.date).equals(required)
    results['acceptance'] = {'passed':bool(covered and results['primary']['primary_pass']),
        'expected_premarket_days':len(required),'observed_premarket_days':len(test),'complete_premarket_coverage':covered,
        'new_buy_eligible':False,'live_entry_not_replaced':True,'independent_forward_validation_passed':False,
        'future_nonnegative_month_guarantee':False}
    core.dump(output/'results.json',results);print(json.dumps(results,indent=2),flush=True)
    return results


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
