"""Daily local calibration and causal continuation policy iteration."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sequential import core, META, meta_state, date_releases, decision, append_day
from path_engine import day_cutoff
from walkforward import label_tables
from risk_ledger import replay
from study import save_replay
from run_intraday import score
from run_breadth import paired_uncertainty

SOURCE=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_labels(path):
    rows=json.loads(Path(path).read_text())
    for row in rows:
        for key in ['decision_at','entry_at','entry_date','expiry_at','expiry_date','exit_at','exit_bar_start',
                    'entry_bar_end','label_available_at','label_window_end']:
            if row.get(key) is not None:
                row[key]=pd.Timestamp(row[key])
        for d in row.get('dividends',[]):
            d['ex']=pd.Timestamp(d['ex']);d['pay']=pd.Timestamp(d['pay'])
    return rows


def expected_times(calendar,date,protocol):
    return [at for c in protocol['analysis_clocks_et']
            if (at:=pd.Timestamp(str(date.date())+' '+c,tz=core.ET).tz_convert('UTC')) < pd.Timestamp(calendar[date]['close_at'])]


def run(output):
    output=Path(output)
    if output.exists():
        raise ValueError('Use a new output directory; preserve all attempts')
    output.mkdir(parents=True);(output/'executed_sources').mkdir();(output/'training').mkdir()
    protocol=json.loads((SOURCE/'protocol.json').read_text())
    prereg=json.loads((SOURCE/'evidence/preregistration.json').read_text())
    assert prereg['protocol_sha256']==sha(SOURCE/'protocol.json')
    source=core.ROOT/protocol['source_run']
    inputs=['features.parquet','labels.json','double_cost_labels.json','results.json','input_manifest.json',
            'verification.json','input_verification.json','numerical_audit.json']
    predictions={}
    for variant in ['base','breadth']:
        predictions[variant]={}
        for algorithm in ['ridge_log','boosted_log','ensemble_log']:
            name=variant+'__'+algorithm+'_predictions.json';inputs.append(name)
            predictions[variant][algorithm]=pd.read_json(source/name,orient='table')
    for name in ['verification','input_verification','numerical_audit']:
        assert json.loads((source/(name+'.json')).read_text())['passed']
    old_manifest=json.loads((source/'input_manifest.json').read_text())
    assert old_manifest['feature_sha256']==sha(source/'features.parquet')
    market=core.load_market();assert old_manifest['market']==market.manifest
    core.dump(output/'input_manifest.json',{'source_run':str(source),'source_sha256':{n:sha(source/n) for n in inputs},
                                          'market':market.manifest})
    files=list(SOURCE.glob('*.py'))+[SOURCE/'protocol.json']
    files += [SOURCE.parent/f'soxl_soxs_14d5pct_{v}'/n for v,n in [
        ('v2','core.py'),('v2','study.py'),('v3','risk_ledger.py'),('v4','walkforward.py'),('v4','daily_study.py'),
        ('v6','path_engine.py'),('v6','run_intraday.py'),('v7','breadth.py'),('v7','run_breadth.py')]]
    core.dump(output/'registration.json',{'started_at':datetime.now(timezone.utc).isoformat(),'preregistration':prereg,
             'source_sha256':{str(p.relative_to(core.ROOT)):sha(p) for p in files}})
    for path in files:
        dest=output/'executed_sources'/path.parent.name/path.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(path.read_bytes())
    features=pd.read_parquet(source/'features.parquet');labels=load_labels(source/'labels.json');tables=label_tables(labels)
    releases=date_releases(features,tables)
    states={}
    for variant,books in predictions.items():
        states[variant]={at:meta_state(row,books['ensemble_log'].loc[at],books['ridge_log'].loc[at],books['boosted_log'].loc[at])
                         for at,row in features.iterrows()}
    policies=protocol['policies'];histories={n:[] for n in policies}
    pools={n:{c:pd.DataFrame(index=pd.DatetimeIndex([],tz='UTC')) for c in protocol['analysis_clocks_et'][1:]} for n in policies}
    registry=[];coverage=[]
    for i,(date,group) in enumerate(features.groupby('date')):
        expected=expected_times(market.calendar,date,protocol)
        complete=group.index.equals(pd.DatetimeIndex(expected))
        coverage.append({'date':date,'expected':expected,'observed':list(group.index),'complete':complete})
        release=releases.get(date,pd.NaT) if complete else pd.NaT
        record={'date':date,'cutoff':date.tz_localize(core.ET),'release':release,'policies':{}}
        for name,settings in policies.items():
            events=[];audits=[]
            for at in expected:
                clock=at.tz_convert(core.ET).strftime('%H:%M:%S');has_later=any(t>at for t in expected)
                if clock==protocol['analysis_clocks_et'][0]:
                    event={'at':at,'action':'WAIT','selected_symbol':None,'ready':False,'cutoff':day_cutoff(at),
                           'reason':'WAIT_FOR_REGULAR_INFORMATION'};audit={'eligible':[],'neighbors':[],'ready':False}
                else:
                    q=states[settings['input']].get(at)
                    event,audit=decision(q,pools[name][clock],at,has_later,settings,protocol)
                events.append(event);audits.append({'at':at,**audit})
            # Append only after every decision on this date was generated.
            # Release timestamps still prevent these labels entering the same day.
            append_day(pools[name],states[settings['input']],events,tables,release,expected)
            histories[name].extend(events);record['policies'][name]={'events':events,'audits':audits}
        core.dump(output/'training'/(str(date.date())+'.json'),record)
        registry.append({'date':date,'cutoff':record['cutoff'],'complete_input_clocks':complete,
            'trained_meta_dates':{n:[len(a['eligible']) for a in record['policies'][n]['audits']] for n in policies},
            'buys':{n:sum(e['action'] in core.ETFS for e in record['policies'][n]['events']) for n in policies}})
        if (i+1)%50==0:print('calibrated through',date.date(),flush=True)
    core.dump(output/'training_registry.json',registry);core.dump(output/'coverage.json',coverage)
    for name in policies:
        core.dump(output/(name+'_predictions.json'),histories[name])
        for clock,pool in pools[name].items():
            pool.to_json(output/(name+'_'+clock.replace(':','')+'_mature_states.json'),orient='table',date_format='iso',double_precision=15)
    start,end=protocol['evaluation_start'],protocol['evaluation_end'];test=features[(features.date>=start)&(features.date<=end)]
    primary=protocol['primary'];actions={e['at']:e['action'] for e in histories[primary]}
    selected=[e for e in histories[primary] if e['at'] in test.index]
    core.dump(output/'decisions_frozen.json',{'frozen_at':datetime.now(timezone.utc).isoformat(),'primary':primary,
        'new_policy_case_results_seen_before_decisions_frozen':False,'previous_versions_same_history_exposed':True,'decisions':selected})
    results={'primary_name':primary,'policies':{},'controls':{}};books={}
    for name,events in histories.items():
        book=replay(market,test,labels,{e['at']:e['action'] for e in events},start,end)
        books[name]=book;save_replay(output,name,book);results['policies'][name]=score(book,protocol['regression_months'])
    old_results=json.loads((source/'results.json').read_text())
    for variant in ['base','breadth']:
        name=variant+'__ensemble_log'
        book=replay(market,test,labels,predictions[variant]['ensemble_log'].action.to_dict(),start,end)
        books[name]=book;save_replay(output,name,book);results['controls'][name]=score(book,protocol['regression_months'])
        assert results['controls'][name]==old_results['routes'][name],'Unchanged reference must reproduce'
    stress=replay(market,test,load_labels(source/'double_cost_labels.json'),actions,start,end,cost_multiple=2)
    save_replay(output,'double_cost',stress);results['double_cost']=score(stress,protocol['regression_months'])
    results['primary']=results['policies'][primary]
    results['paired_increment']=paired_uncertainty(books[primary][0],books['breadth__ensemble_log'][0])
    test_coverage=[x for x in coverage if pd.Timestamp(start)<=x['date']<=pd.Timestamp(end)]
    required=market.sessions[(market.sessions>=start)&(market.sessions<=end)]
    covered=len(test_coverage)==len(required) and all(x['complete'] for x in test_coverage)
    results['acceptance']={'passed':bool(covered and results['primary']['primary_pass']),
        'expected_days':len(required),'observed_days':len(test_coverage),'complete_day_and_premarket_coverage':covered,
        'diagnostic_pass_count':sum(x['primary_pass'] for n,x in results['policies'].items() if n!=primary),
        'new_buy_eligible':False,'live_entry_not_replaced':True,'independent_forward_validation_passed':False,
        'future_nonnegative_month_guarantee':False}
    core.dump(output/'results.json',results);print(json.dumps(results,indent=2),flush=True)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
