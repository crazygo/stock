"""Raw-prefix, latency, prequential fitting and capital checks, separate from the runner."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from micro_manager import core, ENTRY_COLUMNS, MANAGER_COLUMNS, MICRO_FIELDS, DIRECTIONAL
from verify_continuation import (near, read, key, labels, independent_state, calibration_check,
                                 raw_exit, audit_ledger)


def verify_states(market, features, baseline, records, multiple):
    by_day = {s:{d:f for d,f in market.bars[s].groupby('date')} for s in core.SYMBOLS}
    result = {}; count = 0; latency = 0
    for r in records:
        r['at'] = pd.Timestamp(r['at']);r['entry_date'] = pd.Timestamp(r['entry_date']);at = r['at']
        day = at.tz_convert(core.ET).tz_localize(None).normalize();label = baseline[r['key']]
        assert label['entry_at'] < at < label['expiry_at'] and (label['exit_at'] is None or label['exit_at'] > at)
        pre = features[features.date == day].iloc[0]
        state = independent_state(pre,label['symbol']);opening = pd.Timestamp(market.calendar[day]['open_at'])
        regular = at >= opening;state['regular_ready'] = float(regular)
        seen = by_day[label['symbol']][day];seen = seen[(seen.available_at<=at)&(seen.end_at<=at)]
        assert len(seen) and at-seen.end_at.iloc[-1]<=pd.Timedelta(minutes=6)
        price = float(seen.econ_close.iloc[-1]);entry = label['entry_fill_economic']
        prior = market.daily[label['symbol']]
        prior = prior[(prior.index>=label['entry_date'])&(prior.index<day)]
        peak = max(entry,prior.high.max() if len(prior) else 0.)
        for symbol,prefix in [(label['symbol'],'own'),('SOXX','sector'),('QQQ','qqq')]:
            if not regular:
                state.update({'micro_'+prefix+'_'+k:0. for k in MICRO_FIELDS});continue
            f = by_day[symbol][day];f = f[f.session_type.eq('regular')&(f.available_at<=at)&(f.end_at<=at)]
            expected = pd.date_range(opening,at.floor('5min'),freq='5min',inclusive='right')
            assert f.index.equals(expected) and len(f)
            if prefix=='own':peak = max(peak,float(f.econ_high.max()))
            close = float(f.econ_close.iloc[-1]);start = float(f.econ_open.iloc[0]);n = len(f)
            high,low = float(f.econ_high.max()),float(f.econ_low.min())
            volume = float(f.econ_volume.sum());vwap = float(((f.econ_high+f.econ_low+f.econ_close)/3*f.econ_volume).sum()/volume)
            changes = np.diff(np.r_[start,f.econ_close.to_numpy()]);distance = float(np.abs(changes).sum())
            old = market.daily[symbol].reindex(market.sessions[market.sessions<day][-20:])
            raw = {'session_return':close/start-1,'vwap_distance':close/vwap-1,
                   'last5':close/float(f.econ_open.iloc[-1])-1,'recent15':close/float(f.econ_open.iloc[max(0,n-3)])-1,
                   'range_position':(close-low)/(high-low) if high>low else .5,
                   'efficiency':(close-start)/distance if distance else 0.,'observed_range':(high-low)/start,
                   'bars':n/78.,'relative_volume':volume/float(old.volume.mean())}
            for k,v in raw.items():
                if label['symbol']=='SOXS' and prefix!='own':
                    if k in DIRECTIONAL:v = -v
                    elif k=='range_position':v = 1-v
                state['micro_'+prefix+'_'+k] = v
        age = (day-label['entry_date']).days
        state.update(age_calendar=age,remaining_calendar=13-age,mark_return=price/entry-1,
            target_distance=label['limit_economic']/price-1,stop_distance=entry*.94/price-1,
            observed_peak_drawdown=price/peak-1)
        assert r['state'] is not None
        for col,value in state.items():near(r['state'][col],value)
        own = by_day[label['symbol']][day]
        future = own[own.session_type.eq('regular')&(own.start_at>at+pd.Timedelta(seconds=30))]
        assert len(future) and future.start_at.iloc[0]-at<=pd.Timedelta(minutes=6)
        first = future.iloc[0];sale = r['sale'];assert pd.Timestamp(sale['scheduled_open'])==first.start_at
        earlier = label['exit_at'] is not None and label['exit_bar_start'] < first.start_at
        assert sale['barrier_during_latency']==earlier
        if earlier:
            near(r['target'],0.);near(sale['sale_net'],label['exit_fill_economic']*(1-.0005*multiple));latency+=1
        else:
            target = math.ceil(label['limit_economic']/first.share_factor*100-1e-10)/100
            immediate = min(first.open,target)*first.share_factor*(1-.0005*multiple)**2
            near(sale['sale_net'],immediate)
            if label['exit_at'] is not None:
                future_div = sum(d['economic_amount'] for d in label['dividends'] if d['ex']>day)
                near(r['target'],np.log((label['exit_fill_economic']*(1-.0005*multiple)+future_div)/immediate))
        result[r['id']] = r;count+=1
    return result,count,latency


def verify(run):
    run = Path(run);registration = read(run/'registration.json');protocol = registration['preregistration']['protocol']
    manifest = read(run/'input_manifest.json');previous = Path(manifest['previous_run'])
    for source,value in registration['source_sha256'].items():
        assert hashlib.sha256((core.ROOT/source).read_bytes()).hexdigest()==value
    for source,value in manifest['previous_files_sha256'].items():
        assert hashlib.sha256((previous/source).read_bytes()).hexdigest()==value
    assert hashlib.sha256(Path(manifest['source_features']).read_bytes()).hexdigest()==manifest['source_features_sha256']
    market = core.load_market();assert market.manifest==manifest['market'];features = pd.read_parquet(run/'features.parquet')
    baseline = labels(run/'baseline_labels.json');stress_base = labels(run/'stress_baseline_labels.json')
    managed = labels(run/'managed_labels.json');stressed = labels(run/'stress_managed_labels.json');simple = labels(run/'simple_trend_labels.json')
    states = {};state_checks = 0;latency_checks = 0
    for name,lab,filename,multiple in [('base',baseline,'review_states.json',1),('stress',stress_base,'stress_review_states.json',2)]:
        states[name],n,l = verify_states(market,features,lab,read(run/filename),multiple)
        state_checks+=n;latency_checks+=l
        # Coverage is independent of the emitted event list and includes current-day holdings.
        expected = set()
        for k,x in lab.items():
            if 'entry_fill_economic' not in x:continue
            for _,row in features[(features.date>=x['entry_date'])&(features.date<=x['expiry_date'])].iterrows():
                for clock in protocol['review_clocks_et']:
                    at = pd.Timestamp(str(row.date.date())+' '+clock,tz=core.ET).tz_convert('UTC')
                    if x['entry_at']<at<x['expiry_at'] and at<pd.Timestamp(market.calendar[row.date]['close_at']) and (x['exit_at'] is None or x['exit_at']>at):
                        expected.add(k+'@'+at.isoformat())
        assert expected==set(states[name]),name
    releases = {}
    for at,row in features.iterrows():
        pair = [baseline[key(at,s)] for s in core.ETFS]
        if all(x.get('complete_window') and x.get('label_available_at') is not None and np.isfinite(x.get('net_return',np.nan)) for x in pair):
            for x in pair:assert x['label_available_at']>=x['expiry_at']+pd.Timedelta(seconds=1)
            releases[row.date]=max(x['label_available_at'] for x in pair)
    by_day = {name:{} for name in states}
    for name,rows in states.items():
        for r in rows.values():by_day[name].setdefault(r['at'].tz_convert(core.ET).tz_localize(None).normalize(),[]).append(r)
    all_events = [];seen_entries = [];old_manager = [];closed = {'base':set(),'stress':set()};first_intent = {'base':{},'stress':{}}
    forecast_checks = 0;membership_checks = 0;refits = 0;cancelled = 0
    with threadpool_limits(limits=1):
        for path in sorted((run/'training').glob('*.json')):
            r = read(path);at = pd.Timestamp(r['at']);day = pd.Timestamp(r['date']);cutoff = day.tz_localize(core.ET).tz_convert('UTC')
            assert pd.Timestamp(r['cutoff'])==cutoff
            dates = [d for d,release in sorted(releases.items()) if release<cutoff and d.tz_localize(core.ET)<cutoff][-252:]
            date_set = set(dates);assert dates==list(pd.to_datetime(r['eligible_dates']))
            old_entries = [(key(e['at'],e['top_symbol']),pd.Timestamp(e['entry_date']),
                np.log1p(managed[key(e['at'],e['top_symbol'])]['net_return'])-e['raw'][e['top_symbol']]) for e in seen_entries if e['top_symbol'] is not None]
            calibration_check(r['calibrations']['entry'],old_entries,date_set)
            calibration_check(r['calibrations']['manager'],old_manager,date_set)
            train = {'entry':[{'id':key(t,s),'entry_date':row.date,'state':independent_state(row,s),'target':np.log1p(managed[key(t,s)]['net_return'])}
                        for t,row in features[features.date.isin(dates)].iterrows() for s in core.ETFS],
                'manager':[x for x in states['base'].values() if x['entry_date'] in date_set and x['state'] is not None and np.isfinite(x['target'])]}
            with gzip.open(run/'models'/(str(day.date())+'.pkl.gz'),'rb') as f:models = pickle.load(f)
            for name,rows in train.items():
                info = r['fits'][name];parents = sorted({x['entry_date'] for x in rows})
                assert info['members']==[x['id'] for x in rows] and list(pd.to_datetime(info['dates']))==parents
                assert info['ready']==(len(parents)>=120)
                if info['ready']:near(info['weight_sum'],len(parents))
                else:assert models[name] is None
                membership_checks+=1
                if info['ready'] and str(day.date()) in ['2026-05-01','2026-07-01','2026-09-01']:
                    cols = ENTRY_COLUMNS if name=='entry' else MANAGER_COLUMNS
                    X = np.array([[x['state'][c] for c in cols] for x in rows]);y = np.array([x['target'] for x in rows])
                    counts = pd.Series([x['entry_date'] for x in rows]).value_counts();weights = np.array([1/counts[x['entry_date']] for x in rows])
                    model = HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,min_samples_leaf=20,
                        learning_rate=.05,l2_regularization=5,early_stopping=False,random_state=814).fit(X,y,sample_weight=weights)
                    np.testing.assert_allclose(model.predict(X),models[name].predict(X),atol=1e-13,rtol=1e-11);refits+=1
            event = r['entry'];cal = r['calibrations']['entry']
            if models['entry'] is None:assert event['action']=='CASH' and not event['raw']
            else:
                for s in core.ETFS:
                    x = independent_state(features.loc[at],s)
                    near(event['raw'][s],models['entry'].predict([[x[c] for c in ENTRY_COLUMNS]])[0]);forecast_checks+=1
                top = max(core.ETFS,key=lambda s:event['raw'][s]);assert top==event['top_symbol']
                if cal['correction'] is not None:near(event['adjusted'],event['raw'][top]+cal['correction'])
                assert event['ready']==cal['ready']
                assert event['action']==(top if cal['ready'] and event['adjusted']>0 else 'CASH')
            seen_entries.append(event)
            expected_events = []
            for name in ['base','stress']:
                for state in sorted(by_day[name].get(day,[]),key=lambda x:(x['at'],x['key'])):
                    if state['key'] in closed[name]:continue
                    expected_events.append((name,state['id']))
                    # Read in the chronological order required by the runner; ensure none were omitted.
                    e = r['manager'][len(expected_events)-1]
                    assert (e['cost_path'],e['id'])==(name,state['id'])
                    cal = r['calibrations']['manager'];model = models['manager']
                    ready = model is not None and state['state'] is not None and cal['ready'];assert e['ready']==ready
                    if model is not None and state['state'] is not None:
                        raw = model.predict([[state['state'][c] for c in MANAGER_COLUMNS]])[0]
                        near(e['raw'],raw);forecast_checks+=1
                        if cal['correction'] is not None:near(e['adjusted'],raw+cal['correction'])
                        if name=='base':old_manager.append((state['id'],state['entry_date'],state['target']-raw))
                    assert e['action']==('EXIT' if ready and e['adjusted']<=0 else 'HOLD')
                    if e['action']=='EXIT':
                        first_intent[name].setdefault(state['key'],state['at'])
                        sale = state['sale']
                        expected_execution = 'unavailable' if sale is None else 'cancelled_by_barrier_during_latency' if sale['barrier_during_latency'] else 'scheduled_market_exit'
                        assert e['execution']==expected_execution
                        if sale is not None:
                            closed[name].add(state['key']);cancelled+=int(sale['barrier_during_latency'])
            assert len(expected_events)==len(r['manager']);all_events.extend(r['manager'])
    assert all_events==read(run/'manager_predictions.json') and seen_entries==read(run/'entry_predictions.json')
    # Independently obtain the first structural-control intent for each hypothetical trade.
    simple_intents = {}
    for state in sorted(states['base'].values(),key=lambda x:(x['at'],x['key'])):
        x = state['state']
        good = x is not None and x['regular_ready'] and all(x[c]<0 for c in [
            'micro_sector_session_return','micro_sector_vwap_distance','micro_sector_recent15','micro_qqq_session_return','micro_qqq_vwap_distance'])
        if good and state['sale'] is not None:simple_intents.setdefault(state['key'],state['at'])
    raw_checks = 0
    for lab,intents,multiple in [(baseline,{},1),(stress_base,{},2),(managed,first_intent['base'],1),
                                (stressed,first_intent['stress'],2),(simple,simple_intents,1)]:
        for k,x in lab.items():
            if 'entry_fill_economic' not in x:continue
            audited = dict(x)
            if x['reason'] in ['intraday_value_exit','simple_trend_exit']:audited['reason']='continuation_exit'
            raw_exit(market,audited,intents.get(k),multiple);raw_checks+=1
            base = (stress_base if multiple==2 else baseline)[k]
            assert x['label_available_at']==base['label_available_at'] and x['complete_window']==base['complete_window']
    original_labels = labels(previous/'managed_labels.json')
    old_actions = {pd.Timestamp(e['at']):e['action'] for e in read(previous/'managed_entry_predictions.json')}
    actions = {pd.Timestamp(e['at']):e['action'] for e in seen_entries};results = read(run/'results.json');c = protocol['controls']
    books = [audit_ledger(run,name,market,features,lab,chosen,multiple,results['routes'][name])
        for name,lab,chosen,multiple in [(protocol['primary'],managed,actions,1),(c[0],baseline,actions,1),
            (c[1],original_labels,actions,1),(c[2],simple,actions,1),(c[3],original_labels,old_actions,1),('double_cost',stressed,actions,2)]]
    frozen = read(run/'decisions_frozen.json');assert len(frozen['entry'])==105
    for e in frozen['entry']:assert actions[pd.Timestamp(e['at'])]==e['action']
    assert frozen['manager_sha256']==hashlib.sha256((run/'manager_predictions.json').read_bytes()).hexdigest()
    out = {'passed':True,'training_membership_checks':membership_checks,'raw_forecast_checks':forecast_checks,
        'independent_model_refits':refits,'observed_state_checks':state_checks,'latency_zero_target_checks':latency_checks,
        'manager_intents_cancelled_by_earlier_barrier':cancelled,'raw_counterfactual_exit_checks':raw_checks,
        'capital_books':books,'primary_case_premarket_decisions':105,'full_window_maturity_preserved':True,
        'source_inputs_unchanged':True,'no_effect_pass_inferred':True}
    core.dump(run/'verification.json',out);print(json.dumps(out,indent=2));return out


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    verify(parser.parse_args().run)
