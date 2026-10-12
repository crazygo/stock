"""Registered daily paired-regime study; preserve every failed attempt."""
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
from paired_entry import (core, MODELS, BASE_COLUMNS, REGIME_COLUMNS, restrict_market, development_gate,
    baseline_labels, joint_releases, eligible, day_cutoff, prepare, fit, predict,
    paired_calibration, decision, load_panel, premarket_volumes, factor_frame)
from risk import direction
from risk_ledger import replay
from run_intraday import score
from study import save_replay
from run_breadth import paired_uncertainty

SOURCE = Path(__file__).resolve().parent


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authorize_cases(run, protocol):
    run = Path(run); registration = json.loads((run/'registration.json').read_text())
    verification = json.loads((run/'verification.json').read_text()); result = json.loads((run/'results.json').read_text())
    assert registration['phase'] == 'development' and registration['protocol_sha256'] == sha(SOURCE/'protocol.json')
    assert verification['passed'] and verification['results_sha256'] == sha(run/'results.json')
    gate = development_gate(result['primary'], result['coverage']['complete'], protocol)
    if not gate['passed']: raise ValueError('Pre-case development gate failed; case replay is forbidden')
    return {'run': str(run.resolve()), 'results_sha256': sha(run/'results.json'), 'gate': gate}


def run(output, case_after=None):
    p = json.loads((SOURCE/'protocol.json').read_text()); prereg = json.loads((SOURCE/'evidence/preregistration.json').read_text())
    assert prereg['protocol_sha256'] == sha(SOURCE/'protocol.json')
    authorization = authorize_cases(case_after, p) if case_after else None
    phase = 'cases' if case_after else 'development'; settings = p[phase]
    start, end = settings['evaluation_start'], settings['evaluation_end']; cutoff = pd.Timestamp(end, tz=core.ET)+pd.Timedelta(days=1)
    output = Path(output)
    if output.exists(): raise ValueError('Use a new output directory; preserve every attempt')
    for sub in ['', 'models', 'training', 'executed_sources']: (output/sub).mkdir(parents=True, exist_ok=True)
    source = core.ROOT/p['source_features']; source_manifest = json.loads((source.parent/'input_manifest.json').read_text())
    assert source_manifest['feature_sha256'] == sha(source)
    for name in ['verification', 'input_verification', 'numerical_audit']:
        assert json.loads((source.parent/(name+'.json')).read_text())['passed']
    original = core.load_market(); assert original.manifest == source_manifest['market']
    market = restrict_market(original, cutoff); features = pd.read_parquet(source)
    features = features[features.clock.eq(p['analysis_clock_et']) & (features.index<cutoff)]
    assert not features.date.duplicated().any()
    features.to_parquet(output/'features.parquet', compression='zstd', compression_level=7)
    panel_bars, _, panel_manifest = load_panel(original.calendar)
    volumes, volume_checks = premarket_volumes(panel_bars, original.sessions, cutoff)
    factors = factor_frame(features, volumes)
    volumes.to_parquet(output/'premarket_volumes.parquet', compression='zstd', compression_level=7)
    core.dump(output/'volume_audit.json', volume_checks)
    files = list(SOURCE.glob('*.py')) + [SOURCE/'protocol.json']
    files += [SOURCE.parent/f'soxl_soxs_14d5pct_{v}'/n for v,n in [
        ('v2','core.py'),('v2','study.py'),('v3','risk.py'),('v3','risk_ledger.py'),('v4','walkforward.py'),
        ('v4','daily_study.py'),('v6','path_engine.py'),('v6','run_intraday.py'),('v7','breadth.py'),
        ('v7','run_breadth.py'),('v9','continuation.py'),('v10','micro_manager.py'),('v11','uncertainty.py'),('v12','entry_value.py'),('v13','linear_entry.py')]]
    core.dump(output/'registration.json', {'phase': phase, 'started_at': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha(SOURCE/'protocol.json'), 'preregistration': prereg, 'authorization': authorization,
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        'source_sha256': {str(f.relative_to(core.ROOT)): sha(f) for f in files}})
    for f in files:
        dest = output/'executed_sources'/f.parent.name/f.name; dest.parent.mkdir(exist_ok=True); dest.write_bytes(f.read_bytes())
    base = baseline_labels(market, features); stress = baseline_labels(market, features, 2)
    releases = joint_releases(features, base)
    releases = {d:t for d,t in releases.items() if bool(factors.loc[factors.date.eq(d),'feature_ready'].all())}
    panel = prepare(features, base, releases, factors)
    panel.to_parquet(output/'panel.parquet', compression='zstd', compression_level=7)
    core.dump(output/'baseline_labels.json', list(base.values())); core.dump(output/'stress_labels.json', list(stress.values()))
    core.dump(output/'input_manifest.json', {'source_feature_file': str(source), 'source_feature_sha256': sha(source),
        'full_market': original.manifest, 'observation_cutoff_exclusive': cutoff, 'feature_sha256': sha(output/'features.parquet'),
        'panel_sha256': sha(output/'panel.parquet'), 'rows': len(features), 'base_columns': BASE_COLUMNS, 'regime_columns': REGIME_COLUMNS, 'panel_market': panel_manifest,
        'volume_sha256': sha(output/'premarket_volumes.parquet'),
        'restricted_market': {s: {'rows': len(f), 'last_bar_end': f.end_at.max(), 'last_available_at': f.available_at.max()} for s,f in market.bars.items()}})
    history = {name: [] for name in MODELS}; controls = {name: [] for name in ['same_primary_direction', 'same_primary_raw']}; registry = []
    print('prepared', len(features), 'dates;', len(releases), 'joint mature dates', flush=True)
    with threadpool_limits(limits=1):
        for i, (at, row) in enumerate(features.iterrows()):
            cutoff_day = day_cutoff(at); dates = eligible(releases, cutoff_day)
            current = panel.loc[[at]]; record = {'date': row.date, 'at': at, 'cutoff': cutoff_day, 'routes': {}}; models = {}
            for name in MODELS:
                model, info = fit(panel, dates, name); raw = predict(model, current)
                cal = paired_calibration(history[name], panel, dates)
                event = {'date': row.date, 'at': at, **decision(raw, cal)}
                history[name].append(event); models[name] = model
                record['routes'][name] = {'fit': info, 'calibration': cal, 'event': event,
                    'latest_label_release': max((releases[d] for d in dates), default=None)}
                if name == p['primary']:
                    for control, method in [('same_primary_direction','direction'), ('same_primary_raw','raw')]:
                        controls[control].append({'date': row.date, 'at': at, **decision(raw, cal, method)})
            with gzip.open(output/'models'/(str(row.date.date())+'.pkl.gz'), 'wb') as f: pickle.dump(models,f)
            core.dump(output/'training'/(str(row.date.date())+'.json'), record)
            registry.append({'date': row.date, 'cutoff': cutoff_day, 'first_training_date': min(dates,default=None),
                'last_training_date': max(dates,default=None), 'latest_label_release': max((releases[d] for d in dates),default=None),
                'routes': {name: {'training_dates': len(v['fit']['dates']), 'effective_parent_n': v['fit']['parent_effective_n'],
                    'calibration_dates': len(v['calibration']['dates']), 'calibration_coefficients': {k:m['coefficient'] for k,m in v['calibration']['maps'].items()}, 'action': v['event']['action']}
                    for name,v in record['routes'].items()}})
            if (i+1)%50 == 0: print('trained through', row.date.date(), flush=True)
    histories = {**history, **controls}; core.dump(output/'predictions.json', histories); core.dump(output/'training_registry.json', registry)
    test = features[(features.date>=start)&(features.date<=end)]
    actions = {name: {e['at']: e['action'] for e in events} for name,events in histories.items()}
    actions['trend20'] = {at: direction(row,'trend20') for at,row in test.iterrows()}; actions['cash'] = {at:'CASH' for at in test.index}
    actions['always_SOXL'] = {at:'SOXL' for at in test.index}
    actions['double_cost'] = actions[p['primary']]
    core.dump(output/'decisions_frozen.json', {'frozen_at': datetime.now(timezone.utc).isoformat(), 'phase': phase,
        'new_route_outcomes_seen_before_decisions_frozen': False, 'development_interval_previously_exposed': True,
        'prediction_sha256': sha(output/'predictions.json'), 'actions': {name: [{'at':at,'action':a[at]} for at in test.index] for name,a in actions.items()}})
    months = settings.get('months', settings.get('regression_months')); results = {'phase':phase,'primary_name':p['primary'],'routes':{}}; books = {}
    for name,a in actions.items():
        multiple = 2 if name=='double_cost' else 1
        book = replay(market,test,list((stress if multiple==2 else base).values()),a,start,end,cost_multiple=multiple)
        books[name] = book; save_replay(output,name,book); results['routes'][name] = score(book,months)
    required = market.sessions[(market.sessions>=start)&(market.sessions<=end)]; covered = pd.DatetimeIndex(test.date).equals(required)
    results['coverage'] = {'complete':covered,'expected_days':len(required),'observed_days':len(test)}; results['primary'] = results['routes'][p['primary']]
    results['paired_increments'] = {n: paired_uncertainty(books[p['primary']][0],books[n][0]) for n in ['regret_base','sign_regime','ridge_regime','mean_pair','same_primary_direction','same_primary_raw','always_SOXL','trend20']}
    if phase=='development':
        gate = development_gate(results['primary'],covered,p); results['development_gate'] = gate
        results['case_regression'] = {'executed':False,'eligible_after_verification':gate['passed']}
    else: results['case_acceptance'] = {'passed':bool(covered and results['primary']['primary_pass'])}
    results.update(goal_complete=False,new_buy_eligible=False,future_nonnegative_month_guarantee=False)
    core.dump(output/'results.json',results)
    print(json.dumps({n:{'mean':r['primary_average'],'return':r['continuous_return'],'pass':r['primary_pass']} for n,r in results['routes'].items()},indent=2),flush=True)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True); parser.add_argument('--case-after',type=Path)
    a=parser.parse_args(); run(a.output,a.case_after)
