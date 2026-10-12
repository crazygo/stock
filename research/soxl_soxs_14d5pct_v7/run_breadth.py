"""Matched, daily-retrained input ablation; one preregistered primary."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from breadth import (core, PANEL, PANEL_ROOT, load_panel, freshness, timing_at, panel_at, variant_columns)
from path_engine import build_context, feature_at, exit_label, day_cutoff, digest
from learning import fit_daily, predict
from walkforward import label_tables
from risk_ledger import replay
from study import save_replay
from run_intraday import score

SOURCE = Path(__file__).resolve().parent


class MeanModel:
    def __init__(self, models):
        self.models = models

    def predict(self, values):
        return np.mean([m.predict(values) for m in self.models], axis=0)


def build_features(market, panel, protocol):
    bars, daily, manifest = panel
    context = build_context(market)
    timing = {s: freshness(market.bars[s][market.bars[s].session_type.eq('regular')], market.calendar)
              for s in ['SOXX', 'QQQ']}
    rows, audits = [], []
    for date in market.daily['SOXL'].index:
        for clock in protocol['analysis_clocks_et']:
            at = pd.Timestamp(str(date.date()) + ' ' + clock, tz=core.ET).tz_convert('UTC')
            if at >= pd.Timestamp(market.calendar[date]['close_at']):
                continue
            base, audit = feature_at(market, context, at)
            leader, leader_audit = panel_at(bars, daily, market.sessions, market.calendar, at)
            event, event_audit = {}, {}
            if base is not None and base['regular_context_ready']:
                event, event_audit = timing_at(timing, at)
            good = base is not None and leader is not None and event is not None
            audits.append({'decision_at': at, 'ready': good, 'base': audit, 'panel': leader_audit, 'timing': event_audit})
            if good:
                rows.append({'decision_at': at, 'date': date, 'clock': clock, **base, **event, **leader})
    features = pd.DataFrame(rows).set_index('decision_at').sort_index()
    pre = features.regular_context_ready.eq(0.)
    columns = [c for c in features if c.startswith(('x_', 'q_', 'timing_'))]
    features.loc[pre, columns] = features.loc[pre, columns].fillna(0.)
    if features.isna().any().any() or not np.isfinite(features.drop(columns=['date', 'clock']).to_numpy()).all():
        raise ValueError('Unexpected missing/nonfinite feature; abstain instead of fabricating observations')
    return features, audits, context


def paired_uncertainty(primary, control):
    assert primary.index.equals(control.index)
    a = np.diff(np.log(np.r_[1., primary.equity.to_numpy()]))
    b = np.diff(np.log(np.r_[1., control.equity.to_numpy()]))
    delta = a - b; n = len(delta); rng = np.random.default_rng(707)
    starts = rng.integers(0, n, size=(5000, int(np.ceil(n / 10))))
    index = ((starts[..., None] + np.arange(10)) % n).reshape(5000, -1)[:, :n]
    boot = delta[index].mean(axis=1)
    return {'method': 'paired circular block bootstrap of daily log capital returns', 'dates': n,
        'block_sessions': 10, 'replicates': 5000, 'seed': 707, 'mean_daily_log_difference': float(delta.mean()),
        'interval95': list(np.quantile(boot, [.025, .975])),
        'positive_increment_supported': bool(np.quantile(boot, .025) > 0),
        'not_independent_forward_validation': True}


def run(output, panel_root=PANEL_ROOT):
    output = Path(output)
    if output.exists():
        raise ValueError('Output must be new; retain failed attempts')
    output.mkdir(parents=True); (output / 'training').mkdir(); (output / 'executed_sources').mkdir()
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    prereg = json.loads((SOURCE / 'evidence/preregistration.json').read_text())
    assert prereg['protocol_sha256'] == hashlib.sha256((SOURCE / 'protocol.json').read_bytes()).hexdigest()
    assert protocol['panel'] == PANEL
    files = list(SOURCE.glob('*.py')) + [SOURCE / 'protocol.json']
    files += [SOURCE.parent / f'soxl_soxs_14d5pct_{v}' / name for v, name in [
        ('v2', 'core.py'), ('v2', 'study.py'), ('v3', 'risk_ledger.py'), ('v4', 'walkforward.py'),
        ('v4', 'daily_study.py'), ('v6', 'path_engine.py'), ('v6', 'learning.py'), ('v6', 'run_intraday.py')]]
    core.dump(output / 'registration.json', {'started_at': datetime.now(timezone.utc).isoformat(),
        'preregistration': prereg, 'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
        'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        'source_sha256': {str(p.relative_to(core.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}})
    for path in files:
        dest = output / 'executed_sources' / path.parent.name / path.name
        dest.parent.mkdir(exist_ok=True); dest.write_bytes(path.read_bytes())
    market = core.load_market(); panel = load_panel(market.calendar, panel_root)
    features, audits, context = build_features(market, panel, protocol)
    features.to_parquet(output / 'features.parquet', compression='zstd', compression_level=7)
    core.dump(output / 'coverage.json', audits)
    core.dump(output / 'input_manifest.json', {'market': market.manifest, 'panel': panel[2],
        'feature_rows': len(features), 'feature_dates': features.date.nunique(),
        'feature_sha256': hashlib.sha256((output / 'features.parquet').read_bytes()).hexdigest(),
        'feature_columns': {v: variant_columns(features, v) for v in protocol['feature_variants']}})
    regular = features[features.regular_context_ready.eq(1.)]
    labels = [exit_label(market, context, s, at, 'flat_close') for at in regular.index for s in core.ETFS]
    tables = label_tables(labels); core.dump(output / 'labels.json', labels)
    print('features', len(features), 'dates', features.date.nunique(), 'labels', len(labels), flush=True)
    variants = {v: regular[['date', 'clock', *variant_columns(features, v)]] for v in protocol['feature_variants']}
    names = [v + '__' + a for v in protocol['feature_variants'] for a in protocol['algorithms']]
    histories = {n: pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC')) for n in names}
    registry = []; start_time = time.monotonic()
    for i, (date, group) in enumerate(features.groupby('date')):
        record = {'date': date, 'cutoff': day_cutoff(group.index[0]), 'variants': {}}
        same_members = None
        for variant, train_features in variants.items():
            fit, info, members = fit_daily(train_features, tables, group.index[0], protocol)
            if same_members is None:
                same_members = members
            assert members.equals(same_members), 'Ablation must use identical training rows'
            record['variants'][variant] = info
            if fit:
                models, columns = fit
                models['ensemble_log'] = {s: MeanModel([models[a][s] for a in ['ridge_log', 'boosted_log']]) for s in core.ETFS}
            for at, row in group.iterrows():
                for algorithm in protocol['algorithms']:
                    name = variant + '__' + algorithm
                    result = {'clock': row.clock, 'action': 'CASH', 'ready': False,
                        'SOXL_raw': np.nan, 'SOXS_raw': np.nan, 'training_cutoff': day_cutoff(at)}
                    if not row.regular_context_ready:
                        result['reason'] = 'WAIT_FOR_REGULAR_INFORMATION'
                    elif fit is not None:
                        result.update(predict(fit, algorithm, row, histories[name], tables, members, protocol))
                    new = pd.DataFrame([result], index=pd.DatetimeIndex([at]))
                    histories[name] = new if histories[name].empty else pd.concat([histories[name], new])
        record['daily_id'] = digest(record)[:24]
        core.dump(output / 'training' / (str(date.date()) + '.json'), record)
        registry.append({'date': date, 'daily_id': record['daily_id'],
            'training_dates': record['variants']['breadth']['training_dates'],
            'first_training_date': record['variants']['breadth']['first_date'],
            'last_training_date': record['variants']['breadth']['last_date']})
        if (i + 1) % 25 == 0:
            print('trained through', date.date(), 'elapsed', round(time.monotonic() - start_time), 's', flush=True)
    core.dump(output / 'training_registry.json', registry)
    for name, history in histories.items():
        history.to_json(output / (name + '_predictions.json'), orient='table', date_format='iso', double_precision=15)
    start, end = protocol['evaluation_start'], protocol['evaluation_end']
    test = features[(features.date >= start) & (features.date <= end)]
    primary = protocol['primary']; actions = histories[primary].action.to_dict()
    core.dump(output / 'decisions_frozen.json', {'frozen_at': datetime.now(timezone.utc).isoformat(), 'primary': primary,
        'this_experiment_case_results_seen_before_rules_frozen': False, 'older_experiments_exposed_same_case_history': True,
        'actions': [{'decision_at': at, 'action': actions[at]} for at in test.index]})
    books, results = {}, {'routes': {}}
    for name, history in histories.items():
        book = replay(market, test, labels, history.action.to_dict(), start, end)
        books[name] = book; save_replay(output, name, book)
        results['routes'][name] = score(book, protocol['regression_months'])
    results['primary_name'] = primary; results['primary'] = results['routes'][primary]
    double_labels = [exit_label(market, context, s, at, 'flat_close', 2) for at in test[test.regular_context_ready.eq(1.)].index for s in core.ETFS]
    core.dump(output / 'double_cost_labels.json', double_labels)
    stress = replay(market, test, double_labels, actions, start, end, cost_multiple=2)
    save_replay(output, 'double_cost', stress); results['double_cost'] = score(stress, protocol['regression_months'])
    expected = []
    for date in market.sessions[(market.sessions >= start) & (market.sessions <= end)]:
        for clock in protocol['analysis_clocks_et']:
            at = pd.Timestamp(str(date.date()) + ' ' + clock, tz=core.ET).tz_convert('UTC')
            if at < pd.Timestamp(market.calendar[date]['close_at']):
                expected.append(at)
    complete = test.index.equals(pd.DatetimeIndex(expected))
    results['paired_feature_increment'] = paired_uncertainty(books[primary][0], books['base__ensemble_log'][0])
    results['acceptance'] = {'passed': bool(results['primary']['primary_pass'] and complete),
        'new_buy_eligible': False, 'primary_candidate': primary, 'all_expected_feature_clocks_complete': complete,
        'expected_clocks': len(expected), 'observed_clocks': len(test),
        'diagnostic_pass_count': sum(r['primary_pass'] for n, r in results['routes'].items() if n != primary),
        'independent_forward_validation_passed': False, 'future_nonnegative_month_guarantee': False,
        'live_entry_not_replaced': True}
    core.dump(output / 'results.json', results)
    core.dump(output / 'status.json', {'stage': 'complete', 'acceptance': results['acceptance']})
    print(json.dumps({'primary': results['primary'], 'stress': results['double_cost'], 'acceptance': results['acceptance']}, indent=2), flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--panel-root', type=Path, default=PANEL_ROOT)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output, args.panel_root)
