"""Independently audit as-of memberships, predictions, raw exits and continuous NAV."""
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
from continuation import core, ENTRY_COLUMNS, MANAGER_COLUMNS, OWN, SIGNED, UNSIGNED


def near(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10), (a, b)


def read(path):
    return json.loads(Path(path).read_text())


def key(at, symbol):
    return pd.Timestamp(at).isoformat() + '|' + symbol


def labels(path):
    result = {}
    for r in read(path):
        for field in ['decision_at', 'entry_date', 'entry_at', 'expiry_date', 'expiry_at',
                      'exit_at', 'exit_bar_start', 'entry_bar_end', 'label_available_at', 'manager_decision_at']:
            if r.get(field) is not None:
                r[field] = pd.Timestamp(r[field])
        for d in r.get('dividends', []):
            d['ex'] = pd.Timestamp(d['ex']); d['pay'] = pd.Timestamp(d['pay'])
        result[key(r['decision_at'], r['symbol'])] = r
    return result


def independent_state(row, symbol):
    sign = 1. if symbol == 'SOXL' else -1.
    return {'direction': sign, **{'own_' + k: float(row[symbol + '_' + k]) for k in OWN},
        **{'signed_' + k: sign * float(row[k]) for k in SIGNED}, **{k: float(row[k]) for k in UNSIGNED}}


def calibration_check(saved, records, eligible):
    # records contain (ID,parent-date,actual-minus-old-raw); each parent has equal weight.
    rows = [r for r in records if r[1] in eligible and np.isfinite(r[2])]
    dates = sorted({r[1] for r in rows})[-60:]; rows = [r for r in rows if r[1] in dates]
    assert saved['members'] == [r[0] for r in rows]
    assert list(pd.to_datetime(saved['dates'])) == dates
    assert saved['ready'] == (len(dates) >= 40)
    if not dates:
        assert saved['correction'] is None
    else:
        near(saved['correction'], np.mean([np.mean([r[2] for r in rows if r[1] == d]) for d in dates]))


def raw_exit(market, label, intent, multiple):
    """Raw RTH bars + a saved earlier management intent, no production exit helper."""
    f = market.bars[label['symbol']]; entry_at = label['entry_at']; at = label['decision_at']
    first = f[(f.start_at > at + pd.Timedelta(seconds=30)) & f.date.eq(label['entry_date'])
              & f.session_type.eq('regular')].iloc[0]
    assert entry_at == first.start_at and entry_at - at <= pd.Timedelta(minutes=6)
    fee = slip = .0005 * multiple
    entry = first.open * (1 + slip) * first.share_factor
    target = math.ceil(first.open * (1 + slip) * 1.05 * 100 - 1e-10) / 100 * first.share_factor
    stop = entry * .94
    near(label['entry_fill_economic'], entry); near(label['limit_economic'], target)
    horizon = label['entry_date'] + pd.Timedelta(days=13)
    expiry = market.sessions[market.sessions <= horizon][-1]
    assert label['expiry_date'] == expiry
    close_at = pd.Timestamp(market.calendar[expiry]['close_at'])
    path = f[(f.start_at >= entry_at) & (f.end_at <= close_at) & f.session_type.eq('regular')]
    event = None
    for bar in path.itertuples():
        rt = math.ceil(target / bar.share_factor * 100 - 1e-10) / 100
        rs = math.floor(stop / bar.share_factor * 100 + 1e-10) / 100
        if intent is not None and bar.start_at > intent + pd.Timedelta(seconds=30):
            assert bar.start_at - intent <= pd.Timedelta(minutes=6)
            event = (bar.start_at, min(bar.open, rt) * bar.share_factor * (1 - slip), 'continuation_exit'); break
        if bar.low <= rs:
            event = (bar.end_at, min(bar.open, rs) * bar.share_factor * (1 - slip), 'risk_stop'); break
        if bar.high >= rt:
            event = (bar.end_at, rt * bar.share_factor, 'target'); break
    if event is None and len(path) and path.iloc[-1].end_at == close_at:
        event = (close_at, path.iloc[-1].econ_close * (1 - slip), 'expiry')
    if event is None:
        assert label['exit_at'] is None
        return
    assert label['exit_at'] == event[0] and label['reason'] == event[2], (label, event)
    near(label['exit_fill_economic'], event[1])
    end = event[0].tz_convert(core.ET).tz_localize(None).normalize(); div = 0.
    for action in market.actions[label['symbol']]:
        ex = pd.Timestamp(action['date'])
        if action['type'] == 'CashDividends' and label['entry_date'] < ex <= end:
            factor = f[f.date >= ex].share_factor.iloc[0]
            div += float(action['value']) * factor
    near(label['dividend_per_economic_unit'], div)
    near(label['net_return'], (event[1] * (1 - fee) + div) / (entry * (1 + fee)) - 1)


def audit_ledger(run, name, market, features, lab, actions, multiple, stats):
    eq = pd.read_csv(run / (name + '_equity.csv'), parse_dates=['date']).set_index('date')
    decisions = pd.read_csv(run / (name + '_decisions.csv'))
    decisions['decision_at'] = pd.to_datetime(decisions.decision_at, utc=True)
    decisions = decisions.set_index('decision_at')
    try:
        trades = pd.read_csv(run / (name + '_trades.csv'))
    except pd.errors.EmptyDataError:
        trades = pd.DataFrame()
    actual = {pd.Timestamp(r.decision_at): r for r in trades.itertuples()}
    cash = 1.; settlements = []; receivables = []; held = None; units = 0.; count = 0
    for date, mark in eq.iterrows():
        cash += sum(v for d, v in settlements if d <= date); settlements = [(d, v) for d, v in settlements if d > date]
        cash += sum(v for d, v in receivables if d <= date); receivables = [(d, v) for d, v in receivables if d > date]
        if held is not None:
            for div in held['dividends']:
                if div['ex'] == date:
                    value = units * div['economic_amount']
                    if div['pay'] <= date: cash += value
                    else: receivables.append((div['pay'], value))
        at = features[features.date == date].index[0]
        # Single premarket time: yesterday's sales have settled, today's sales are still in the future.
        if held is None and cash > 1e-15 and actions[at] in core.ETFS:
            chosen = lab[key(at, actions[at])]
            if 'entry_fill_economic' in chosen:
                assert at in actual and actual[at].symbol == actions[at]
                near(actual[at].capital, cash)
                for field in ['entry_fill_economic', 'exit_fill_economic', 'net_return', 'dividend_per_economic_unit']:
                    if np.isfinite(chosen.get(field, np.nan)):
                        near(getattr(actual[at], field), chosen[field])
                assert actual[at].reason == chosen['reason']
                if chosen['exit_at'] is not None:
                    assert pd.Timestamp(actual[at].exit_at) == chosen['exit_at']
                units = cash / (chosen['entry_fill_economic'] * (1 + .0005 * multiple))
                near(actual[at].units_economic, units)
                held = chosen; cash = 0.; count += 1
                assert decisions.loc[at, 'decision'] == 'buy_' + actions[at]
        else:
            assert at not in actual
        if held is not None and held['exit_at'] is not None and held['exit_at'].tz_convert(core.ET).date() == date.date():
            due = market.sessions[market.sessions > date][0]
            settlements.append((due, units * held['exit_fill_economic'] * (1 - .0005 * multiple)))
            held = None
        value = units * market.daily[held['symbol']].loc[date, 'close'] if held is not None else 0.
        outstanding = sum(v for _, v in settlements); receivable = sum(v for _, v in receivables)
        for field, expected in [('equity', cash + outstanding + receivable + value), ('cash_settled', cash),
                                ('cash_unsettled', outstanding), ('position_value', value), ('receivables', receivable)]:
            near(mark[field], expected)
        assert mark.symbol == (held['symbol'] if held else 'CASH') and mark.mark_complete
    assert count == len(trades)
    prior = 1.; rates = []
    for month, group in eq.groupby(eq.index.to_period('M')):
        rate = group.equity.iloc[-1] / prior - 1; prior = group.equity.iloc[-1]
        rates.append((str(month), rate))
        near(rate, next(m['return'] for m in stats['monthly'] if m['month'] == str(month)))
    avg = np.mean([v for m, v in rates if m in ['2026-05', '2026-07', '2026-09']])
    near(stats['primary_average'], avg); near(stats['continuous_return'], eq.equity.iloc[-1] - 1)
    assert stats['primary_pass'] == (avg >= .1 and all(v >= -1e-12 for _, v in rates))
    return {'name': name, 'trades': count, 'daily_marks': len(eq), 'dividends_settlement_and_mtm_reconciled': True}


def verify(run):
    run = Path(run); registration = read(run / 'registration.json')
    protocol = registration['preregistration']['protocol']; manifest = read(run / 'input_manifest.json')
    for source, digest in registration['source_sha256'].items():
        assert hashlib.sha256((core.ROOT / source).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(Path(manifest['source_feature_file']).read_bytes()).hexdigest() == manifest['source_feature_sha256']
    market = core.load_market(); assert market.manifest == manifest['market']
    features = pd.read_parquet(run / 'features.parquet')
    normal = labels(run / 'baseline_labels.json'); managed = labels(run / 'managed_labels.json')
    stress_base = labels(run / 'stress_baseline_labels.json'); stressed = labels(run / 'stress_managed_labels.json')
    release = {}
    for at, row in features.iterrows():
        pair = [normal[key(at, s)] for s in core.ETFS]
        if all(x.get('complete_window') and x.get('label_available_at') is not None and np.isfinite(x.get('net_return', np.nan)) for x in pair):
            for x in pair: assert x['label_available_at'] >= x['expiry_at'] + pd.Timedelta(seconds=1)
            release[row.date] = max(x['label_available_at'] for x in pair)
    states = {}; state_checks = 0
    for name, lab in [('base', normal), ('stress', stress_base)]:
        states[name] = {}
        for r in read(run / ('review_states.json' if name == 'base' else 'stress_review_states.json')):
            at = pd.Timestamp(r['at']); r['at'] = at; r['entry_date'] = pd.Timestamp(r['entry_date'])
            x = lab[r['key']]; f = market.bars[x['symbol']]
            if r['state'] is not None:
                row = features.loc[at]; state = independent_state(row, x['symbol'])
                observed = f[(f.available_at <= at) & (f.end_at <= at)]; quote = observed.iloc[-1]
                peak = max(x['entry_fill_economic'], observed[(observed.start_at >= x['entry_at']) & observed.session_type.eq('regular')].econ_high.max())
                price = quote.econ_close; age = (row.date - x['entry_date']).days
                state.update(age_calendar=age, remaining_calendar=13-age,
                    mark_return=price/x['entry_fill_economic']-1, target_distance=x['limit_economic']/price-1,
                    stop_distance=x['entry_fill_economic']*.94/price-1, observed_peak_drawdown=price/peak-1)
                for col, value in state.items(): near(r['state'][col], value)
                assert quote.available_at <= at and at - quote.end_at <= pd.Timedelta(minutes=6)
                if r['sale'] is not None and x['exit_at'] is not None:
                    first = f[(f.date == row.date) & f.session_type.eq('regular') & (f.start_at > at + pd.Timedelta(seconds=30))].iloc[0]
                    multiple = 2 if name == 'stress' else 1
                    rt = math.ceil(x['limit_economic']/first.share_factor*100-1e-10)/100
                    immediate = min(first.open, rt)*first.share_factor*(1-.0005*multiple)**2
                    remaining = sum(d['economic_amount'] for d in x['dividends'] if d['ex'] > row.date)
                    expected = np.log((x['exit_fill_economic']*(1-x['fee_each_side'])+remaining)/immediate)
                    near(r['target'], expected); near(r['sale']['sale_net'], immediate)
                state_checks += 1
            states[name][r['id']] = r
    entry_history = {n: read(run / (n + '_entry_predictions.json')) for n in ['managed', 'baseline']}
    manager_events = read(run / 'manager_predictions.json'); previous_manager = []
    seen_entries = {n: [] for n in entry_history}; forecast_checks = 0; membership_checks = 0; refits = 0
    intended = {'base': {}, 'stress': {}}
    with threadpool_limits(limits=1):
        for path in sorted((run / 'training').glob('*.json')):
            record = read(path); at = pd.Timestamp(record['at']); date = pd.Timestamp(record['date'])
            cutoff = date.tz_localize(core.ET).tz_convert('UTC')
            assert pd.Timestamp(record['cutoff']) == cutoff
            dates = [d for d, r in sorted(release.items()) if r < cutoff and d.tz_localize(core.ET) < cutoff][-252:]
            assert dates == list(pd.to_datetime(record['eligible_dates']))
            with gzip.open(run / 'models' / (str(date.date()) + '.pkl.gz'), 'rb') as f: models = pickle.load(f)
            train = {}
            for name, lab in [('managed', managed), ('baseline', normal)]:
                rows = [{'id': key(t, s), 'entry_date': row.date, 'state': independent_state(row, s),
                         'target': np.log1p(lab[key(t, s)]['net_return'])}
                        for t, row in features[features.date.isin(dates)].iterrows() for s in core.ETFS]
                train[name] = rows
                old = [(key(r['at'], r['top_symbol']), pd.Timestamp(r['entry_date']),
                        np.log1p(lab[key(r['at'], r['top_symbol'])]['net_return']) - r['raw'][r['top_symbol']])
                       for r in seen_entries[name] if r['top_symbol'] is not None]
                calibration_check(record['calibrations'][name], old, dates)
            train['manager'] = [r for r in states['base'].values() if r['entry_date'] in dates and r['state'] is not None and np.isfinite(r['target'])]
            calibration_check(record['calibrations']['manager'], previous_manager, dates)
            for name, rows in train.items():
                saved = record['fits'][name]; unique = sorted({r['entry_date'] for r in rows})
                assert saved['members'] == [r['id'] for r in rows] and list(pd.to_datetime(saved['dates'])) == unique
                assert saved['ready'] == (len(unique) >= 120)
                if saved['ready']: near(saved['weight_sum'], len(unique))
                else: assert models[name] is None
                membership_checks += 1
                # Refit three fixed calendar anchors, without calling the experiment's fit function.
                if saved['ready'] and str(date.date()) in ['2026-05-01', '2026-07-01', '2026-09-01']:
                    cols = MANAGER_COLUMNS if name == 'manager' else ENTRY_COLUMNS
                    X = np.array([[r['state'][c] for c in cols] for r in rows]); y = np.array([r['target'] for r in rows])
                    counts = pd.Series([r['entry_date'] for r in rows]).value_counts()
                    weights = np.array([1/counts[r['entry_date']] for r in rows])
                    refit = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=7, min_samples_leaf=20,
                        learning_rate=.05, l2_regularization=5, early_stopping=False, random_state=814).fit(X, y, sample_weight=weights)
                    np.testing.assert_allclose(refit.predict(X), models[name].predict(X), atol=1e-13, rtol=1e-11); refits += 1
            for name in seen_entries:
                event = record['entry'][name]; cal = record['calibrations'][name]
                if models[name] is None: assert event['action'] == 'CASH' and not event['raw']
                else:
                    for s in core.ETFS:
                        x = independent_state(features.loc[at], s)
                        near(event['raw'][s], models[name].predict([[x[c] for c in ENTRY_COLUMNS]])[0]); forecast_checks += 1
                    top = max(core.ETFS, key=lambda s: event['raw'][s]); assert top == event['top_symbol']
                    if cal['correction'] is not None: near(event['adjusted'], event['raw'][top]+cal['correction'])
                    assert event['action'] == (top if cal['ready'] and event['adjusted'] > 0 else 'CASH')
                seen_entries[name].append(event)
            expected_reviews = {(name, r['id']) for name in states for r in states[name].values()
                                if r['at'] == at and r['key'] not in intended[name]}
            assert {(e['cost_path'], e['id']) for e in record['manager']} == expected_reviews
            for event in record['manager']:
                name = event['cost_path']; r = states[name][event['id']]; x = (normal if name == 'base' else stress_base)[r['key']]
                assert x['entry_at'] < at and (x['exit_at'] is None or x['exit_at'] > at)
                assert r['key'] not in intended[name]
                cal = record['calibrations']['manager']; model = models['manager']
                assert event['ready'] == (model is not None and r['state'] is not None and cal['ready'])
                if model is not None and r['state'] is not None:
                    raw = model.predict([[r['state'][c] for c in MANAGER_COLUMNS]])[0]
                    near(event['raw'], raw); forecast_checks += 1
                    if cal['correction'] is not None: near(event['adjusted'], raw+cal['correction'])
                    if name == 'base': previous_manager.append((r['id'], r['entry_date'], r['target']-raw))
                assert event['action'] == ('EXIT' if event['ready'] and event['adjusted'] <= 0 else 'HOLD')
                if event['action'] == 'EXIT' and r['sale'] is not None: intended[name][r['key']] = at
    assert all(seen_entries[n] == entry_history[n] for n in seen_entries)
    assert manager_events == [e for p in sorted((run/'training').glob('*.json')) for e in read(p)['manager']]
    raw_checks = 0
    for name, lab, intents, multiple in [('baseline', normal, {}, 1), ('managed', managed, intended['base'], 1),
        ('stress_baseline', stress_base, {}, 2), ('stress_managed', stressed, intended['stress'], 2)]:
        for k, x in lab.items():
            if 'entry_fill_economic' not in x: continue
            raw_exit(market, x, intents.get(k), multiple); raw_checks += 1
            original = (stress_base if multiple == 2 else normal)[k]
            assert x['label_available_at'] == original['label_available_at']
    results = read(run/'results.json'); primary_actions = {pd.Timestamp(r['at']):r['action'] for r in entry_history['managed']}
    baseline_actions = {pd.Timestamp(r['at']):r['action'] for r in entry_history['baseline']}
    trend = {at: 'SOXL' if r.SOXX_ret20 > 0 else 'SOXS' if r.SOXX_ret20 < 0 else 'CASH' for at,r in features.iterrows()}
    books = [audit_ledger(run, name, market, features, lab, actions, mult, results['routes'][name])
             for name,lab,actions,mult in [(protocol['primary'],managed,primary_actions,1),
                (protocol['controls'][0],normal,primary_actions,1),(protocol['controls'][1],normal,baseline_actions,1),
                (protocol['controls'][2],normal,trend,1),('double_cost',stressed,primary_actions,2)]]
    frozen = read(run/'decisions_frozen.json')
    assert len(frozen['entry']) == 105
    for r in frozen['entry']: assert primary_actions[pd.Timestamp(r['at'])] == r['action']
    assert frozen['manager_predictions_sha256'] == hashlib.sha256((run/'manager_predictions.json').read_bytes()).hexdigest()
    out = {'passed': True, 'training_membership_checks': membership_checks, 'raw_forecast_checks': forecast_checks,
           'independent_model_refits': refits, 'observed_holding_state_checks': state_checks,
           'raw_counterfactual_exit_checks': raw_checks, 'capital_books': books,
           'primary_case_premarket_decisions': len(frozen['entry']), 'full_window_maturity_preserved': True,
           'source_inputs_unchanged': True, 'no_effect_pass_inferred': True}
    core.dump(run/'verification.json', out); print(json.dumps(out, indent=2)); return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('run', type=Path)
    verify(parser.parse_args().run)
