"""Independent source joins, scalar algebra and weighted summaries; no producer import."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
CLASS = ['target', 'ordinary_stop', 'gap_stop', 'expiry']
TERM = ['prediction', 'source_prediction', 'actual', 'total_error', 'regression_adjustment',
        'category_error', 'payoff_error', 'ordinary_contribution', 'gap_contribution', 'expiry_contribution']
checks = 0


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a, b):
    global checks
    checks += 1
    assert np.isclose(a, b, atol=2e-12, rtol=2e-11, equal_nan=True), (a, b)


def verify(run):
    result = read(run / 'results.json'); inputs = read(run / 'inputs.json')
    for f, h in inputs.items():
        assert sha(ROOT / f) == h, f
    for v in [78, 79, 80, 81]:
        manifest = read(ROOT / f'research/soxl_soxs_14d5pct_v{v}/evidence/delivery_manifest.json')
        prefix = manifest['cache_root'] + '/'
        for f, h in inputs.items():
            if f.startswith(prefix):
                assert h == manifest['cache_files_sha256'][f[len(prefix):]]
    def src(v, relative):
        p = ROOT / f'.cache/soxl_soxs_14d5pct_v{v}/runs/development' / relative
        assert str(p.relative_to(ROOT)) in inputs
        return p
    plan = read(SOURCE / 'protocol.json')
    panel = pd.read_parquet(src(81, 'semiconductor_panel.parquet'))
    panel = panel[panel.date.between(plan['evaluation_start'], plan['evaluation_end'])]
    cutoff = pd.Timestamp('2026-05-01 00:00:00', tz='America/New_York').tz_convert('UTC')
    t = pd.read_csv(run / 'candidate_rows.csv').set_index('at', drop=False)
    c = pd.read_csv(run / 'model_rows.csv')
    actual = pd.read_csv(run / 'executed_rows.csv')
    assert set(t.index) == {str(x) for x in panel.index} and len(t) == len(panel)
    assert not t.index.duplicated().any()
    records = {}; reference = []
    bank = {p['date'][:10]: p for p in read(src(81, 'paired_registry.json'))}
    for at, p in panel.iterrows():
        key = str(at); row = t.loc[key]
        stamp = at.tz_convert('America/New_York').strftime('%Y-%m-%d_%H%M')
        old = {v: read(src(v, 'training/' + stamp + '.json')) for v in [78, 79, 80, 81]}
        records[key] = old
        s = old[80]['events']['recent_ratio']; e = old[81]['events']['paired_state']
        assert row.source_ready == s['ready'] and row.paired_ready == e['ready']
        assert row.source_eligible == (s['action'] != 'CASH')
        assert row.paired_accepted == (e['action'] != 'CASH')
        for route in ['absolute_pair', 'absolute_now']:
            assert row[route + '_accepted'] == (old[81]['events'][route]['action'] != 'CASH')
        b = bank.get(row.date)
        assert row.paired_mature == bool(b and b['release'] and pd.Timestamp(b['release']) < cutoff)
        assert row.visible_leadership == (p.trend5 > 0 and p.relative5 > 0)
        if not (s['ready'] and e['ready']):
            assert not row.known
            continue
        j = ['SOXL', 'SOXS'].index(s['top_symbol']); side = 'L' if j == 0 else 'S'
        y = int(p[side + '_gap']); r = p[side + '_log']
        known = pd.notna(p.terminal_release) and p.terminal_release < cutoff and 0 <= y < 4 and np.isfinite(r)
        assert row.known == known and row.symbol == s['top_symbol']
        assert row.date == str(p.date.date()) and row.clock == p.clock
        sign = 1 if j == 0 else -1
        for field, value in [('aligned_trend5', sign*p.trend5), ('aligned_current', sign*p.current),
                             ('relative5', p.relative5), ('sector_trend5', p.trend5)]:
            close(row[field], value)
        state = ('trend_positive' if sign*p.trend5 > 0 else 'trend_nonpositive') + '/' + ('current_positive' if sign*p.current > 0 else 'current_nonpositive')
        assert row.state == state
        q = s['probabilities'][j]; m = old[78]['distributions']['gap']['payoff_means'][j]
        n = e['values'][0]; g = sum(q[k]*m[k] for k in range(4))
        close(row.prediction, n); close(row.source_prediction, g)
        close(g, s['values'][j]); close(g, e['source_value'])
        for k in range(4):
            close(row[f'q{k}'], q[k]); close(row[f'm{k}'], m[k])
        if known:
            expected = [n, g, r, n-r, n-g, g-m[y], m[y]-r]
            expected += [(q[k]-int(y == k))*(m[k]-m[0]) for k in range(1,4)]
            for f, value in zip(TERM, expected):
                close(row[f], value)
            assert row.truth == y and row.truth_name == CLASS[y]
        else:
            assert pd.isna(row.truth) and pd.isna(row.actual)
            assert all(pd.isna(row[f]) for f in TERM[3:])
        estimates = [old[78]['estimates']['gap_' + key] for key in ['prior', 'raw', 'blend']]
        estimates += [old[79]['events']['top_tilt'], s]
        for name, estimate in zip(plan['prediction_sources'], estimates):
            cr = c[(c['at'] == key) & (c.model == name)]
            assert len(cr) == 1
            cr = cr.iloc[0]; pq = estimate['probabilities'][j]
            close(cr.prediction, sum(pq[k]*m[k] for k in range(4)))
            assert cr.known == known and cr.symbol == row.symbol and cr.date == row.date
            assert cr.source_eligible == row.source_eligible and cr.paired_accepted == row.paired_accepted
            for k in range(4):
                close(cr[f'q{k}'], pq[k])
            if known:
                close(cr.brier, sum((pq[k]-int(k == y))**2 for k in range(4)))
                close(cr.log_loss, -math.log(max(pq[y], 1e-8)))
                close(cr.error, cr.prediction-r); close(cr.squared_return_error, (cr.prediction-r)**2)
                close(cr.category_error, cr.prediction-m[y]); close(cr.payoff_error, m[y]-r)
            else:
                assert all(pd.isna(cr[f]) for f in ['actual', 'brier', 'log_loss', 'error', 'squared_return_error'])
    assert len(c) == int((t.source_ready & t.paired_ready).sum()) * 5
    masks = {'all_top': t.source_ready & t.paired_ready,
             'source_eligible': t.source_ready & t.paired_ready & t.source_eligible,
             'source_rejected': t.source_ready & t.paired_ready & ~t.source_eligible,
             'paired_accepted': t.paired_accepted,
             'paired_rejected': t.source_ready & t.paired_ready & t.source_eligible & ~t.paired_accepted,
             'absolute_pair_accepted': t.absolute_pair_accepted, 'absolute_now_accepted': t.absolute_now_accepted}
    def weights(group):
        sizes = group.date.value_counts().to_dict()
        return np.array([1/len(sizes)/sizes[x] for x in group.date]) if len(group) else np.array([])
    def stats(group, saved):
        g = group[group.known]; w = weights(g)
        assert saved['rows'] == len(group) and saved['dates'] == group.date.nunique()
        assert saved['known_rows'] == len(g) and saved['known_dates'] == g.date.nunique()
        assert saved['unknown_rows'] == len(group)-len(g)
        assert saved['paired_mature_rows'] == group.paired_mature.sum()
        available = group[group.paired_ready]; aw = weights(available)
        if len(available):
            for field in ['prediction', 'source_prediction']:
                close(saved['all_date_equal_predictions'][field], np.dot(aw, available[field]))
        else:
            assert saved['all_date_equal_predictions'] is None
        if not len(g):
            assert saved['date_equal_terms'] is None
            assert all(v is None for v in saved['class_statistics'].values())
            return
        for field in TERM:
            close(saved['date_equal_terms'][field], np.dot(w, g[field]))
        for k, name in enumerate(CLASS):
            st = saved['class_statistics'][name]; mask = g.truth == k; mass = float(w[mask].sum())
            close(st['predicted_probability'], np.dot(w, g[f'q{k}']))
            close(st['observed_frequency'], mass)
            assert st['rows'] == mask.sum() and st['dates'] == g[mask].date.nunique()
            close(st['amplitude_contribution'], np.dot(w[mask], g.loc[mask, f'm{k}']-g.loc[mask,'actual']))
            if mass:
                close(st['conditional_predicted_payoff'], np.dot(w[mask],g.loc[mask,f'm{k}'])/mass)
                close(st['conditional_actual_payoff'], np.dot(w[mask],g.loc[mask,'actual'])/mass)
            else:
                assert st['conditional_predicted_payoff'] is None and st['conditional_actual_payoff'] is None
    grid = sorted(t.date.unique()); n = len(grid)
    starts = np.random.default_rng(707).integers(0, n, size=(5000, math.ceil(n/10)))
    for name, mask in masks.items():
        group = t[mask]; stats(group, result['cohorts'][name])
        for month, saved in result['by_month'][name].items():
            stats(group[group.date.str[:7] == month], saved)
        known = group[group.known]; byday = {d: g[TERM].mean().to_numpy() for d,g in known.groupby('date')}
        draws = []
        for ss in starts:
            indices = [(int(start)+offset)%n for start in ss for offset in range(10)][:n]
            chosen = [byday[grid[i]] for i in indices if grid[i] in byday]
            if chosen: draws.append(np.mean(chosen, axis=0))
        saved = result['uncertainty'][name]
        assert saved['unknown_draws'] == 5000-len(draws) and saved['full_grid_dates'] == n
        assert saved['nonempty_dates'] == len(byday)
        for k, field in enumerate(TERM):
            if draws:
                interval = np.percentile(np.array(draws)[:,k], [2.5,97.5])
                for a,b in zip(interval, saved['intervals'][field]): close(a,b)
            else: assert saved['intervals'][field] is None
    for name, saved in result['states'].items():
        group = t[masks[name]]
        for key, value in saved['direction_quadrants'].items(): stats(group[group.state == key],value)
        for key, value in saved['visible_leadership'].items(): stats(group[group.visible_leadership == (key=='true')],value)
    for name, saved in result['model_scores'].items():
        keys = set(t[masks[name]]['at'])
        for model, value in saved.items():
            group = c[(c.model == model)&c['at'].isin(keys)]; g=group[group.known];w=weights(g)
            assert value['rows']==len(group) and value['known_rows']==len(g) and value['known_dates']==g.date.nunique()
            for field, val in value['date_equal_scores'].items():close(val,np.dot(w,g[field]))
    for route in plan['executed_routes']:
        trades = pd.read_csv(src(81, route+'_trades.csv')); group = actual[actual.route==route]
        assert set(group['at'])==set(trades.decision_at) and len(group)==len(trades)
        for tr in trades.itertuples():
            a=group[group['at']==tr.decision_at].iloc[0];p=t.loc[tr.decision_at]
            y=0 if tr.reason=='target' else 2 if tr.reason=='risk_stop' and tr.gap_below_stop else 1 if tr.reason=='risk_stop' else 3
            assert a.truth==y and a.truth_name==CLASS[y] and a.gap_below_stop==bool(tr.gap_below_stop)
            assert a.symbol==tr.symbol and a.date==p.date
            r=math.log1p(tr.net_return);close(a.actual_execution_log_return,r);close(a.net_return,tr.net_return)
            old=records[tr.decision_at];j=['SOXL','SOXS'].index(tr.symbol)
            pred=old[80]['events']['recent_ratio']['values'][j] if route=='source_recent' else old[81]['events'][route]['values'][0]
            close(a.prediction,pred);close(a.actual,p.actual)
            for k in range(4):close(a[f'q{k}'],p[f'q{k}']);close(a[f'm{k}'],p[f'm{k}'])
            for field in ['source_prediction','category_error','payoff_error','ordinary_contribution','gap_contribution','expiry_contribution']:
                close(a[field],p[field])
            close(a.regression_adjustment,pred-p.source_prediction);close(a.total_error,pred-p.actual)
            close(a.execution_difference,p.actual-r);close(a.total_execution_error,pred-r)
            close(a.total_execution_error,a.regression_adjustment+a.category_error+a.payoff_error+a.execution_difference)
        saved=result['executed'][route];assert saved['entries']==len(group)
        for cls in CLASS:assert saved['counts'][cls]==int((group.truth_name==cls).sum())
        for k,cls in enumerate(CLASS):
            close(saved['class_probabilities'][cls],group[f'q{k}'].mean())
            close(saved['class_frequencies'][cls],(group.truth_name==cls).mean())
        for field,value in saved['entry_equal_terms'].items():close(value,group[field].mean())
    assert not result['strategy_changed'] and not result['new_cases_run'] and not result['goal_complete']
    output=dict(passed=True,numeric_checks=checks,candidate_rows=len(t),model_rows=len(c),executed_rows=len(actual),
                input_files=len(inputs),cohorts=len(masks),bootstrap_draws_per_cohort=5000,
                result_sha256=sha(run/'results.json'),inputs_sha256=sha(run/'inputs.json'),
                source_sha256=sha(Path(__file__)),goal_complete=False)
    (run/'verification.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);verify(p.parse_args().run)
