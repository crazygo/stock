"""Read-only accounting of as-issued probability, payoff and regression errors."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent
CACHE = ROOT / '.cache/soxl_soxs_14d5pct_v82'
CLASSES = ['target', 'ordinary_stop', 'gap_stop', 'expiry']
ETFS = ['SOXL', 'SOXS']
MODELS = ['v78_prior', 'v78_raw', 'v78_blend', 'v79_top', 'v80_recent']
TERMS = ['prediction', 'source_prediction', 'actual', 'total_error', 'regression_adjustment',
         'category_error', 'payoff_error', 'ordinary_contribution', 'gap_contribution', 'expiry_contribution']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def tag(at):
    return pd.Timestamp(at).tz_convert('America/New_York').strftime('%Y-%m-%d_%H%M')


class LockedSources:
    def __init__(self):
        self.files = {}
        self.manifests = {}
        for v in [78, 79, 80, 81]:
            p = ROOT / f'research/soxl_soxs_14d5pct_v{v}/evidence/delivery_manifest.json'
            self.files[str(p.relative_to(ROOT))] = sha(p)
            self.manifests[v] = read(p)

    def path(self, version, relative):
        m = self.manifests[version]
        p = ROOT / m['cache_root'] / relative
        h = sha(p)
        if h != m['cache_files_sha256'][relative]:
            raise ValueError(f'Mutated source: {p}')
        self.files[str(p.relative_to(ROOT))] = h
        return p

    def record(self, version, relative):
        return read(self.path(version, relative))


def decompose(probabilities, payoffs, prediction, truth, actual):
    q = np.asarray(probabilities, float)
    m = np.asarray(payoffs, float)
    if q.shape != (4,) or m.shape != (4,) or not np.isfinite(np.r_[q, m, prediction]).all():
        raise ValueError('Four finite probabilities/payoffs and a finite prediction required')
    if (q < 0).any() or abs(q.sum() - 1) > 1e-10:
        raise ValueError('Invalid probability distribution')
    g = float(q @ m)
    if truth not in range(4) or not np.isfinite(actual):
        raise ValueError('Known valid class and finite return required')
    category = float(g - m[truth])
    contributions = (q[1:] - np.eye(4)[truth, 1:]) * (m[1:] - m[0])
    out = dict(prediction=float(prediction), source_prediction=g, actual=float(actual),
               total_error=float(prediction - actual), regression_adjustment=float(prediction - g),
               category_error=category, payoff_error=float(m[truth] - actual),
               ordinary_contribution=float(contributions[0]), gap_contribution=float(contributions[1]),
               expiry_contribution=float(contributions[2]))
    assert abs(category - sum(contributions)) < 1e-12
    assert abs(out['total_error'] - sum(out[k] for k in ['regression_adjustment', 'category_error', 'payoff_error'])) < 1e-12
    return out


def actual_class(reason, gap):
    if reason == 'target':
        return 0
    if reason == 'risk_stop':
        return 2 if bool(gap) else 1
    if reason == 'expiry':
        return 3
    raise ValueError(f'Unscored reason: {reason}')


def cohort(table, name):
    flags = {
        'all_top': table.source_ready & table.paired_ready,
        'source_eligible': table.source_ready & table.paired_ready & table.source_eligible,
        'source_rejected': table.source_ready & table.paired_ready & ~table.source_eligible,
        'paired_accepted': table.paired_accepted,
        'paired_rejected': table.source_ready & table.paired_ready & table.source_eligible & ~table.paired_accepted,
        'absolute_pair_accepted': table.absolute_pair_accepted,
        'absolute_now_accepted': table.absolute_now_accepted,
    }
    return table[flags[name]]


def summary(table):
    known = table[table.known]
    daily = known.groupby('date')[TERMS].mean()
    all_prediction = table[table.paired_ready].groupby('date')[['prediction', 'source_prediction']].mean()
    out = dict(rows=len(table), dates=int(table.date.nunique()), known_rows=len(known),
               known_dates=len(daily), unknown_rows=int((~table.known).sum()),
               paired_mature_rows=int(table.paired_mature.sum()),
               all_date_equal_predictions=all_prediction.mean().to_dict() if len(all_prediction) else None,
               date_equal_terms=daily.mean().to_dict() if len(daily) else None,
               class_statistics={})
    for k, name in enumerate(CLASSES):
        if len(known):
            # Equal date mass, including dates without the class in frequency denominators.
            w = 1 / known.groupby('date').date.transform('count').to_numpy(float) / len(daily)
            mask = known.truth.to_numpy(int) == k
            mass = float(w[mask].sum())
            out['class_statistics'][name] = dict(
                predicted_probability=float(w @ known[f'q{k}'].to_numpy()), observed_frequency=mass,
                rows=int(mask.sum()), dates=int(known.loc[mask, 'date'].nunique()),
                conditional_predicted_payoff=float(w[mask] @ known.loc[mask, f'm{k}'].to_numpy() / mass) if mass else None,
                conditional_actual_payoff=float(w[mask] @ known.loc[mask, 'actual'].to_numpy() / mass) if mass else None,
                amplitude_contribution=float(w[mask] @ (known.loc[mask, f'm{k}'] - known.loc[mask, 'actual']).to_numpy()))
        else:
            out['class_statistics'][name] = None
    return out


def bootstrap(table, grid):
    daily = table[table.known].groupby('date')[TERMS].mean().reindex(grid)
    present = daily.notna().all(axis=1).to_numpy(float)
    values = daily.fillna(0).to_numpy(float)
    n = len(grid)
    rng = np.random.default_rng(707)
    starts = rng.integers(0, n, size=(5000, math.ceil(n / 10)))
    ix = ((starts[:, :, None] + np.arange(10)[None, None, :]) % n).reshape(5000, -1)[:, :n]
    denominator = present[ix].sum(axis=1)
    good = denominator > 0
    sampled = values[ix].sum(axis=1)[good] / denominator[good, None]
    return dict(draws=5000, unknown_draws=int((~good).sum()), full_grid_dates=n,
                nonempty_dates=int(present.sum()), intervals={name: np.quantile(sampled[:, k], [.025, .975]).tolist()
                if len(sampled) else None for k, name in enumerate(TERMS)})


def build():
    plan = read(SOURCE / 'protocol.json')
    locked = LockedSources()
    panel = pd.read_parquet(locked.path(81, 'semiconductor_panel.parquet'))
    cutoff = pd.Timestamp(plan['label_cutoff_exclusive_et'], tz='America/New_York').tz_convert('UTC')
    selection = panel[panel.date.between(plan['evaluation_start'], plan['evaluation_end'])]
    bank = {p['date'][:10]: p for p in locked.record(81, 'paired_registry.json')}
    records = {}; rows = []; comparisons = []
    for at, row in selection.iterrows():
        rel = 'training/' + tag(at) + '.json'
        old = {v: locked.record(v, rel) for v in [78, 79, 80, 81]}
        assert old[81]['parent_source_sha256'] == sha(locked.path(80, rel))
        assert old[80]['base_source_sha256'] == sha(locked.path(78, rel))
        assert old[80]['parent_source_sha256'] == sha(locked.path(79, rel))
        src = old[80]['events']['recent_ratio']; e = old[81]['events']['paired_state']
        date = str(row.date.date()); p = bank.get(date)
        item = dict(at=str(at), date=date, clock=row.clock, source_ready=bool(src['ready']),
                    paired_ready=bool(e['ready']), source_eligible=src['action'] in ETFS,
                    paired_accepted=e['action'] in ETFS,
                    absolute_pair_accepted=old[81]['events']['absolute_pair']['action'] in ETFS,
                    absolute_now_accepted=old[81]['events']['absolute_now']['action'] in ETFS,
                    paired_mature=bool(p and p['release'] and pd.Timestamp(p['release']) < cutoff),
                    known=False, symbol=src.get('top_symbol'), terminal_release=str(row.terminal_release),
                    visible_leadership=bool(row.trend5 > 0 and row.relative5 > 0))
        if src['ready'] and e['ready']:
            j = ETFS.index(src['top_symbol']); side = 'L' if j == 0 else 'S'
            q = np.asarray(src['probabilities'][j]); m = np.asarray(old[78]['distributions']['gap']['payoff_means'][j])
            g = float(q @ m)
            assert abs(g - src['values'][j]) < 1e-12 and abs(g - e['source_value']) < 1e-12
            prediction = float(e['values'][0]); item.update(prediction=prediction, source_prediction=g)
            item.update({f'q{k}': float(q[k]) for k in range(4)})
            item.update({f'm{k}': float(m[k]) for k in range(4)})
            sign = 1 if j == 0 else -1
            item.update(aligned_trend5=float(sign * row.trend5), aligned_current=float(sign * row.current),
                        relative5=float(row.relative5), sector_trend5=float(row.trend5))
            item['state'] = ('trend_positive' if sign * row.trend5 > 0 else 'trend_nonpositive') + '/' + ('current_positive' if sign * row.current > 0 else 'current_nonpositive')
            known = pd.notna(row.terminal_release) and row.terminal_release < cutoff and row[side + '_gap'] in range(4) and np.isfinite(row[side + '_log'])
            item['known'] = bool(known)
            if known:
                y = int(row[side + '_gap']); r = float(row[side + '_log'])
                item.update(truth=y, truth_name=CLASSES[y], **decompose(q, m, prediction, y, r))
            estimates = [old[78]['estimates'][key] for key in ['gap_prior', 'gap_raw', 'gap_blend']]
            estimates += [old[79]['events']['top_tilt'], src]
            for name, est in zip(MODELS, estimates):
                assert est['ready']
                pq = np.array(est['probabilities'][j]); pg = float(pq @ m)
                assert abs(pg - est['values'][j]) < 1e-12
                c = {k: item[k] for k in ['at', 'date', 'symbol', 'known', 'source_eligible', 'paired_accepted']}
                c.update(model=name, prediction=pg)
                c.update({f'q{k}': float(pq[k]) for k in range(4)})
                if known:
                    c.update(actual=r, brier=float(np.sum((pq - np.eye(4)[y]) ** 2)),
                             log_loss=float(-np.log(max(pq[y], 1e-8))), squared_return_error=float((pg - r) ** 2),
                             error=float(pg - r), category_error=float(pg - m[y]), payoff_error=float(m[y] - r))
                comparisons.append(c)
        records[str(at)] = old
        rows.append(item)
    table = pd.DataFrame(rows)
    comparison = pd.DataFrame(comparisons)
    cohorts = {name: summary(cohort(table, name)) for name in plan['cohorts']}
    grid = sorted(table.date.unique())
    intervals = {name: bootstrap(cohort(table, name), grid) for name in plan['cohorts']}
    by_month = {name: {month: summary(g) for month, g in cohort(table, name).groupby(cohort(table, name).date.str[:7])} for name in plan['cohorts']}
    states = {}; model_scores = {}
    for name in ['all_top', 'source_eligible', 'paired_accepted']:
        group = cohort(table, name)
        states[name] = {'direction_quadrants': {}, 'visible_leadership': {}}
        for a in ['trend_positive', 'trend_nonpositive']:
            for b in ['current_positive', 'current_nonpositive']:
                key = a + '/' + b; states[name]['direction_quadrants'][key] = summary(group[group.state == key])
        for key in [True, False]:
            states[name]['visible_leadership'][str(key).lower()] = summary(group[group.visible_leadership == key])
        keys = set(group['at'])
        model_scores[name] = {}
        for model in MODELS:
            s = comparison[(comparison.model == model) & comparison['at'].isin(keys)]
            fields = ['prediction', 'actual', 'brier', 'log_loss', 'squared_return_error', 'error', 'category_error', 'payoff_error']
            daily = s[s.known].groupby('date')[fields].mean()
            model_scores[name][model] = dict(rows=len(s), known_rows=int(s.known.sum()), known_dates=len(daily),
                                             date_equal_scores=daily.mean().to_dict() if len(daily) else None)
    entries = []
    for route in plan['executed_routes']:
        trades = pd.read_csv(locked.path(81, route + '_trades.csv'))
        for t in trades.itertuples():
            at = str(pd.Timestamp(t.decision_at)); row = table.set_index('at').loc[at]
            old = records[at]; src = old[80]['events']['recent_ratio']; j = ETFS.index(t.symbol)
            assert t.symbol == src['top_symbol'] and row.known
            y = actual_class(t.reason, t.gap_below_stop)
            assert y == row.truth
            r = float(np.log1p(t.net_return)); q = src['probabilities'][j]
            m = old[78]['distributions']['gap']['payoff_means'][j]
            predicted = src['values'][j] if route == 'source_recent' else old[81]['events'][route]['values'][0]
            d = decompose(q, m, predicted, y, float(row.actual))
            d.update(route=route, at=at, date=row.date, symbol=t.symbol, truth=y, truth_name=CLASSES[y],
                     net_return=float(t.net_return), actual_execution_log_return=r,
                     execution_difference=float(row.actual - r), total_execution_error=float(predicted - r),
                     gap_below_stop=bool(t.gap_below_stop), reason=t.reason)
            d.update({f'q{k}': float(q[k]) for k in range(4)})
            d.update({f'm{k}': float(m[k]) for k in range(4)})
            entries.append(d)
    executed = pd.DataFrame(entries)
    executed_summary = {}
    for route, g in executed.groupby('route'):
        executed_summary[route] = dict(entries=len(g), counts={c: int((g.truth_name == c).sum()) for c in CLASSES},
            entry_equal_terms=g[TERMS + ['actual_execution_log_return', 'execution_difference', 'total_execution_error']].mean().to_dict(),
            class_probabilities={c: float(g[f'q{k}'].mean()) for k, c in enumerate(CLASSES)},
            class_frequencies={c: float((g.truth_name == c).mean()) for c in CLASSES})
    result = dict(kind='post_outcome_algebraic_audit', label_cutoff=str(cutoff), cohorts=cohorts,
                  uncertainty=intervals, by_month=by_month, states=states, model_scores=model_scores,
                  executed=executed_summary, strategy_changed=False, new_cases_run=False, goal_complete=False)
    return table, comparison, executed, result, locked.files


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=CACHE / 'audit')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    registration = read(SOURCE / 'evidence/analysis_registration.json')
    for f, h in registration['code_sha256'].items():
        assert sha(ROOT / f) == h, f
    table, comparison, executed, result, files = build()
    for name, frame in [('candidate_rows', table), ('model_rows', comparison), ('executed_rows', executed)]:
        frame.to_csv(args.output / (name + '.csv'), index=False)
    dump(args.output / 'results.json', result)
    dump(args.output / 'inputs.json', files)
    print(json.dumps({'cohorts': result['cohorts'], 'executed': result['executed']}, indent=2))


if __name__ == '__main__':
    main()
