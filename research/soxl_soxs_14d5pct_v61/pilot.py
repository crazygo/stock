"""Frozen numerical experiment: repeated integration, without future labels/capital."""
import json
from datetime import datetime, timezone
from threadpoolctl import threadpool_limits
from path_learning import *
from audit_paths import check_forecast
from run_planning import json_ready


def pilot():
    registration = read(SOURCE/'evidence/pilot_registration.json')
    for f, h in registration['source_sha256'].items():
        assert sha(core.ROOT/f) == h, f
    source = Path(registration['inputs'])
    manifest = read(source/'input_manifest.json')
    assert sha(source/'input_manifest.json') == registration['input_manifest_sha256']
    for f, h in manifest['derived_files_sha256'].items():
        assert sha(source/f) == h
    panel = pd.read_parquet(source/'activity_panel.parquet')
    bank = dict(np.load(source/'path_bank.npz'))
    market = core.load_market()
    assert market.manifest == manifest['full_market']
    market = restrict_market(market, pd.Timestamp(manifest['observation_cutoff_exclusive']))
    protocol = read(SOURCE/'protocol.json')['numerical_pilot']
    records = []
    canonical = lambda x: json.loads(json.dumps(json_ready(x), default=str))
    with threadpool_limits(limits=1):
        for at, row in panel[panel.date.astype(str).isin(protocol['dates'])].iterrows():
            for kind in VARIANTS:
                first_model, info, raw = forecast(bank, market, panel, at, kind)
                assert raw is not None
                check_forecast(bank, market, panel, at, kind, canonical(info), first_model)
                members = np.array(info['members'])
                estimates = {'rqmc': [], 'iid': []}
                reported = {'rqmc': [], 'iid': []}
                for replication in range(protocol['independent_estimators']):
                    selected, seed = draw_blocks(members, info['kernels']['first']['probability'],
                        info['kernels']['later']['probability'], at, row.clock,
                        len(info['plan']['dates']), replication=replication)
                    pair = simulate(bank, selected, row.clock, np.array([q['raw_close'] for q in info['query_quote']]), info['plan'])
                    estimates['rqmc'].append(pair['log_returns'].mean(axis=0))
                    reported['rqmc'].append(integration_statistics(pair['log_returns'])['mean_covariance'])
                    # Separate, predeclared independent IID estimators; same 2048-path budget.
                    u = np.random.default_rng(np.random.SeedSequence([seed, replication, 991])).random(selected.shape)
                    for column in range(u.shape[1]):
                        weights = info['kernels']['first' if column == 0 else 'later']['probability']
                        cdf = np.cumsum(weights); cdf[-1] = 1.
                        selected[:, column] = members[np.searchsorted(cdf, u[:, column], side='right')]
                    pair = simulate(bank, selected, row.clock, np.array([q['raw_close'] for q in info['query_quote']]), info['plan'])
                    estimates['iid'].append(pair['log_returns'].mean(axis=0))
                    reported['iid'].append(np.cov(pair['log_returns'], rowvar=False)/DRAWS)
                empirical = {k: np.cov(values, rowvar=False) for k, values in estimates.items()}
                records.append({'at': at, 'kind': kind, 'source_blocks': len(members),
                    'estimates': estimates, 'reported_covariances': reported,
                    'empirical_covariance': empirical,
                    'empirical_trace_ratio': np.trace(empirical['rqmc'])/np.trace(empirical['iid'])})
            print('numerical pilot', at, flush=True)
    summary = {}
    for kind in VARIANTS:
        rows = [r for r in records if r['kind'] == kind]
        variance = {k: float(np.mean([np.trace(r['empirical_covariance'][k]) for r in rows])) for k in ['rqmc', 'iid']}
        summary[kind] = {'queries': len(rows), 'mean_empirical_trace': variance,
                        'pooled_variance_ratio': variance['rqmc']/variance['iid'],
                        'median_query_variance_ratio': float(np.median([r['empirical_trace_ratio'] for r in rows]))}
    passed = all(summary[k]['queries'] == 24 for k in VARIANTS) and summary['conditional']['pooled_variance_ratio'] < 1
    output = {'finished_at': datetime.now(timezone.utc).isoformat(), 'passed': passed,
              'summary': summary, 'records': records, 'new_capital_outcomes_seen': False,
              'future_labels_used': False, 'return_acceptance_established': False,
              'registration_sha256': sha(SOURCE/'evidence/pilot_registration.json')}
    dump(SOURCE/'evidence/pilot.json', output)
    print({'passed': passed, 'summary': summary}, flush=True)


if __name__ == '__main__':
    pilot()
