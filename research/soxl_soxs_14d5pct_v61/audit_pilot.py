"""Revalue every pilot estimator using the independent full-horizon contract audit."""
from threadpoolctl import threadpool_limits
from path_learning import *
from audit_paths import reference_values
from scipy.stats import qmc


def audit():
    registration = read(SOURCE/'evidence/pilot_registration.json')
    result = read(SOURCE/'evidence/pilot.json')
    assert result['registration_sha256'] == sha(SOURCE/'evidence/pilot_registration.json')
    bank = dict(np.load(Path(registration['inputs'])/'path_bank.npz'))
    old = core.ROOT/'.cache/soxl_soxs_14d5pct_v60/runs/development_20261013'
    old_v = read(old/'verification.json')
    assert old_v['passed'] and old_v['results_sha256'] == sha(old/'results.json')
    # Same immutable source bank, historical members, state weights and quotes.
    for name, values in bank.items():
        np.testing.assert_array_equal(values, np.load(old/'path_bank.npz')[name])
    expected_times = {pd.Timestamp(day+' '+clock, tz=core.ET).tz_convert('UTC')
        for day in read(SOURCE/'protocol.json')['numerical_pilot']['dates'] for clock in CLOCKS}
    assert len(result['records']) == 48
    assert {(pd.Timestamp(r['at']),r['kind']) for r in result['records']} == {(t,k) for t in expected_times for k in VARIANTS}
    summary = {}; n = 0
    with threadpool_limits(limits=1):
        for row in result['records']:
            at = pd.Timestamp(row['at']); kind = row['kind']; clock = at.tz_convert(core.ET).strftime('%H:%M:%S')
            old_record_path = old/'training'/(tag(at)+'.json')
            receipt = read(core.ROOT/'research/soxl_soxs_14d5pct_v60/evidence/delivery_manifest.json')['runs']['development_20261013']['training_records_sha256']
            assert sha(old_record_path) == receipt[old_record_path.name]
            f = read(old_record_path)['fits'][kind]; members = np.array(f['members']); plan = f['plan']; seed=1061+at.tz_convert(core.ET).date().toordinal()*16+CLOCKS.index(clock)
            assert len(members) == row['source_blocks']
            dimension=(len(plan['dates'])+1)//2
            for replica in range(16):
                children=np.random.SeedSequence([seed,replica]).spawn(8)
                sobol=np.vstack([qmc.Sobol(dimension,scramble=True,bits=30,rng=np.random.default_rng(child)).random_base2(8) for child in children])
                iid=np.random.default_rng(np.random.SeedSequence([seed,replica,991])).random((2048,dimension))
                for method,u in [('rqmc',sobol),('iid',iid)]:
                    selections=[]
                    for column in range(dimension):
                        p=f['kernels']['first' if column==0 else 'later']['probability'];cum=np.cumsum(p);cum[-1]=1.
                        selections.append(members[np.searchsorted(cum,u[:,column],side='right')])
                    values=reference_values(bank,np.array(selections).T,clock,np.array([q['raw_close'] for q in f['query_quote']]),plan)['log_returns']
                    np.testing.assert_allclose(row['estimates'][method][replica],values.mean(axis=0),rtol=1e-10,atol=1e-11)
                    observations=np.array([v.mean(axis=0) for v in np.split(values,8)]) if method=='rqmc' else values
                    covariance=np.cov(observations,rowvar=False)/len(observations)
                    np.testing.assert_allclose(row['reported_covariances'][method][replica],covariance,rtol=1e-9,atol=1e-12)
                    n+=4096
            cov={k:np.cov(row['estimates'][k],rowvar=False) for k in ['rqmc','iid']}
            for k in cov:np.testing.assert_allclose(row['empirical_covariance'][k],cov[k],rtol=1e-12)
            np.testing.assert_allclose(row['empirical_trace_ratio'],np.trace(cov['rqmc'])/np.trace(cov['iid']))
            print('pilot independent values',at,kind,flush=True)
    for kind in VARIANTS:
        rows=[r for r in result['records'] if r['kind']==kind]
        means={k:float(np.mean([np.trace(r['empirical_covariance'][k]) for r in rows])) for k in ['rqmc','iid']}
        summary[kind]={'queries':len(rows),'mean_empirical_trace':means,'pooled_variance_ratio':means['rqmc']/means['iid'],'median_query_variance_ratio':float(np.median([r['empirical_trace_ratio'] for r in rows]))}
    assert summary==result['summary']
    assert result['passed']==(summary['conditional']['pooled_variance_ratio']<1)
    out={'passed':True,'contracts_independently_valued':n,'summary':summary,'pilot_sha256':sha(SOURCE/'evidence/pilot.json'),'new_capital_outcomes_seen':False}
    dump(SOURCE/'evidence/pilot_audit.json',out);print(out,flush=True)


if __name__=='__main__':audit()
