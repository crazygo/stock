import unittest,copy,json
from test_paths import fixture
from path_learning import *
from audit_absorption import audit_integration
from audit_paths import reference_values
from run_planning import json_ready

AT=pd.Timestamp('2026-01-05 14:25:30Z')


def run(bank,plan,weights=None,count=256,replication=0):
    members=np.arange(len(bank['blocks']));weights=np.ones(len(members))/len(members) if weights is None else np.array(weights,float)
    model,part,raw,integ,freq=integrate_absorbed(bank,members,weights,weights,AT,CLOCKS[0],np.array([100.,100.]),plan,count,replication)
    info={'ready':True,'replication':replication,'seed':part['seed'],'paths':len(model['selected_blocks']),'requested_paths':count,'partition':part,'raw':raw,'integration':integ,'monte_carlo_se':integ['standard_error'],'reason_frequencies':freq}
    values,n=audit_integration(bank,members,CLOCKS[0],AT,[100.,100.],plan,[{'probability':weights}]*2,'conditional',info,model,count,replication)
    np.testing.assert_allclose(values,[raw['SOXL'],raw['SOXS']],atol=1e-12)
    return model,info,n


class AbsorptionTests(unittest.TestCase):
    def test_all_resolved_exact_no_numerical_error(self):
        b,p,_=fixture();b['ohlc'][:,4,0,1]=110;b['ohlc'][:,4,1,2]=90
        m,i,n=run(b,p);self.assertEqual(i['paths'],0);self.assertIsNone(m['result']);self.assertEqual(n,10)
        np.testing.assert_array_equal(i['monte_carlo_se'],[0.,0.]);self.assertEqual(i['partition']['unresolved_probability'],0.)

    def test_all_unresolved_matches_ordinary_draws_and_returns(self):
        b,p,_=fixture();m,i,_=run(b,p);w=np.ones(5)/5;s,_=draw_blocks(np.arange(5),w,w,AT,CLOCKS[0],4,256)
        np.testing.assert_array_equal(m['selected_blocks'],s)
        expected=summarize(simulate(b,s,CLOCKS[0],[100.,100.],p))
        for k,v in expected.items():self.assertAlmostEqual(i['raw'][k],v)
        self.assertEqual(i['partition']['unresolved_probability'],1.)

    def test_one_side_resolved_stays_in_unresolved_stratum(self):
        b,p,_=fixture();b['ohlc'][:,4,0,1]=110;m,i,_=run(b,p)
        self.assertFalse(m['resolved'].any());self.assertEqual(i['partition']['unresolved_probability'],1.)
        self.assertTrue((m['result']['reason'][:,0]==1).all());self.assertTrue((m['result']['reason'][:,1]==3).all())

    def test_mixed_exact_finite_tree_identity(self):
        b,p,_=fixture();b['blocks']=np.array([[0,1],[2,3],[4,5]]);b['ohlc'][0,4,0,1]=110;b['ohlc'][0,4,1,2]=90
        b['ohlc'][2,4,0,1]=110;b['ohlc'][5,7,:,1]=110;w=np.array([.2,.3,.5]);m,i,_=run(b,p,w)
        pairs=np.array([[a,c] for a in range(3) for c in range(3)]);vals=reference_values(b,pairs,CLOCKS[0],[100.,100.],p)
        expected=sum(w[a]*w[c]*payoff_moments(vals['log_returns'][j:j+1])[0] for j,(a,c) in enumerate(pairs))
        r=m['resolved'];short=m['partition_result'];exact=w[r]@payoff_moments(short['log_returns'][r]);mass=sum(w[~r])
        remainder=sum(w[a]/mass*w[c]*payoff_moments(vals['log_returns'][j:j+1])[0] for j,(a,c) in enumerate(pairs) if not r[a])
        np.testing.assert_allclose(expected,exact+mass*remainder,atol=1e-14)
        self.assertTrue(r.any() and (~r).any());self.assertTrue(set(m['selected_blocks'][:,0]).isdisjoint(np.where(r)[0]))

    def test_partition_honors_prefix_halfday_and_split_dividends(self):
        b,p,_=fixture();p['first_offset']=12;p['bar_counts'][1]=42
        b['ohlc'][0,5,:,1]=110;b['ohlc'][1,50,:,1]=110;b['split_ratio'][1]=10.;b['dividend_ratio'][1]=.02
        m,i,_=run(b,p);self.assertFalse(m['resolved'][0]);self.assertTrue((m['partition_result']['reason'][0]==3).all())
        b['ohlc'][1,30,:,1]=106;m,i,_=run(b,p);self.assertTrue(m['resolved'][0]);np.testing.assert_allclose(m['partition_result']['cash'][0],2.)

    def test_resolved_contract_ignores_all_later_cash_and_gaps(self):
        b,p,_=fixture();b['blocks']=np.array([[0,1],[2,3],[4,5]]);b['ohlc'][0,4,:,1]=110
        b['ohlc'][2,0]=[[80,85,75,81]]*2;b['dividend_ratio'][2]=.5
        m,i,_=run(b,p,[.5,.25,.25]);short=m['partition_result'];r=short['log_returns'][0]
        full=reference_values(b,np.array([[0,1],[0,2]]),CLOCKS[0],[100.,100.],p)
        np.testing.assert_allclose(full['log_returns'],np.tile(r,(2,1)),atol=1e-14)

    def test_replication_reproducible_and_distinct(self):
        b,p,_=fixture();a,_,_=run(b,p,replication=100);b1,_,_=run(b,p,replication=100);c,_,_=run(b,p,replication=101)
        np.testing.assert_array_equal(a['selected_blocks'],b1['selected_blocks']);self.assertFalse(np.array_equal(a['selected_blocks'],c['selected_blocks']))

    def test_partition_does_not_short_exit_original_horizon(self):
        b,p,_=fixture();b['blocks']=np.array([[0,1],[2,3],[4,5]]);b['ohlc'][3,4,:,1]=110
        m,i,_=run(b,p,[1.,0.,0.]);self.assertTrue((m['result']['reason']==3).all());self.assertTrue((m['result']['exit_day']==3).all())

    def test_no_invalid_budget_or_shortened_contract(self):
        b,p,_=fixture()
        with self.assertRaises(ValueError):run(b,p,count=255)
        p['dates']=p['dates'][:2];p['bar_counts']=p['bar_counts'][:2]
        with self.assertRaises(ValueError):run(b,p)

if __name__=='__main__':unittest.main()
