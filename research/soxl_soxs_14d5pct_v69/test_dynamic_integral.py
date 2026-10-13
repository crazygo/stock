"""Original-horizon dynamic integral boundaries and independent path reconstruction."""
import unittest,copy
from test_state_paths import fixture as state_fixture,scalar_state,scalar_walk
from state_paths import integrate_dynamic,dynamic_blocks,StateUnavailable
from path_learning import *
from audit_state_paths import audit_dynamic_integral
from audit_paths import reference_values
from sampling import uniforms
AT=pd.Timestamp('2026-01-05 14:25:30Z')


def fixture():
    bank,ref,h,q,_,_,_,plan=state_fixture();n=len(bank['states']);members=np.arange(100);w=np.arange(1,101,dtype=float);w/=sum(w)
    bank.update(ohlc=np.full((n,78,2,4),100.),quotes=np.full((n,8,2),100.),previous_close=np.full((n,2),100.),split_ratio=np.ones((n,2)),dividend_ratio=np.zeros((n,2)))
    context={'reference':ref,'queries':{'history':h[None,:],'quotes':q[None,:]},'positions':{AT.value:0}};plan={**plan,'dates':[0,1,2,3],'bar_counts':[78,42,78,78],'expiry_at':pd.Timestamp('2026-01-08 21:00Z')}
    return bank,members,w,context,plan


def run(bank,members,w,context,plan,count=256,replication=0):
    model,part,raw,integ,freq=integrate_dynamic(bank,members,w,AT,CLOCKS[0],[100.,100.],plan,context,count,replication);t=model['state_path']
    state={'ready':True,'paths':len(model['selected_blocks']),'minimum_effective_dates':float(t['effective_dates'][:,1:].min()) if t is not None else None,'mean_absolute_state_change':np.abs(t['states_before_blocks'][:,1:]-t['states_before_blocks'][:,:1]).mean(axis=(0,1)).tolist() if t is not None else [0.,0.,0.]}
    info={'ready':True,'state_transition':state,'partition':part,'raw':raw,'integration':integ,'reason_frequencies':freq,'seed':part['seed'],'replication':replication,'paths':len(model['selected_blocks']),'requested_paths':count,'monte_carlo_se':integ['standard_error']}
    expected,_=audit_dynamic_integral(bank,members,CLOCKS[0],AT,[100.,100.],plan,[{'probability':w}],info,model,context,count,replication);np.testing.assert_allclose(expected,[raw['SOXL'],raw['SOXS']],atol=1e-12);return model,info


class DynamicIntegralTests(unittest.TestCase):
    def test_all_resolved_exact_without_state_tail(self):
        b,m,w,c,p=fixture();b['ohlc'][:,20,0,1]=110;b['ohlc'][:,20,1,2]=90;model,info=run(b,m,w,c,p)
        self.assertEqual(info['paths'],0);self.assertIsNone(model['state_path']);self.assertEqual(info['partition']['unresolved_probability'],0.);np.testing.assert_array_equal(info['monte_carlo_se'],[0.,0.])

    def test_all_unresolved_is_original_dynamic_draw(self):
        b,m,w,c,p=fixture();model,info=run(b,m,w,c,p);seed=info['seed'];q=c['queries'];expected=dynamic_blocks(m,w,uniforms(seed,2,256),q['history'][0],q['quotes'][0],c['reference'],b,CLOCKS[0],p)
        np.testing.assert_array_equal(model['selected_blocks'],expected['selected_blocks']);self.assertEqual(info['partition']['unresolved_probability'],1.)
        self.assertTrue((model['result']['exit_day']==3).all())

    def test_one_sided_resolution_keeps_full_horizon(self):
        b,m,w,c,p=fixture();b['ohlc'][:,20,0,1]=110;model,info=run(b,m,w,c,p);self.assertFalse(model['resolved'].any());self.assertTrue((model['result']['reason'][:,0]==1).all());self.assertTrue((model['result']['reason'][:,1]==3).all());self.assertTrue((model['result']['exit_day'][:,1]==3).all())

    def test_mixed_strata_have_no_resolved_tail_entries(self):
        b,m,w,c,p=fixture();b['ohlc'][::7,20,0,1]=110;b['ohlc'][::7,20,1,2]=90;b['ohlc'][::5,20,0,1]=110
        model,info=run(b,m,w,c,p);self.assertTrue(model['resolved'].any() and (~model['resolved']).any());self.assertFalse(np.isin(model['selected_blocks'][:,0],m[model['resolved']]).any());self.assertGreater(info['partition']['unresolved_probability'],0)

    def test_finite_two_block_tree_total_expectation(self):
        b,m,w,c,p=fixture();b['ohlc'][::7,20,0,1]=110;b['ohlc'][::7,20,1,2]=90;model,info=run(b,m,w,c,p)
        pairs=np.array([[a,d] for a in m for d in m]);values=reference_values(b,pairs,CLOCKS[0],[100.,100.],p);g=payoff_moments(values['log_returns']).reshape(len(m),len(m),4);conditional=[];q=c['queries'];source=b['states'][b['blocks'][m,0],0,:3]
        for i in range(len(m)):
            first=np.zeros(len(m));first[i]=1.;_,_,_,h,_=scalar_walk(m,first,[.5],q['history'][0],q['quotes'][0],c['reference'],b,CLOCKS[0],p);conditional.append(kernel(source,scalar_state(h),'conditional')['probability'])
        conditional=np.array(conditional);all_mean=np.sum(w[:,None,None]*conditional[:,:,None]*g,axis=(0,1));r=model['resolved'];u=~r;mass=w[u].sum();part=np.array(info['partition']['exact_moments'])+mass*np.sum((w[u]/mass)[:,None,None]*conditional[u,:,None]*g[u],axis=(0,1));np.testing.assert_allclose(part,all_mean,atol=1e-13)
        for i in np.where(r)[0]:np.testing.assert_allclose(g[i],np.broadcast_to(g[i,0],g[i].shape),atol=1e-13)

    def test_replicates_reproducible_and_distinct(self):
        args=fixture();a,_=run(*args,replication=101);b,_=run(*args,replication=101);c,_=run(*args,replication=102);np.testing.assert_array_equal(a['selected_blocks'],b['selected_blocks']);self.assertFalse(np.array_equal(a['selected_blocks'],c['selected_blocks']))

    def test_halfday_prefix_dividends_and_split_units(self):
        b,m,w,c,p=fixture();b['ohlc'][::3,60,:,1]=110;b['split_ratio'][::5]=10.;b['dividend_ratio'][::5]=.02;model,info=run(b,m,w,c,p);self.assertTrue(np.isfinite(model['result']['net']).all());self.assertTrue((model['result']['cash']>0).any())

    def test_unsupported_tail_invalidates_entire_forecast(self):
        b,m,w,c,p=fixture();m=m[:19];w=np.ones(19)/19
        with self.assertRaises(StateUnavailable):integrate_dynamic(b,m,w,AT,CLOCKS[0],[100.,100.],p,c,256)

    def test_invalid_budget_and_short_contract_rejected(self):
        b,m,w,c,p=fixture()
        with self.assertRaises(ValueError):integrate_dynamic(b,m,w,AT,CLOCKS[0],[100.,100.],p,c,255)
        p['dates']=p['dates'][:2];p['bar_counts']=p['bar_counts'][:2]
        with self.assertRaises(ValueError):integrate_dynamic(b,m,w,AT,CLOCKS[0],[100.,100.],p,c,256)

if __name__=='__main__':unittest.main()
