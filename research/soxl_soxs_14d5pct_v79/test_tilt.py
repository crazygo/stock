import copy,unittest
from tilt_learning import *
import reference_tilt as ref


def synthetic():
    parents=[];history={};rows=[]
    for i,d in enumerate(pd.bdate_range('2024-01-01',periods=80)):
        ids=[pd.Timestamp(str(d.date())+' '+c,tz=core.ET).tz_convert('UTC') for c in CLOCKS[:2]]
        parents.append({'date':d,'members':ids,'feature_ready':True,'release':ids[-1]+pd.Timedelta(days=13)})
        for at in ids:
            eligible=i%3!=0;info={'ready':True,'values':[.01,-.01],'probabilities':[[.65,.25,.08,.02],[.3,.5,.15,.05]]}
            history[at]={'estimates':{'gap_blend':info},'events':{'gap_blend':{'ready':True,'action':'SOXL' if eligible else 'CASH'}},
                'distributions':{'gap':{'payoff_means':[[.05,-.06,-.12,0],[.05,-.06,-.12,0]]}}}
            rows.append({'at':at,'L_gap':i%4,'S_gap':(i+1)%4})
    return pd.DataFrame(rows).set_index('at'),parents,history


class TiltTests(unittest.TestCase):
    def test_gradient_hessian_and_independent_solver(self):
        rng=np.random.default_rng(79);P=rng.dirichlet(np.ones(4),120);y=rng.integers(4,size=120);w=np.ones(120);b=np.array([.1,-.2,0,.05])
        loss,g,H=derivatives(b,P,y,w)
        v,gg=ref.objective(b,P,y,w);self.assertAlmostEqual(v,loss);np.testing.assert_allclose(g,gg,atol=1e-12)
        for j in range(4):
            delta=np.eye(4)[j]*1e-5;a=derivatives(b+delta,P,y,w);c=derivatives(b-delta,P,y,w)
            self.assertAlmostEqual(g[j],(a[0]-c[0])/2e-5,places=7);np.testing.assert_allclose(H[:,j],(a[1]-c[1])/2e-5,atol=1e-7)
        model=fit(P,y,w);np.testing.assert_allclose(model['beta'],ref.solve(P,y,w),atol=1e-7,rtol=1e-6)
        self.assertAlmostEqual(sum(model['beta']),0,places=10)

    def test_identity_and_zero_support(self):
        P=np.array([[.7,.3,0,0],[.2,.8,0,0]]);np.testing.assert_allclose(tilt(P,[0]*4),P)
        q=tilt(P,[2,-1,30,-50]);np.testing.assert_allclose(q.sum(1),1);np.testing.assert_array_equal(q[:,2:],0)
        self.assertFalse(fit(P,[0,2],[1,1])['ready']);self.assertTrue(fit(P,[0,1],[1,1])['ready'])

    def test_invalid_inputs(self):
        for P in [[[.5,.5,.1,0]],[[np.nan,.5,.5,0]],[[-.1,.1,.5,.5]]]:
            with self.assertRaises(ValueError):tilt(P,[0]*4)
        with self.assertRaises(ValueError):fit([[.25]*4],[-1],[1])
        with self.assertRaises(ValueError):fit([[.25]*4],[0],[0])

    def test_population_weights_and_date_gaps(self):
        panel,parents,history=synthetic()
        for mode in MODES:
            P,y,w,meta=samples(panel,parents,history,mode);Q,z,v,other=ref.data(panel,parents,history,mode)
            np.testing.assert_array_equal(P,Q);np.testing.assert_array_equal(y,z);np.testing.assert_allclose(w,v,atol=1e-13)
            self.assertEqual(meta['real_dates'],53 if mode=='eligible' else 80);self.assertAlmostEqual(sum(w),meta['date_ess'])
        _,_,w,m=samples(panel,parents,history,'eligible')
        self.assertAlmostEqual(w[2]/w[0],2**(1/63));self.assertAlmostEqual(w[4]/w[2],2**(2/63))

    def test_clocks_do_not_multiply_date_mass(self):
        p,parents,h=synthetic();a=samples(p,parents,h,'eligible');one=[dict(q,members=q['members'][:1]) for q in parents];b=samples(p,one,h,'eligible')
        self.assertEqual(a[3]['date_ess'],b[3]['date_ess']);np.testing.assert_allclose(fit(*a[:3])['beta'],fit(*b[:3])['beta'],atol=1e-10)

    def test_history_maturity_and_future_labels(self):
        panel,reg,h=synthetic();at=pd.Timestamp('2024-06-01',tz='UTC');parents,fb=mature_prefix(reg,at)
        a=daily_models(panel,parents,fb,h,at);self.assertTrue(a['ready'])
        future=dict(reg[-1],date=pd.Timestamp('2025-01-01'),release=pd.Timestamp('2025-01-20',tz='UTC'))
        pp,ff=mature_prefix(reg+[future],at);b=daily_models(panel,pp,ff,h,at)
        np.testing.assert_array_equal(a['models']['eligible']['beta'],b['models']['eligible']['beta'])
        reg[50]['release']=None;pp,ff=mature_prefix(reg,at);self.assertEqual(len(pp),50);self.assertFalse(daily_models(panel,pp,ff,h,at)['ready'])

    def test_no_reselection_no_cash_promotion(self):
        p,parents,h=synthetic();old=copy.deepcopy(next(iter(h.values())));daily={'ready':True,'models':{'eligible':{'beta':[0]*4}}}
        self.assertEqual(estimate(old,daily,'eligible')['action'],'CASH')
        old['events']['gap_blend']['action']='SOXL';old['estimates']['gap_blend']['values']=[.01,-.01]
        old['estimates']['gap_blend']['probabilities']=[[.1,.9,0,0],[.9,.1,0,0]]
        self.assertEqual(estimate(old,daily,'eligible')['action'],'CASH')
        daily['ready']=False;self.assertFalse(estimate(old,daily,'eligible')['ready'])

    def test_readiness_uses_eligible_dates_not_rows(self):
        p,parents,h=synthetic()
        for d in parents[10:]:
            for at in d['members']:h[at]['events']['gap_blend']['action']='CASH'
        model=daily_models(p,parents,{'expired_unresolved_prefix':False},h,p.index[-1]);self.assertFalse(model['ready'])


if __name__=='__main__':unittest.main()
