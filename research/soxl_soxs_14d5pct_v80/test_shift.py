import unittest
from shift_learning import *
import reference_shift as ref


class ShiftTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(801);self.X=rng.normal(size=(80,14));self.Z=rng.normal(size=(32,14))+.4
        self.u=np.linspace(.4,1.2,80);self.v=np.ones(32)

    def test_gradient_hessian_and_independent_optimizer(self):
        m=domain_fit(self.X,self.Z,self.u,self.v);A,y,w,_,_=ref.design(self.X,self.Z,self.u,self.v)
        beta=np.asarray(m['beta']);loss,g,H=domain_derivatives(beta,A,y,w)
        np.testing.assert_allclose(beta,ref.solve(A,y,w),atol=1e-7,rtol=1e-6)
        np.testing.assert_allclose(g,ref.objective(beta,A,y,w)[1],atol=1e-12)
        eps=1e-5
        for j in range(15):
            d=np.eye(15)[j]*eps
            np.testing.assert_allclose(H[:,j],(ref.objective(beta+d,A,y,w)[1]-ref.objective(beta-d,A,y,w)[1])/(2*eps),atol=1e-8)
        self.assertLess(max(abs(g)),1e-7);self.assertTrue(np.linalg.eigvalsh(H).min()>=10-1e-10)

    def test_identical_domains_identity_and_constant_field(self):
        self.X[:,3]=2
        m=domain_fit(self.X,self.X,self.u,self.u)
        np.testing.assert_allclose(m['beta'],0,atol=1e-12);np.testing.assert_allclose(m['ratios'],1,atol=1e-12)
        self.assertEqual(m['scale'][3],1)

    def test_repeated_clocks_do_not_inflate_domain_mass(self):
        a=domain_fit(self.X,self.Z,self.u,self.v)
        b=domain_fit(np.repeat(self.X,2,axis=0),np.repeat(self.Z,3,axis=0),np.repeat(self.u/2,2),np.repeat(self.v/3,3))
        np.testing.assert_allclose(a['beta'],b['beta'],atol=1e-9,rtol=1e-8)
        self.assertAlmostEqual(sum(b['old_domain_weights']),20);self.assertAlmostEqual(sum(b['new_domain_weights']),20)

    def test_units_and_translation_do_not_change_ratios(self):
        a=domain_fit(self.X,self.Z,self.u,self.v);scales=np.arange(1,15)*20;offset=np.arange(14)*9
        b=domain_fit(self.X*scales+offset,self.Z*scales+offset,self.u,self.v)
        np.testing.assert_allclose(a['ratios'],b['ratios'],atol=1e-9,rtol=1e-8)

    def test_shift_gives_larger_weights_to_matching_old_states(self):
        X=np.zeros((40,14));X[:,0]=np.linspace(-2,2,40);Z=np.zeros((20,14));Z[:,0]=2
        d=domain_fit(X,Z,np.ones(40),np.ones(20));self.assertGreater(d['ratios'][-1],d['ratios'][0])
        self.assertTrue(((d['ratios']>=.2-1e-12)&(d['ratios']<=5+1e-12)).all())

    def test_weight_mass_date_ess_and_duplicate_clocks(self):
        dates=[pd.Timestamp('2025-01-02 15:00Z'),pd.Timestamp('2025-01-03 15:00Z')];members=[[t,0] for t in dates]
        a,e=reweight([2.,3.],[5.,.2],members);b,f=ref.weights([2.,3.],[5.,.2],members)
        np.testing.assert_allclose(a,b);self.assertAlmostEqual(e,f);self.assertAlmostEqual(sum(a),5)
        c,g=reweight([1.,1.,1.5,1.5],[5.,5.,.2,.2],[members[0]]*2+[members[1]]*2)
        self.assertAlmostEqual(e,g)

    def test_window_uses_no_outcomes_and_rejects_missing_recent_date(self):
        sessions=pd.bdate_range('2025-01-01',periods=45);reg=[];history={}
        for d in sessions:
            t=pd.Timestamp(str(d.date())+' 09:25:30',tz=core.ET).tz_convert('UTC')
            reg.append({'date':d,'feature_ready':True,'members':[t],'terminal_release':pd.NaT})
            history[t]={'events':{'gap_blend':{'ready':True}}}
        at=pd.Timestamp(str(sessions[-1].date())+' 09:25:30',tz=core.ET).tz_convert('UTC')
        a=context_windows(reg,sessions,history,at);self.assertTrue(a['recent']['ready']);self.assertEqual(a['recent']['dates'],list(sessions[-21:-1]))
        for r in reg:r['terminal_release']=pd.Timestamp('2099-01-01');r['future_success']=True
        self.assertEqual(a,context_windows(reg,sessions,history,at));reg[-2]['feature_ready']=False
        b=context_windows(reg,sessions,history,at);self.assertFalse(b['recent']['ready']);self.assertTrue(b['lagged']['ready'])
        reg[-1]['feature_ready']=False
        self.assertEqual(b,context_windows(reg,sessions,history,at))

    def test_calibration_cannot_change_support_or_promote_source_cash(self):
        p=np.array([[.5,.3,.2,0.],[.5,.3,.2,0.]])
        old={'events':{'gap_blend':{'ready':True,'action':'CASH'}},'estimates':{'gap_blend':{'values':[.1,.2],'probabilities':p}},
            'distributions':{'gap':{'payoff_means':np.ones((2,4))}}}
        daily={'ready':True,'models':{'recent':{'beta':[1.,2.,3.,4.]}}};e=prior.estimate(old,daily,'recent')
        self.assertEqual(e['action'],'CASH');np.testing.assert_array_equal(e['probabilities'][:,3],0)
        old['events']['gap_blend']['action']='SOXS';e=prior.estimate(old,daily,'recent');self.assertEqual(e['action'],'SOXS')
        self.assertFalse(prior.fit(p,[3,1],[1,1])['ready'])

    def test_invalid_contexts_rejected(self):
        for X,Z,u,v in [(self.X[:0],self.Z,self.u[:0],self.v),(self.X,self.Z,self.u*0,self.v),(self.X[:,:13],self.Z,self.u,self.v)]:
            with self.assertRaises(ValueError):domain_fit(X,Z,u,v)


if __name__=='__main__':unittest.main()
