import unittest
from class_learning import *
from reference_class import setup,evaluate,newton


def fixture(n=125,classes=4):
    rng=np.random.default_rng(78);X=rng.normal(size=(n*6,13));y=rng.integers(classes,size=n*6)
    outcomes=np.array([.048,-.064,-.11,.002])[y]+rng.normal(0,.001,len(y))
    groups=np.repeat(np.arange(6),n);dw=2.**(-np.arange(n)[::-1]/126)
    return X,y,outcomes,groups,dw


class ClassTests(unittest.TestCase):
    def test_independent_statistics_gradient_and_newton(self):
        args=fixture();m=fit_distribution(*args,4);ref=setup(*args[:4],m['date_weights'],4)
        for k in ['center','scale','weights','prior','payoff_means','pooled_prior','pooled_payoff_sum','own_class_mass','own_payoff_sum']:
            np.testing.assert_allclose(m[k],ref[k],atol=1e-12,rtol=1e-11)
        loss,g=evaluate(m['coefficients'],ref);self.assertLess(np.max(abs(g)),1e-6)
        self.assertAlmostEqual(m['objective'],loss,places=10)
        B,_=newton(ref);np.testing.assert_allclose(m['coefficients'],B,atol=1e-7,rtol=1e-6)

    def test_analytic_gradient_finite_differences(self):
        args=fixture(n=20);m=fit_distribution(*args,4);r=setup(*args[:4],m['date_weights'],4)
        B=np.random.default_rng(9).normal(0,.02,m['coefficients'].shape);y=np.argmax(r['Y'],axis=1)
        value,g=objective(B.ravel(),r['A'],y,r['weights'],r['offset']);eps=1e-5
        for j in [0,1,5,17,38,75]:
            plus=B.ravel().copy();minus=plus.copy();plus[j]+=eps;minus[j]-=eps
            numeric=(objective(plus,r['A'],y,r['weights'],r['offset'])[0]-objective(minus,r['A'],y,r['weights'],r['offset'])[0])/(2*eps)
            self.assertAlmostEqual(g[j],numeric,places=6)
        H=hessian(B.ravel(),r['A'],r['weights'],r['offset'])
        for j in [0,17,75]:
            plus=B.ravel().copy();minus=plus.copy();plus[j]+=eps;minus[j]-=eps
            numeric=(objective(plus,r['A'],y,r['weights'],r['offset'])[1]-objective(minus,r['A'],y,r['weights'],r['offset'])[1])/(2*eps)
            np.testing.assert_allclose(H[:,j],numeric,atol=1e-7,rtol=1e-6)

    def test_class_distribution_support_and_convex_payoff(self):
        args=fixture();m=fit_distribution(*args,4);np.testing.assert_allclose(m['prior'].sum(axis=1),1,atol=1e-13)
        for k in range(4):
            truth=args[2][args[1]==k]
            self.assertTrue((m['payoff_means'][:,k]>=min(truth)-1e-12).all())
            self.assertTrue((m['payoff_means'][:,k]<=max(truth)+1e-12).all())

    def test_missing_and_single_class(self):
        args=list(fixture(classes=3));m=fit_distribution(*args,4)
        self.assertNotIn(3,m['active']);np.testing.assert_array_equal(m['prior'][:,3],0)
        args[1][:]=0;m=fit_distribution(*args,4);np.testing.assert_allclose(m['coefficients'],0,atol=1e-12)
        np.testing.assert_allclose(m['prior'][:,0],1,atol=1e-13,rtol=0)

    def test_date_mass_not_six_times_independent_dates(self):
        m=fit_distribution(*fixture(),4)
        self.assertEqual(m['real_dates'],125);self.assertLess(m['date_ess'],125)
        self.assertAlmostEqual(sum(m['weights']),m['date_ess'])

    def test_directional_transform_and_missing_values(self):
        x=np.arange(18,dtype=float);a=signed_price(x,1);b=signed_price(x,-1)
        for j,i in enumerate(PRICE_IDS):self.assertEqual(b[j],-a[j] if i in DIRECTIONAL else a[j])
        changed=x.copy();changed[11:16]=100;np.testing.assert_array_equal(signed_price(changed,-1),b)
        x[11]=np.nan
        with self.assertRaises(ValueError):signed_price(x,1)

    def test_mixture_and_score_never_leave_class_payoff_support(self):
        raw={'raw':np.array([[.7,.2,.1,0],[.1,.4,.4,.1]]),'prior':np.array([[.2,.4,.3,.1],[.4,.3,.2,.1]]),
             'payoff_means':np.array([[.049,-.062,-.10,0],[.049,-.064,-.12,.01]])}
        day={'ready':True,'alpha':.3,'feedback':{'expired_unresolved_prefix':False}}
        for variant in ['blend','raw','prior','equal']:
            out=issue_distribution(raw,day,variant);np.testing.assert_allclose(out['probabilities'].sum(axis=1),1)
            self.assertTrue((out['values']<=raw['payoff_means'].max(axis=1)).all())
        day['feedback']['expired_unresolved_prefix']=True;self.assertFalse(issue_distribution(raw,day,'blend')['ready'])

    def test_choice_cash_ties_and_notready(self):
        self.assertEqual(choose({'ready':True,'values':[0.,0.]})['action'],'CASH')
        self.assertEqual(choose({'ready':True,'values':[.01,.01]})['action'],'SOXL')
        self.assertEqual(choose({'ready':False,'values':None})['action'],'CASH')

    def test_invalid_training_inputs(self):
        args=list(fixture());args[-1][0]=np.nan
        with self.assertRaises(ValueError):fit_distribution(*args,4)

    def test_mixture_date_equal_logscore_and_future_isolation(self):
        rows=[];reg=[];hist={};raw=np.array([[.6,.2,.1,.1],[.3,.4,.2,.1]])
        for date in pd.bdate_range('2020-01-01',periods=70):
            ids=[pd.Timestamp(str(date.date())+' '+clock,tz=core.ET).tz_convert('UTC') for clock in CLOCKS[:2]]
            reg.append({'date':date,'members':ids,'feature_ready':True,'release':ids[-1]+pd.Timedelta(days=13)})
            for t in ids:
                rows.append({'at':t,'L_gap':0,'S_gap':1});hist[t]={'raw':raw.copy(),'prior':np.ones((2,4))*.25}
        panel=pd.DataFrame(rows).set_index('at');at=pd.Timestamp('2020-06-01 13:25:30',tz='UTC')
        result=fit_mixture(panel,hist,reg,at,'gap');self.assertTrue(result['ready'])
        expected=.1*70*(np.log(.6/.25)+np.log(.4/.25))/2
        self.assertAlmostEqual(result['logodds'],expected)
        future={'date':pd.Timestamp('2020-07-01'),'members':[],'feature_ready':True,'release':pd.Timestamp('2020-07-20',tz='UTC')}
        unchanged=fit_mixture(panel,hist,reg+[future],at,'gap');self.assertEqual(result['alpha'],unchanged['alpha'])
        pending=[dict(r) for r in reg];pending[59]['release']=None
        stopped=fit_mixture(panel,hist,pending,at,'gap');self.assertFalse(stopped['ready']);self.assertTrue(stopped['feedback']['expired_unresolved_prefix'])
        args=list(fixture());args[1][0]=-1
        with self.assertRaises(ValueError):fit_distribution(*args,4)


if __name__=='__main__':unittest.main()
