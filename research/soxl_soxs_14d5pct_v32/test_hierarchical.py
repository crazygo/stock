import unittest,copy,json
from hierarchical_error import *
from audit_hierarchical import solve,model_check,reference_penalty,daily_check
from test_shared import fixture
from run_planning import json_ready


class HierarchicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p,cls.h,cls.m=fixture();cls.at=cls.p.index[-4];cls.reg=registry(cls.p,cls.m);cls.daily=fit_day(cls.p,cls.h,cls.reg,cls.at)

    def test_independent_full_bootstrap_equations(self):
        saved=json.loads(json.dumps(json_ready(self.daily),default=str));_,n=daily_check(self.p,self.m,self.h,self.at,saved);self.assertEqual(n,1200)

    def test_penalty_preserves_common_and_shrinks_deviations(self):
        rng=np.random.default_rng(19);b=rng.normal(size=(4,2));mean=b.mean(axis=0)
        for kind,lam in [('partial',20),('independent',5)]:
            v=b.ravel();self.assertAlmostEqual(v@penalty(kind)@v,20*sum(mean**2)+lam*np.sum((b-mean)**2))
            np.testing.assert_allclose(penalty(kind),reference_penalty(kind));self.assertGreater(np.linalg.eigvalsh(penalty(kind)).min(),0)
        common=np.tile([2.,3.],4);self.assertAlmostEqual(common@penalty('partial')@common,20*13)
        self.assertAlmostEqual(common@penalty('independent')@common,20*13)

    def test_date_weights_and_halfday(self):
        d=self.daily;self.assertIn(3,[len(r['members']) for r in d['parents']]);self.assertAlmostEqual(sum(d['row_weights']),len(d['parents']))
        for j,w in enumerate(d['day_weights']):self.assertAlmostEqual(sum(d['row_weights'][d['row_groups']==j]),w)

    def test_pending_clock_blocks_date_prefix(self):
        r=copy.deepcopy(self.reg);r[10]['release']=None;parents,f=pool(r,self.h,self.at);self.assertEqual(len(parents),10);self.assertTrue(f['expired_unresolved_prefix'])

    def test_cutoff_and_future_labels(self):
        r=copy.deepcopy(self.reg);r[10]['release']=day_cutoff(self.at);parents,_=pool(r,self.h,self.at);self.assertEqual(len(parents),10)
        p=self.p.copy();p.loc[p.date>=p.loc[self.at,'date'],['C','D']]=999
        self.assertEqual(pool(registry(p,self.m),self.h,self.at),pool(self.reg,self.h,self.at))

    def test_zero_weight_immunity(self):
        rng=np.random.default_rng(32);X=np.column_stack((rng.normal(size=(40,2)),np.arange(40)%4));Y=rng.normal(size=(40,2));w=np.ones(40);w[-1]=0
        for kind in VARIANTS:
            a=fit_map(X,Y,w,kind,10);xx=X.copy();yy=Y.copy();xx[-1]=np.nan;yy[-1]=np.nan
            self.assertEqual(a,fit_map(xx,yy,w,kind,10))

    def test_clock_units_and_offsets_do_not_change_fit_predictions(self):
        rng=np.random.default_rng(6);X=np.column_stack((rng.normal(size=(100,2)),np.arange(100)%4));Y=rng.normal(size=(100,2));w=np.ones(100)
        for kind in VARIANTS:
            before=fit_map(X,Y,w,kind,25);xx=X.copy()
            for c in range(4):xx[xx[:,2]==c,:2]=xx[xx[:,2]==c,:2]*(c+2)+c*100
            after=fit_map(xx,Y,w,kind,25)
            np.testing.assert_allclose(forecast(before,X),forecast(after,xx),rtol=1e-10,atol=1e-10)

    def test_independent_balanced_clocks_equals_four_ridge20_models(self):
        from sklearn.linear_model import Ridge
        rng=np.random.default_rng(8);X=np.column_stack((rng.normal(size=(400,2)),np.arange(400)%4));Y=rng.normal(size=(400,2));w=np.ones(400)/4
        m=fit_map(X,Y,w,'independent',100)
        for c in range(4):
            ids=X[:,2]==c;Z=(X[ids,:2]-m['input_center'][c])/m['input_scale'][c]
            h=Ridge(alpha=20,solver='svd').fit(Z,Y[ids]);np.testing.assert_allclose(forecast(m,X[ids]),h.predict(Z),atol=1e-12)

    def test_partial_retains_distinct_opposite_clock_slopes(self):
        rng=np.random.default_rng(5);raw=rng.normal(size=(200,2));raw-=raw.mean(axis=0);raw/=raw.std(axis=0)
        X=np.vstack([np.column_stack((raw,np.repeat(c,len(raw)))) for c in range(4)])
        Y=np.vstack([raw[:,0,None]*np.array([[(-1.)**c,-(-1.)**c]]) for c in range(4)]);w=np.ones(len(X))/4
        a=fit_map(X,Y,w,'partial',200);b=fit_map(X,Y,w,'independent',200);s=fit_map(X,Y,w,'shared',200)
        self.assertGreater(a['coefficients'][0][0][0],0);self.assertLess(a['coefficients'][1][0][0],0)
        self.assertLess(np.var(np.array(a['coefficients']),axis=0).sum(),np.var(np.array(b['coefficients']),axis=0).sum())
        self.assertLess(np.mean((forecast(a,X)-Y)**2),np.mean((forecast(s,X)-Y)**2))

    def test_constant_scores_keep_clock_intercepts(self):
        X=np.column_stack((np.ones((40,2)),np.arange(40)%4));Y=np.column_stack((X[:,2],-X[:,2]))
        for kind in VARIANTS:np.testing.assert_allclose(forecast(fit_map(X,Y,np.ones(40),kind,10),X),Y,atol=1e-12)

    def test_whole_day_weight_and_minimum_not_clock_rows(self):
        p,h,m=fixture(50);d=fit_day(p,h,registry(p,m),p.index[-4]);self.assertFalse(d['ready']);self.assertGreater(d['feedback']['candidate_rows'],60)
        self.assertTrue(all(t<day_cutoff(self.at) for t in self.daily['members']))

    def test_common_margin_and_expired_abstention(self):
        d=issue_hierarchical(self.p,self.at,self.h[self.at],self.daily)
        for k in VARIANTS:np.testing.assert_allclose(np.array(d[k]['point'])-d[k]['lower'],[d[k]['radius']]*2)
        self.assertEqual(decide(d['partial'],{'expired_unresolved_prefix':True})['action'],'CASH')
        self.assertFalse(issue_hierarchical(self.p,self.at,None,self.daily)['partial']['ready'])


if __name__=='__main__':unittest.main()
