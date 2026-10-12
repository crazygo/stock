import unittest
from copy import deepcopy
from selected_risk import *
from verify_selected import fit_audit,daily_audit,verify_panel,canonical,check_event
from test_shared import fixture
from sklearn.linear_model import LogisticRegression


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p,h,cls.market=fixture(65);p['growth_acceleration']=np.linspace(-.1,.2,len(p));p['recent_earnings_share']=.25
        cls.original=p;cls.records={}
        for at,raw in h.items():
            point=raw*1.3+.005;lower=point-.01;j=int(point[1]>point[0]);action=core.ETFS[j] if lower[j]>1e-12 else 'CASH'
            cls.records[at]={'at':at,'raws':{'full':raw},'estimates':{'full':{'ready':True,'point':point,'lower':lower}},
                'events':{'full_band':{'at':at,'date':p.loc[at,'date'],'action':action,'ready':True,'allocation':float(action!='CASH')}}}
        cls.panel=selected_panel(p,cls.records);cls.at=cls.panel.index[-4];cls.reg=registry(cls.panel,cls.market)
        cls.daily=fit_day(cls.panel,cls.records,cls.reg,cls.at)

    def arrays(self):
        rng=np.random.default_rng(37);X=rng.normal(size=(320,8));r=np.where(X[:,0]+rng.normal(size=320)>0,.049,-.065)
        w=np.repeat(2.**(-np.arange(80)[::-1]/63)/4,4);return X,r,w

    def test_sklearn_equivalence(self):
        X,r,w=self.arrays();m=fit_logit(X,r,w,80);Z=(X-m['center'])/m['scale'];v=w*np.abs(r);v*=80/v.sum()
        ref=LogisticRegression(C=.1,solver='lbfgs',tol=1e-12,max_iter=2000).fit(Z,(r>0).astype(int),sample_weight=v)
        np.testing.assert_allclose(m['coef'],ref.coef_[0],atol=1e-7,rtol=1e-7);self.assertAlmostEqual(m['intercept'],ref.intercept_[0],places=7)

    def test_gradient_equations(self):
        X,r,w=self.arrays();self.assertEqual(fit_audit(canonical(fit_logit(X,r,w,80)),X,r,w,80),1)

    def test_loss_magnitude_changes_decision(self):
        X=np.zeros((10,0));w=np.ones(10);r=np.r_[np.repeat(.04,6),np.repeat(-.08,4)]
        m=fit_logit(X,r,w,10);self.assertGreater(m['raw_win_fraction'],.5);self.assertLess(quality(m,[]),.5)

    def test_zero_returns(self):
        X,r,w=self.arrays();r[:]=0;m=fit_logit(X,r,w,80);self.assertEqual(m['mode'],'constant');self.assertEqual(quality(m,X[0]),.5)

    def test_single_class(self):
        X,r,w=self.arrays()
        for sign,q in [(1,1),(-1,0)]:
            m=fit_logit(X,np.abs(r)*sign,w,80);self.assertEqual(quality(m,X[0]),q)

    def test_zero_weight_rows(self):
        X,r,w=self.arrays();w[:10]=0;m=fit_logit(X,r,w,80);X[:10]=np.nan;r[:10]=np.nan
        self.assertEqual(canonical(m),canonical(fit_logit(X,r,w,80)))

    def test_return_scale_invariance(self):
        X,r,w=self.arrays();a=fit_logit(X,r,w,80);b=fit_logit(X,r*10,w,80)
        np.testing.assert_allclose(a['coef'],b['coef']);self.assertAlmostEqual(a['intercept'],b['intercept'])

    def test_independent_selected_features(self):
        self.assertEqual(verify_panel(self.original,self.panel,self.records),len(self.panel)*14)

    def test_winner_fixed_before_outcome(self):
        p=self.original.copy();p[['C','D']]=999;new=selected_panel(p,self.records)
        pd.testing.assert_frame_equal(new[SCORES+CONTEXT+['top_symbol']],self.panel[SCORES+CONTEXT+['top_symbol']])

    def test_tie_selects_soxl(self):
        records=deepcopy(self.records);records[self.at]['estimates']['full']['point']=np.array([.01,.01])
        p=selected_panel(self.original,records);self.assertEqual(p.loc[self.at,'top_symbol'],'SOXL')

    def test_future_records_do_not_change_daily_fit(self):
        records=deepcopy(self.records);p=self.original.copy();future=p.index>=day_cutoff(self.at);p.loc[future,['C','D']]=999
        for t in p.index[future]:records[t]['estimates']['full']['point']=[999.,-999.];records[t]['raws']['full']=[-999.,999.]
        p=selected_panel(p,records);new=fit_day(p,records,registry(p,self.market),self.at)
        self.assertEqual(canonical(new),canonical(self.daily))

    def test_independent_all_resampled_fits(self):
        self.assertEqual(daily_audit(self.panel,self.market,self.records,self.at,canonical(self.daily)),600)

    def test_unknown_prefix_stops_training(self):
        reg=deepcopy(self.reg);reg[10]['release']=None;d=fit_day(self.panel,self.records,reg,self.at)
        self.assertFalse(d['ready']);self.assertEqual(len(d['parents']),10);self.assertTrue(d['feedback']['expired_unresolved_prefix'])

    def test_warmup_preserves_source_actions(self):
        d=deepcopy(self.daily);d['ready']=False;info=issue(self.panel,self.at,self.records[self.at],d)
        es=events(self.at,self.panel.loc[self.at,'date'],info,d,self.records[self.at]);old=self.records[self.at]['events']['full_band']['action']
        for name,e in es.items():
            if name!='cash':self.assertEqual(e['action'],old)

    def test_point_quality_not_win_probability(self):
        X=np.zeros((10,0));r=np.r_[np.repeat(.04,6),np.repeat(-.08,4)];m=fit_logit(X,r,np.ones(10),10)
        self.assertAlmostEqual(quality(m,[]),.24/.56);self.assertNotAlmostEqual(quality(m,[]),.6)

    def test_source_gate_and_risk_gate(self):
        d=deepcopy(self.daily);info=issue(self.panel,self.at,self.records[self.at],d);record=deepcopy(self.records[self.at])
        for v in info.values():v.update(quality=.9,lower_quality=.8)
        record['events']['full_band']['action']='CASH';record['events']['full_band']['allocation']=0
        es=events(self.at,self.panel.loc[self.at,'date'],info,d,record)
        self.assertEqual(es['context_gate']['action'],'CASH');self.assertNotEqual(es['standalone']['action'],'CASH')
        for v in info.values():v['quality']=.4;v['lower_quality']=.3
        es=events(self.at,self.panel.loc[self.at,'date'],info,d,record)
        for name in ['context_gate','scores_gate','frequency_gate','standalone']:self.assertEqual(es[name]['action'],'CASH')


if __name__=='__main__':unittest.main()
