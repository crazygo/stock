import unittest,copy,json
from threadpoolctl import threadpool_limits
from adaptive_penalty import *
from audit_penalty import solve,model_check,daily_check,select_reference
from test_condition import fixture,wire
from conditional_calibration import fit_calibration as old_daily,issue as old_issue


def setup(n=75):
    p,h,m=fixture(n);raw=h['base_plain'];hist={}
    for t,r in raw.items():
        hist[t]={k:(np.array(r)+(.03 if k=='weak' else .01)).tolist() for k in CANDIDATES} if r is not None else None
    return p,h,m,raw,hist

class PenaltyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits=threadpool_limits(1);cls.p,cls.h,cls.m,cls.raw,cls.hist=setup();cls.at=cls.p.index[-4];cls.reg=registry(cls.p,cls.m);cls.daily=fit_day(cls.p,cls.raw,cls.hist,cls.reg,cls.at)
    @classmethod
    def tearDownClass(cls):cls.limits.restore_original_limits()
    def test_all_models_bootstraps_and_selections_independent(self):
        _,n,s=daily_check(self.p,self.m,self.raw,self.hist,self.at,wire(self.daily));self.assertEqual((n,s),(1600,200))
    def test_source_weak_and_plain_exact(self):
        d=old_daily(self.p,self.h,self.reg,self.at)
        self.assertEqual(wire(self.daily['models']['weak']),wire(d['variants']['base_context']));self.assertEqual(wire(self.daily['models']['plain']),wire(d['variants']['base_plain']))
    def test_source_predictions_exact(self):
        d=old_daily(self.p,self.h,self.reg,self.at);new=issue(self.p,self.at,self.raw[self.at],self.daily);old=old_issue(self.p,self.at,{k:self.h[k][self.at] for k in self.h},d)
        self.assertEqual(wire(new['plain']),wire(old['base_plain']));self.assertEqual(wire(new['weak']),wire(old['base_context']))
    def test_future_labels_and_forecasts_not_used(self):
        p=self.p.copy();h=copy.deepcopy(self.hist);raw=copy.deepcopy(self.raw);cut=day_cutoff(self.at);mask=p.date>=p.loc[self.at,'date'];p.loc[mask,['C','D']+MACRO]=999
        for t in p.index[mask]:h[t]={k:[999.,-999.] for k in CANDIDATES};raw[t]=[999.,999.]
        self.assertEqual(wire(self.daily),wire(fit_day(p,raw,h,self.reg,self.at)))
    def test_selector_maturity_prefix(self):
        reg=copy.deepcopy(self.reg);reg[12]['release']=day_cutoff(self.at);d=fit_day(self.p,self.raw,self.hist,reg,self.at);self.assertFalse(d['ready']);self.assertFalse(d['selector_ready']);self.assertEqual(len(d['parents']),12)
    def test_selector_minimum_is_dates(self):
        h={t:v for t,v in self.hist.items() if self.p.loc[t,'date']>=self.p.date.unique()[25]};d=fit_day(self.p,self.raw,h,self.reg,self.at)
        self.assertTrue(d['ready']);self.assertFalse(d['selector_ready']);self.assertEqual(d['selected'],'plain');self.assertEqual(set(d['bootstrap_selected']),{'plain'})
    def test_warmup_exact_plain_prediction_and_radius(self):
        d=fit_day(self.p,self.raw,{},self.reg,self.at);est=issue(self.p,self.at,self.raw[self.at],d);self.assertEqual(est['adaptive'],est['plain'])
    def test_ties_prefer_more_shrinkage(self):
        self.assertEqual(choose(np.zeros((12,4,2)),np.ones(12))[0],'plain')
        x=np.zeros((12,4,2));x[:,0]=10;self.assertEqual(choose(x,np.ones(12))[0],'strong')
    def test_zero_selected_weight_bootstrap_fallback(self):
        self.assertEqual(choose(np.ones((12,4,2)),np.zeros(12))[0],'plain')
        self.assertEqual(select_reference(self.p,self.hist,self.daily['selector_parents'],np.zeros(len(self.daily['selector_parents'])))[0],'plain')
    def test_selector_uses_pair_not_win_rate(self):
        x=np.zeros((10,4,2));x[:,0]=1;x[:,1]=.1;x[:,2]=.2;x[:,3]=.3;self.assertEqual(choose(x,np.ones(10))[0],'strong')
    def test_date_weights_equal_halfday(self):
        d=self.daily;self.assertIn(3,[len(r['members']) for r in d['parents']]);self.assertAlmostEqual(sum(d['row_weights']),len(d['parents']))
        for i,w in enumerate(d['day_weights']):self.assertAlmostEqual(sum(d['row_weights'][d['row_groups']==i]),w)
    def test_fixed_grid_and_raw_penalty(self):
        self.assertEqual(STRENGTH,{'plain':None,'strong':2000.,'medium':200.,'weak':20.})
        rng=np.random.default_rng(13);X=np.c_[rng.normal(size=(80,10)),np.arange(80)%4];Y=rng.normal(size=(80,2));w=np.ones(80)/4
        for k in CANDIDATES:model_check(fit_candidate(X,Y,w,20,k),solve(X,Y,w,20,k))
    def test_constant_macro_all_predictions_reduce_to_plain(self):
        rng=np.random.default_rng(8);X=np.c_[rng.normal(size=(80,2)),np.ones((80,8)),np.arange(80)%4];Y=rng.normal(size=(80,2));w=np.ones(80)/4
        expected=candidate_prediction(fit_candidate(X,Y,w,20,'plain'),X)
        for k in CANDIDATES:np.testing.assert_allclose(candidate_prediction(fit_candidate(X,Y,w,20,k),X),expected,atol=1e-12)
    def test_model_tampering_detected(self):
        ids=self.daily['members'];X=np.array([[*self.raw[t],*[self.p.loc[t,c] for c in MACRO],CLOCKS.index(self.p.loc[t,'clock'])] for t in ids]);T=self.p.loc[ids,['C','D']].to_numpy();Y=np.c_[T[:,0]+T[:,1],T[:,0]-T[:,1]]-X[:,:2]
        m=copy.deepcopy(self.daily['models']['medium']['point_model']);m['coefficients'][0][3][0]+=.01
        with self.assertRaises(AssertionError):model_check(m,solve(X,Y,self.daily['row_weights'],len(self.daily['parents']),'medium'))
    def test_selector_parameter_tampering_detected(self):
        saved=wire(self.daily);saved['selected']='wrong'
        with self.assertRaises(AssertionError):daily_check(self.p,self.m,self.raw,self.hist,self.at,saved)
    def test_resampling_can_change_choice(self):
        e=np.ones((8,4,2))*3;e[:4,0]=0;e[:4,1]=2;e[4:,0]=2;e[4:,1]=0
        self.assertEqual(choose(e,np.ones(8))[0],'plain');self.assertEqual(choose(e,np.r_[np.zeros(4),np.ones(4)*2])[0],'strong')
    def test_band_includes_choice_changes(self):
        d=copy.deepcopy(self.daily);d['bootstrap_selected']=['weak' if i%2 else 'strong' for i in range(199)];e=issue(self.p,self.at,self.raw[self.at],d)
        draws=np.array([e[k]['bootstrap_predictions'][i] for i,k in enumerate(d['bootstrap_selected'])]);point=np.array(e[d['selected']]['point']);want=max(0.,sorted(np.max(draws-point,axis=1))[179]);self.assertAlmostEqual(e['adaptive']['radius'],want)
    def test_missing_raw_and_expired_abstain(self):
        self.assertTrue(all(not e['ready'] for e in issue(self.p,self.at,None,self.daily).values()));est=issue(self.p,self.at,self.raw[self.at],self.daily);self.assertEqual(decide(est['adaptive'],{'expired_unresolved_prefix':True})['action'],'CASH')

if __name__=='__main__':unittest.main()
