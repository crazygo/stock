import unittest,copy,json,tempfile
from threadpoolctl import threadpool_limits
from fixed_candidate import *
from adaptive_penalty import fit_day as old_fit_day,issue as old_issue
from audit_fixed import daily_check
from test_penalty import setup,wire
from run_fixed import authorize_cases

class FixedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits=threadpool_limits(1);cls.p,cls.h,cls.m,cls.raw,cls.hist=setup();cls.at=cls.p.index[-4];cls.reg=registry(cls.p,cls.m);cls.daily=fit_day(cls.p,cls.raw,cls.reg,cls.at)
    @classmethod
    def tearDownClass(cls):cls.limits.restore_original_limits()
    def test_fixed_values_and_routes(self):
        self.assertEqual(CANDIDATES,['strong','plain']);self.assertEqual(STRENGTH['strong'],2000.);self.assertEqual(len(ROUTES),6)
    def test_independent_all_equations_and_bootstraps(self):
        _,n=daily_check(self.p,self.m,self.raw,self.at,wire(self.daily));self.assertEqual(n,800)
    def test_exact_v45_models(self):
        old=old_fit_day(self.p,self.raw,self.hist,self.reg,self.at)
        for k in CANDIDATES:self.assertEqual(wire(self.daily['models'][k]),wire(old['models'][k]))
    def test_exact_v45_forecasts(self):
        old=old_fit_day(self.p,self.raw,self.hist,self.reg,self.at);a=issue(self.p,self.at,self.raw[self.at],self.daily);b=old_issue(self.p,self.at,self.raw[self.at],old)
        for k in CANDIDATES:self.assertEqual(a[k],b[k])
    def test_cost_and_premarket_use_new_primary(self):
        estimate={'ready':True,'point':[.1,0.],'lower':[.03,-.03]};other={'ready':True,'point':[0.,.1],'lower':[-.03,.03]}
        es=events(self.at,self.p.loc[self.at,'date'],CLOCKS[0],{'strong':estimate,'plain':other},{'expired_unresolved_prefix':False})
        self.assertEqual(es['strong_band'],es['double_cost']);self.assertEqual(es['strong_band'],es['premarket_only']);self.assertEqual(es['strong_band']['action'],'SOXL');self.assertEqual(es['plain_band']['action'],'SOXS')
    def test_future_inputs_labels_and_raw_history_do_not_fit(self):
        p=self.p.copy();h=copy.deepcopy(self.raw);mask=p.date>=p.loc[self.at,'date'];p.loc[mask,['C','D']+MACRO]=999
        for t in p.index[mask]:h[t]=[999.,999.]
        self.assertEqual(wire(self.daily),wire(fit_day(p,h,self.reg,self.at)))
    def test_unmatured_prefix_and_minimum_dates(self):
        r=copy.deepcopy(self.reg);r[10]['release']=day_cutoff(self.at);d=fit_day(self.p,self.raw,r,self.at);self.assertFalse(d['ready']);self.assertEqual(len(d['parents']),10)
    def test_original_calibration_date_weights_preserved(self):
        self.assertIn(3,[len(r['members']) for r in self.daily['parents']]);self.assertAlmostEqual(sum(self.daily['row_weights']),len(self.daily['parents']))
        for i,w in enumerate(self.daily['day_weights']):self.assertAlmostEqual(sum(self.daily['row_weights'][self.daily['row_groups']==i]),w)
    def test_missing_inputs_and_pending_guard(self):
        self.assertTrue(all(not e['ready'] for e in issue(self.p,self.at,None,self.daily).values()));est=issue(self.p,self.at,self.raw[self.at],self.daily);es=events(self.at,self.p.loc[self.at,'date'],CLOCKS[0],est,{'expired_unresolved_prefix':True});self.assertTrue(all(e['action']=='CASH' for e in es.values()))
    def test_tamper_independent_equations(self):
        d=wire(self.daily);d['models']['strong']['point_model']['coefficients'][0][0][0]+=.001
        with self.assertRaises(AssertionError):daily_check(self.p,self.m,self.raw,self.at,d)
    def test_other_version_cannot_authorize_cases(self):
        with self.assertRaises(AssertionError):authorize_cases(core.ROOT/'.cache/soxl_soxs_14d5pct_v45/runs/penalty_20261013',{'primary':'strong_band'})
    def test_guard_rejects_failed_primary(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=read(SOURCE/'protocol.json');stats={'primary_average':-.1,'all_months_nonnegative':False,'completed_trades':2,'monthly':[{'month':m,'return':-.1,'mark_complete':True} for m in p['development']['months']]};r={'primary_name':'strong_band','primary':stats,'routes':{'strong_band':stats},'coverage':{'complete':True}}
            dump(root/'results.json',r);dump(root/'registration.json',{'phase':'development','protocol_sha256':sha(SOURCE/'protocol.json')});dump(root/'verification.json',{'passed':True,'results_sha256':sha(root/'results.json')})
            with self.assertRaisesRegex(ValueError,'Development primary failed'):authorize_cases(root,p)

if __name__=='__main__':unittest.main()
