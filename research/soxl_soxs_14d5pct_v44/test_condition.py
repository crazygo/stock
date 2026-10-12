"""Independent conditional calibration equations and daily information boundaries."""
import unittest,copy,json
from threadpoolctl import threadpool_limits
from conditional_calibration import *
from audit_condition_calibration import solve,model_check,reference_penalty,daily_check,predict as independent_predict
from test_shared import fixture as old_fixture
from macro_learning import fit_bases as original_bases
from run_planning import json_ready


def wire(x):return json.loads(json.dumps(json_ready(x),default=str))
def fixture(n=65):
    p,h,m=old_fixture(n);rng=np.random.default_rng(441)
    for d in sorted(set(p.date)):p.loc[p.date==d,MACRO]=rng.normal(0,.02,len(MACRO))
    histories={k:{t:(v.copy() if k.startswith('macro_') else v*.9+.001) for t,v in h.items()} for k in VARIANTS}
    return p,histories,m

def base_fixture(n=145):
    rng=np.random.default_rng(441);days=pd.bdate_range('2024-01-02',periods=n);ats=[pd.Timestamp(str(d.date())+' '+CLOCKS[0],tz=core.ET).tz_convert('UTC') for d in days]
    p=pd.DataFrame(rng.normal(size=(n,35)),columns=columns(CLOCKS[0],'macro_plain'),index=pd.DatetimeIndex(ats));p['date']=days;p['clock']=CLOCKS[0];p['feature_ready']=True;p['terminal_release']=p.index+pd.Timedelta(minutes=5);p['release']=p.terminal_release;p['C']=rng.normal(-.01,.002,n);p['D']=np.where(p.iloc[:,0]>0,.05,-.06);return p

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits=threadpool_limits(1);cls.p,cls.h,cls.m=fixture();cls.at=cls.p.index[-4];cls.reg=registry(cls.p,cls.m);cls.daily=fit_calibration(cls.p,cls.h,cls.reg,cls.at)
    @classmethod
    def tearDownClass(cls):cls.limits.restore_original_limits()
    def test_independent_all_variants_bootstrap_equations(self):
        _,n=daily_check(self.p,self.m,self.h,self.at,wire(self.daily));self.assertEqual(n,1600)
    def test_only_registered_inputs_and_dimensions(self):
        ids=self.daily['members'];p=self.p.copy();p['C']=999;p['D']=999;p['sector_5']=999
        for k in VARIANTS:
            X=encode_condition(p,self.h[k],ids,k);np.testing.assert_array_equal(X,encode_condition(self.p,self.h[k],ids,k));self.assertEqual(X.shape[1],11 if k.endswith('context') else 3)
    def test_penalty_is_original_per_feature(self):
        for p in [2,10]:
            rng=np.random.default_rng(14);b=rng.normal(size=(4,p));mean=b.mean(0);P=reference_penalty(p)
            np.testing.assert_allclose(P,np.kron(20*np.eye(4)-3.75*np.ones((4,4)),np.eye(p)))
            self.assertAlmostEqual(b.ravel()@P@b.ravel(),20*np.sum(mean**2)+20*np.sum((b-mean)**2));self.assertGreater(np.linalg.eigvalsh(P).min(),0)
    def test_halfday_retains_one_date_weight(self):
        d=self.daily;self.assertIn(3,[len(r['members']) for r in d['parents']]);self.assertAlmostEqual(sum(d['row_weights']),len(d['parents']))
        for j,w in enumerate(d['day_weights']):self.assertAlmostEqual(sum(d['row_weights'][d['row_groups']==j]),w)
    def test_current_future_inputs_labels_and_raws_immune(self):
        p=self.p.copy();h=copy.deepcopy(self.h);mask=p.date>=p.loc[self.at,'date'];p.loc[mask,MACRO+['C','D']]=999
        for k in VARIANTS:
            for t in p.index[mask]:h[k][t]=np.array([999.,-999.])
        self.assertEqual(wire(self.daily),wire(fit_calibration(p,h,registry(p,self.m),self.at)))
    def test_pending_prefix_and_cutoff_equality(self):
        for release in [None,day_cutoff(self.at)]:
            r=copy.deepcopy(self.reg);r[10]['release']=release;d=fit_calibration(self.p,self.h,r,self.at);self.assertEqual(d['feedback']['selected_early_dates'],10);self.assertFalse(d['ready'])
    def test_minimum_dates_not_clock_rows(self):
        p,h,m=fixture(50);d=fit_calibration(p,h,registry(p,m),p.index[-4]);self.assertFalse(d['ready']);self.assertGreater(d['feedback']['candidate_rows'],60)
    def test_last_120_dates(self):
        p,h,m=fixture(150);ps,_=pool(registry(p,m),common_history(h),p.index[-4]);self.assertEqual(len(ps),120)
    def test_missing_raw_not_zero(self):
        h=copy.deepcopy(self.h);h['macro_plain'][self.daily['members'][0]]=None
        ps,_=pool(self.reg,common_history(h),self.at);self.assertEqual(len(ps),len(self.daily['parents'])-1)
    def test_zero_weight_rows_not_in_normalization(self):
        rng=np.random.default_rng(7);X=np.column_stack((rng.normal(size=(80,10)),np.arange(80)%4));Y=rng.normal(size=(80,2));w=np.ones(80);w[-1]=0
        a=fit_condition(X,Y,w,20);X[-1]=np.nan;Y[-1]=np.nan;self.assertEqual(a,fit_condition(X,Y,w,20))
    def test_constant_macro_reduces_to_original_predictions(self):
        rng=np.random.default_rng(12);raw=rng.normal(size=(80,2));clock=np.arange(80)%4;Y=rng.normal(size=(80,2));w=np.ones(80)/4
        oldX=np.column_stack((raw,clock));newX=np.column_stack((raw,np.ones((80,8))*3,clock))
        old=fit_condition(oldX,Y,w,20);new=fit_condition(newX,Y,w,20)
        np.testing.assert_allclose(forecast_condition(old,oldX),forecast_condition(new,newX),atol=1e-12)
    def test_units_offsets_and_weight_units_invariance(self):
        rng=np.random.default_rng(18);X=np.column_stack((rng.normal(size=(120,10)),np.arange(120)%4));Y=rng.normal(size=(120,2));w=np.ones(120);a=fit_condition(X,Y,w,30);Z=X.copy()
        for c in range(4):Z[Z[:,-1]==c,:-1]=Z[Z[:,-1]==c,:-1]*(c+2)+c*10
        b=fit_condition(Z,Y,w*7,30);np.testing.assert_allclose(forecast_condition(a,X),forecast_condition(b,Z),atol=1e-12)
    def test_macro_residual_dependency_recoverable(self):
        rng=np.random.default_rng(12);X=np.column_stack((rng.normal(size=(400,10)),np.arange(400)%4));Y=np.column_stack((X[:,2],-X[:,2]));oldX=X[:,[0,1,10]]
        new=fit_condition(X,Y,np.ones(400)/4,100);old=fit_condition(oldX,Y,np.ones(400)/4,100)
        self.assertLess(np.mean((forecast_condition(new,X)-Y)**2),np.mean((forecast_condition(old,oldX)-Y)**2)/2)
    def test_saved_predictions_independently_reconstructed(self):
        raw={k:self.h[k][self.at] for k in VARIANTS};issued=issue(self.p,self.at,raw,self.daily)
        for k in VARIANTS:
            X=encode_condition(self.p,{self.at:raw[k]},[self.at],k);want=raw[k]+independent_predict(self.daily['variants'][k]['point_model'],X)[0];np.testing.assert_allclose(issued[k]['point'],want,atol=1e-12)
            np.testing.assert_allclose(np.array(issued[k]['point'])-issued[k]['lower'],issued[k]['radius'])
    def test_missing_current_and_expired_abstain(self):
        raw={k:None for k in VARIANTS};self.assertTrue(all(not x['ready'] for x in issue(self.p,self.at,raw,self.daily).values()))
        raw={k:self.h[k][self.at] for k in VARIANTS};e=issue(self.p,self.at,raw,self.daily);self.assertEqual(decide(e['macro_context'],{'expired_unresolved_prefix':True})['action'],'CASH')
    def test_missing_or_fractional_clock_rejected(self):
        rng=np.random.default_rng(9);X=np.column_stack((rng.normal(size=(30,10)),np.arange(30)%3));Y=rng.normal(size=(30,2))
        with self.assertRaises(AssertionError):fit_condition(X,Y,np.ones(30),10)
        X[:,-1]=np.arange(30)%4;X[0,-1]=.1
        with self.assertRaises(AssertionError):fit_condition(X,Y,np.ones(30),10)
    def test_tiny_nonzero_scales_handled_as_registered(self):
        rng=np.random.default_rng(6);X=np.column_stack((rng.normal(size=(80,10))*1e-14,np.arange(80)%4));Y=rng.normal(size=(80,2));m=fit_condition(X,Y,np.ones(80),20);np.testing.assert_array_equal(m['input_scale'],np.ones((4,10)));model_check(m,solve(X,Y,np.ones(80),20))
    def test_parameter_tamper_detected(self):
        k='macro_context';ids=self.daily['members'];X=encode_condition(self.p,self.h[k],ids,k);Y=self.p.loc[ids,['C','D']].to_numpy();Y=np.column_stack((Y[:,0]+Y[:,1],Y[:,0]-Y[:,1]))-X[:,:2]
        m=copy.deepcopy(self.daily['variants'][k]['point_model']);m['coefficients'][0][0][0]+=.01
        with self.assertRaises(AssertionError):model_check(m,solve(X,Y,self.daily['row_weights'],len(self.daily['parents'])))
    def test_base_fits_exact_unchanged_and_reused(self):
        p=base_fixture();at=p.index[-1];a=fit_bases(p,at);b=original_bases(p,at)
        for k in VARIANTS:
            old='macro' if k.startswith('macro_') else 'baseline';self.assertEqual(wire(a[1][k]),wire(b[1][old]));np.testing.assert_array_equal(a[2][k],b[2][old])
    def test_base_future_labels_not_used(self):
        p=base_fixture();at=p.index[130];a=wire(fit_bases(p,at)[1]);p.loc[p.index>=at,['C','D']]=999;self.assertEqual(a,wire(fit_bases(p,at)[1]))

if __name__=='__main__':unittest.main()
