"""Meaningful offset optimum, causal fit, bootstrap and abstention checks."""
import unittest,copy,tempfile
from types import SimpleNamespace
from threadpoolctl import threadpool_limits
from scipy.optimize import minimize
from direction_correction import *
from audit_offset import check_offset,reference_q
from audit_joint import daily_check
from run_planning import json_ready
import json
wire=lambda x:json.dumps(json_ready(x),sort_keys=True,default=str)


def fixture(n=66):
    rng=np.random.default_rng(26);dates=pd.bdate_range('2025-03-03',periods=n);rows=[];records={};histories={k:{} for k in VARIANTS};cal={}
    for date in dates:
        cal[date]={'close_at':pd.Timestamp(str(date.date())+' 16:00',tz=core.ET)}
        for c in CLOCKS:
            at=pd.Timestamp(str(date.date())+' '+c,tz=core.ET).tz_convert('UTC')
            d=float(rng.normal(0,.04));q=float(rng.uniform(.25,.75));v=payoff({'C':-.01,'M':.035},q)
            rows.append(dict(at=at,date=date,clock=c,C=-.01,D=d,terminal_release=cal[date]['close_at'].tz_convert('UTC')+pd.Timedelta(seconds=1),feature_ready=True,
                **{col:float(rng.normal()) for col in option_columns(2)+option_columns(5)}))
            records[at]={'fits':{'full':{'raw':v}}}
            for k in VARIANTS:histories[k][at]=np.array([v[s] for s in core.ETFS])
    p=pd.DataFrame(rows).set_index('at');at=p.index[-4]
    histories={k:{t:v for t,v in h.items() if t<at} for k,h in histories.items()}
    return p,records,histories,SimpleNamespace(calendar=cal),at


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits=threadpool_limits(1);cls.p,cls.records,cls.h,cls.market,cls.at=fixture()
        cls.day=fit_day(cls.p,cls.records,cls.h,registry(cls.p,cls.market),cls.at)
    @classmethod
    def tearDownClass(cls):cls.limits.restore_original_limits()
    def data(self):
        r=np.random.default_rng(45);X=np.column_stack([np.arange(260)%4,r.normal(size=(260,8))]);q=r.uniform(.1,.9,260);D=r.normal(0,.04,260);w=r.uniform(.1,1.,260)
        return X,q,D,w,65
    def test_independent_optimum_and_numerical_derivative(self):
        X,q,D,w,n=self.data();m=fit_offset(X,q,D,w,n);reference,_=check_offset(X,q,D,w,n,m)
        A=np.column_stack([np.eye(4)[X[:,0].astype(int)],(X[:,1:]-m['center'])/m['scale']]);cw=w*abs(D);cw*=n/cw.sum();off=np.log(q/(1-q));y=D>0;lam=np.array([1.]*4+[10.]*8)
        def loss(b):return float(np.sum(cw*(np.logaddexp(0,off+A@b)-y*(off+A@b)))+.5*np.sum(lam*b*b))
        result=minimize(loss,np.zeros(12),method='BFGS',options={'gtol':1e-6})
        np.testing.assert_allclose(result.x,m['coefficients'],atol=2e-6)
        b=np.array(m['coefficients']);eps=1e-5;grad=[(loss(b+eps*e)-loss(b-eps*e))/(2*eps) for e in np.eye(12)]
        self.assertLess(max(abs(np.array(grad))),1e-6)
        self.assertAlmostEqual(correct_q(m,X[0],q[0]),reference_q(reference,X[0],q[0]),places=14)
    def test_single_class_finite_with_extreme_source(self):
        X,q,D,w,n=self.data();q[:2]=[0,1]
        for sign in [1,-1]:
            m=fit_offset(X,q,abs(D)*sign,w,n);check_offset(X,q,abs(D)*sign,w,n,m)
            self.assertTrue(np.isfinite(m['coefficients']).all())
            self.assertTrue(0<correct_q(m,X[0],0)<1)
    def test_zero_direction_exact_source_probability(self):
        X,q,D,w,n=self.data();m=fit_offset(X,q,D*0,w,n);check_offset(X,q,D*0,w,n,m)
        self.assertEqual(m['coefficients'],[0.]*12)
        for value in [0.,.3,1.]:self.assertEqual(correct_q(m,X[0],value),value)
    def test_weight_and_payoff_unit_invariance(self):
        X,q,D,w,n=self.data();a=fit_offset(X,q,D,w,n);b=fit_offset(X,q,D*7,w*13,n)
        np.testing.assert_allclose(a['coefficients'],b['coefficients'],atol=1e-12)
    def test_feature_units_and_constant_feature(self):
        X,q,D,w,n=self.data();X[:,1]=4;a=fit_offset(X,q,D,w,n);Y=X.copy();Y[:,1:]=Y[:,1:]*3+7;b=fit_offset(Y,q,D,w,n)
        self.assertEqual(a['scale'][0],1);np.testing.assert_allclose(a['coefficients'],b['coefficients'],atol=1e-12)
        self.assertAlmostEqual(correct_q(a,X[-1],q[-1]),correct_q(b,Y[-1],q[-1]),places=14)
    def test_tampered_coefficient_rejected(self):
        X,q,D,w,n=self.data();m=fit_offset(X,q,D,w,n);m['coefficients'][0]+=.01
        with self.assertRaises(AssertionError):check_offset(X,q,D,w,n,m)
    def test_joint_explicit_repeated_dates(self):
        m,q,n,nq=daily_check(self.p,self.market,self.records,self.h,self.at,json.loads(wire(self.day)))
        self.assertEqual(n,1600);self.assertEqual(nq,600);self.assertEqual(set(m),set(VARIANTS));self.assertEqual(set(q),set(CORRECTIONS))
    def test_future_labels_inputs_and_source_predictions_immune(self):
        p=self.p.copy();p.loc[p.index>=self.at,['C','D']]=900;p.loc[p.index>self.at,option_columns(2)+option_columns(5)]=900
        rec=copy.deepcopy(self.records)
        for t in rec:
            if t>self.at:rec[t]['fits']['full']['raw']['q']=.99
        b=fit_day(p,rec,self.h,registry(p,self.market),self.at)
        self.assertEqual(wire(self.day),wire(b));self.assertEqual(wire(issue(self.p,self.records,self.at,self.day)),wire(issue(p,rec,self.at,b)))
    def test_unresolved_prefix_cannot_be_skipped(self):
        p=self.p.copy();p.loc[p.index[30*4],'terminal_release']=pd.NaT
        d=fit_day(p,self.records,self.h,registry(p,self.market),self.at)
        self.assertFalse(d['ready']);self.assertEqual(len(d['parents']),30);self.assertTrue(d['feedback']['expired_unresolved_prefix'])
    def test_before_60_dates_keeps_source_but_abstains(self):
        p,r,h,m,at=fixture(60);d=fit_day(p,r,h,registry(p,m),at);raw,detail,est,direct=issue(p,r,at,d)
        self.assertFalse(d['ready']);self.assertEqual(len(d['parents']),59)
        for k in VARIANTS:self.assertEqual(detail[k]['raw'],r[at]['fits']['full']['raw']);self.assertFalse(est[k]['ready'])
        self.assertTrue(all(e['action']=='CASH' for e in events(at,p.loc[at,'date'],p.loc[at,'clock'],est,direct,d['feedback']).values()))
    def test_current_missing_abstains(self):
        p=self.p.copy();p.loc[self.at,'feature_ready']=False
        raw,_,est,direct=issue(p,self.records,self.at,self.day)
        self.assertTrue(all(v is None for v in raw.values()));self.assertFalse(direct['ready'])
    def test_C_M_unchanged_and_baseline_exact(self):
        raw,detail,_,_=issue(self.p,self.records,self.at,self.day);source=self.records[self.at]['fits']['full']['raw']
        for k in VARIANTS:
            self.assertEqual(detail[k]['raw']['C'],source['C']);self.assertEqual(detail[k]['raw']['M'],source['M'])
        self.assertEqual(detail['baseline']['raw'],source)
    def test_query_and_calibration_both_move_in_joint_draw(self):
        raw,_,est,direct=issue(self.p,self.records,self.at,self.day)
        draw=np.array(direct['bootstrap_predictions']);self.assertGreater(float(draw[:,0].std()),1e-5)
        row=np.array([*draw[0],CLOCKS.index(self.p.loc[self.at,'clock'])]);cal=self.day['variants']['conditional']['bootstrap_models'][0]
        np.testing.assert_allclose(est['conditional']['bootstrap_predictions'][0],draw[0]+forecast(cal,row)[0],atol=1e-12)
        self.assertNotEqual(est['conditional']['bootstrap_predictions'][0],(raw['conditional']+forecast(cal,np.r_[raw['conditional'],row[2]])[0]).tolist())
    def test_negative_payoff_abstains_despite_relative_direction(self):
        e=decide({'ready':True,'point':[-.01,-.08],'lower':[-.02,-.09]}, {'expired_unresolved_prefix':False},True)
        self.assertEqual(e['action'],'CASH')
    def test_failed_primary_cannot_open_cases(self):
        from run_correction import authorize_cases
        p=read(SOURCE/'protocol.json')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);s={'primary_average':0.,'completed_trades':0,'all_months_nonnegative':True,'monthly':[{'month':m,'return':0.,'mark_complete':True} for m in p['development']['months']]}
            dump(root/'registration.json',{'phase':'development','protocol_sha256':sha(SOURCE/'protocol.json')})
            dump(root/'results.json',{'primary_name':p['primary'],'primary':s,'routes':{p['primary']:s},'coverage':{'complete':True}})
            dump(root/'verification.json',{'passed':True,'results_sha256':sha(root/'results.json')})
            with self.assertRaisesRegex(ValueError,'Development primary failed'):authorize_cases(root,p)


if __name__=='__main__':unittest.main()
