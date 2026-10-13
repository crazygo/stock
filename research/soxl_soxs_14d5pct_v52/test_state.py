"""Meaningful offset optimum, causal fit, bootstrap and abstention checks."""
import unittest,copy,tempfile
from types import SimpleNamespace
from threadpoolctl import threadpool_limits
from scipy.optimize import minimize,brentq
from state_reliability import *
from audit_shrinkage import check_shrinkage,reference_q
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
            rows.append(dict(at=at,date=date,clock=c,C=-.01,D=d,terminal_release=cal[date]['close_at'].tz_convert('UTC')+pd.Timedelta(seconds=1),feature_ready=True,relative_5=float(rng.normal()),breadth_5=float(rng.uniform(-1,1)),volume_direction_diffusion=float(rng.uniform(-1,1)),
                **{col:float(rng.normal()) for col in option_columns(2)+option_columns(5)}))
            records[at]={'fits':{'full':{'raw':v}}}
            for k in VARIANTS:
                z=payoff(v,.5 if k=='neutral' else q)
                histories[k][at]=np.array([z[s] for s in core.ETFS])
    p=add_state(pd.DataFrame(rows).set_index('at'));at=p.index[-4]
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
        r=np.random.default_rng(45);X=np.column_stack([np.arange(260)%4,r.uniform(0,1,260)]);q=r.uniform(.1,.9,260);D=r.normal(0,.04,260);w=r.uniform(.1,1.,260)
        return X,q,D,w,65
    def test_independent_solver_optimum(self):
        X,q,D,w,n=self.data()
        for kind in CORRECTIONS:
            x=X[:,:1] if kind=='global' else X;m=fit_shrinkage(x,q,D,w,n,kind);ref,_=check_shrinkage(x,q,D,w,n,kind,m)
            f=np.log(q/(1-q));u=X[:,1]
            A=f[:,None] if kind=='global' else np.column_stack([f*(1-u),f*u]) if kind=='state' else np.column_stack([f,2*u-1])
            cw=w*abs(D);cw*=n/cw.sum();y=D>0;penalty=np.array(m['penalty']);center=np.array(m['prior'])
            def loss(b):return float(np.sum(cw*(np.logaddexp(0,A@b)-y*(A@b)))+.5*np.sum(penalty*(b-center)**2))
            result=minimize(loss,center,method='SLSQP',bounds=[(-1,1)]*len(center),options={'ftol':1e-12,'maxiter':1000})
            self.assertTrue(result.success);np.testing.assert_allclose(result.x,m['coefficients'],atol=3e-6)
            self.assertAlmostEqual(correct_q(m,x[0],q[0]),reference_q(ref,x[0],q[0]),places=14)
    def test_exact_quadratic_box_step(self):
        rng=np.random.default_rng(812)
        for _ in range(12):
            A=rng.normal(size=(2,2));H=A.T@A+np.eye(2);g=rng.normal(size=2)*10;b=rng.uniform(-1,1,2)
            d=box_step(H,g,b)
            ref=minimize(lambda x:float(g@x+.5*x@H@x),np.zeros(2),jac=lambda x:g+H@x,method='SLSQP',bounds=list(zip(-1-b,1-b)),options={'ftol':1e-12})
            self.assertTrue(ref.success);np.testing.assert_allclose(d,ref.x,atol=3e-6)
    def test_neutral_state_reduces_to_scalar_objective(self):
        X,q,D,w,n=self.data();X[:,1]=.5;m=fit_shrinkage(X,q,D,w,n,'state');old=prior.fit_shrinkage(X[:,:1],q,D,w,n,'signed')
        np.testing.assert_allclose(m['coefficients'],old['coefficients']*2,atol=1e-8)
        self.assertAlmostEqual(m['objective'],old['objective'],places=10)
    def test_opposite_state_reliability_can_be_learned(self):
        X,q,D,w,n=self.data();X[:,1]=np.arange(len(q))%2
        D=np.where((q-.5)*(2*X[:,1]-1)>0,1.,-1.)
        m=fit_shrinkage(X,q,D,np.ones(len(q)),n,'state');check_shrinkage(X,q,D,np.ones(len(q)),n,'state',m)
        self.assertLess(m['coefficients'][0],0);self.assertGreater(m['coefficients'][1],0)
        self.assertLess(correct_q(m,[0,0],.8),.5);self.assertGreater(correct_q(m,[0,1],.8),.5)
    def test_single_class_and_extreme_source(self):
        X,q,D,w,n=self.data();q[:2]=[0,1]
        for kind in CORRECTIONS:
            x=X[:,:1] if kind=='global' else X
            for sign in [1,-1]:
                m=fit_shrinkage(x,q,abs(D)*sign,w,n,kind);check_shrinkage(x,q,abs(D)*sign,w,n,kind,m)
                self.assertTrue(np.isfinite(m['coefficients']).all());self.assertTrue(0<correct_q(m,x[0],0)<1)
    def test_zero_magnitude_keeps_exact_source(self):
        X,q,D,w,n=self.data()
        for kind in CORRECTIONS:
            x=X[:,:1] if kind=='global' else X;m=fit_shrinkage(x,q,D*0,w,n,kind)
            check_shrinkage(x,q,D*0,w,n,kind,m);self.assertEqual(m['coefficients'],m['prior'])
            for value in [0.,.3,1.]:self.assertEqual(correct_q(m,x[0],value),value)
    def test_weight_and_payoff_units(self):
        X,q,D,w,n=self.data()
        for k in CORRECTIONS:
            x=X[:,:1] if k=='global' else X;a=fit_shrinkage(x,q,D,w,n,k);b=fit_shrinkage(x,q,D*7,w*13,n,k)
            np.testing.assert_allclose(a['coefficients'],b['coefficients'],atol=1e-7)
    def test_conditioned_slope_cannot_amplify_or_create_neutral_direction(self):
        X,q,D,w,n=self.data();m=fit_shrinkage(X,q,D,w,n,'state')
        for u in [0.,.3,.5,.9,1.]:
            for value in [0.,.01,.2,.5,.8,.99,1.]:
                now=correct_q(m,[0,u],value);self.assertLessEqual(abs(now-.5),abs(value-.5)+1e-12)
            self.assertEqual(correct_q(m,[0,u],.5),.5)
    def test_additive_control_can_predict_from_state_with_neutral_source(self):
        X,q,D,w,n=self.data();q[:]=.5;D=np.where(X[:,1]>.5,1.,-1.)
        m=fit_shrinkage(X,q,D,w,n,'additive');self.assertLess(correct_q(m,[0,0],.5),.5);self.assertGreater(correct_q(m,[0,1],.5),.5)
    def test_tampered_or_outside_box_coefficients_rejected(self):
        X,q,D,w,n=self.data()
        for value in [.3,-1.01,1.01]:
            m=fit_shrinkage(X,q,D,w,n,'state');m['coefficients'][0]=value
            with self.assertRaises(AssertionError):check_shrinkage(X,q,D,w,n,'state',m)
    def test_frozen_v51_control_exact(self):
        X,q,D,w,n=self.data();x=X[:,:1]
        self.assertEqual(fit_shrinkage(x,q,D,w,n,'global'),prior.fit_shrinkage(x,q,D,w,n,'signed'))
    def test_state_formula_bounds_and_future_immunity(self):
        p=self.p.copy();r=add_state(p);later=p.index>self.at;p.loc[later,STATE_INPUTS]=.9
        b=add_state(p);pd.testing.assert_series_equal(r.loc[:self.at,'state_heat'],b.loc[:self.at,'state_heat'])
        for t in r.index:
            row=r.loc[t];expected=.5+(np.tanh(row.relative_5)+row.breadth_5+row.volume_direction_diffusion)/6
            self.assertAlmostEqual(row.state_heat,expected,places=14);self.assertTrue(0<=row.state_heat<=1)
    def test_missing_state_does_not_become_neutral(self):
        p=self.p.copy();p.loc[self.at,'relative_5']=np.nan;p.loc[self.at,'feature_ready']=False;p=add_state(p)
        self.assertFalse(p.loc[self.at,'state_ready']);self.assertTrue(pd.isna(p.loc[self.at,'state_heat']))
        self.assertFalse(issue(p,self.records,self.at,self.day)[2]['state']['ready'])
    def test_joint_explicit_repeated_dates(self):
        m,q,n,nq=daily_check(self.p,self.market,self.records,self.h,self.at,json.loads(wire(self.day)))
        self.assertEqual(n,2000);self.assertEqual(nq,600);self.assertEqual(set(m),set(VARIANTS));self.assertEqual(set(q),set(CORRECTIONS))
    def test_future_labels_inputs_and_source_predictions_immune(self):
        p=self.p.copy();p.loc[p.index>=self.at,['C','D']]=900;p.loc[p.index>self.at,STATE_INPUTS]=.8;p=add_state(p)
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
        for k in CORRECTIONS+['baseline']:self.assertEqual(detail[k]['raw'],r[at]['fits']['full']['raw']);self.assertFalse(est[k]['ready'])
        self.assertEqual(detail['neutral']['raw']['q'],.5);self.assertFalse(est['neutral']['ready'])
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
        raw,_,est,direct=issue(self.p,self.records,self.at,self.day);k='state'
        x,q=q_inputs(self.p,self.records,[self.at],k);source=self.records[self.at]['fits']['full']['raw']
        draw=np.array([[payoff(source,correct_q(m,x[0],q[0]))[s] for s in core.ETFS] for m in self.day['offsets'][k]['bootstrap_models']])
        self.assertGreater(float(draw[:,0].std()),1e-5)
        row=np.array([*draw[0],CLOCKS.index(self.p.loc[self.at,'clock'])]);cal=self.day['variants'][k]['bootstrap_models'][0]
        np.testing.assert_allclose(est[k]['bootstrap_predictions'][0],draw[0]+forecast(cal,row)[0],atol=1e-12)
        self.assertNotEqual(est[k]['bootstrap_predictions'][0],(raw[k]+forecast(cal,np.r_[raw[k],row[2]])[0]).tolist())
    def test_neutral_control_erases_raw_direction_not_common_payoff(self):
        raw,detail,_,_=issue(self.p,self.records,self.at,self.day)
        self.assertEqual(detail['neutral']['raw']['q'],.5);self.assertEqual(detail['neutral']['raw']['D'],0.)
        self.assertEqual(raw['neutral'][0],raw['neutral'][1]);self.assertEqual(raw['neutral'][0],self.records[self.at]['fits']['full']['raw']['C'])
    def test_negative_payoff_abstains_despite_relative_direction(self):
        e=decide({'ready':True,'point':[-.01,-.08],'lower':[-.02,-.09]}, {'expired_unresolved_prefix':False},True)
        self.assertEqual(e['action'],'CASH')
    def test_failed_primary_cannot_open_cases(self):
        from run_state import authorize_cases
        p=read(SOURCE/'protocol.json')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);s={'primary_average':0.,'completed_trades':0,'all_months_nonnegative':True,'monthly':[{'month':m,'return':0.,'mark_complete':True} for m in p['development']['months']]}
            dump(root/'registration.json',{'phase':'development','protocol_sha256':sha(SOURCE/'protocol.json')})
            dump(root/'results.json',{'primary_name':p['primary'],'primary':s,'routes':{p['primary']:s},'coverage':{'complete':True}})
            dump(root/'verification.json',{'passed':True,'results_sha256':sha(root/'results.json')})
            with self.assertRaisesRegex(ValueError,'Development primary failed'):authorize_cases(root,p)


if __name__=='__main__':unittest.main()
