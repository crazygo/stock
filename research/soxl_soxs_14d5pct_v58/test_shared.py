"""Convex objective, weight units, sharing invariance and causal prefix tests."""
import unittest,json
from copy import deepcopy
from scipy.optimize import minimize
from sklearn.linear_model import Ridge,LogisticRegression
from shared_learning import *
from audit_base import reference_penalty,check_model
from run_planning import json_ready


def arrays_fixture(seed=81):
    rng=np.random.default_rng(seed);n=140
    z=[rng.normal(size=(n,2)),rng.normal(size=(n,3)),rng.normal(size=(n,3))]
    weights=[np.linspace(.7,1.3,n)*v for v in [.2,.3,.5]]
    d=[rng.normal(.005,.05,n)+a[:,0]*.01 for a in z]
    return {'Z':z,'weights':weights,'C':[rng.normal(0,.02,n) for _ in z],'D':d,'columns':[['a','b'],['a','b','x'],['a','b','x']],'eta':[.2,.3,.5],'N':n}


class SharedTests(unittest.TestCase):
    def test_penalty_independent_outer_product_formula(self):
        a=arrays_fixture()
        for k in VARIANTS:
            for strength in [10.,100.]:np.testing.assert_allclose(penalty(a['columns'],a['eta'],k,strength),reference_penalty(a['columns'],a['eta'],k,strength),atol=1e-12)

    def test_penalty_positive_definite(self):
        a=arrays_fixture()
        for k in VARIANTS:self.assertGreater(np.linalg.eigvalsh(penalty(a['columns'],a['eta'],k,10.)).min(),0)

    def test_independent_matches_separate_sklearn_heads(self):
        a=arrays_fixture();model=fit_variant(a,'independent');mapping,_,_=parameter_layout(a['columns'],'independent');cw,active,_=classification_arrays(a)
        for c,(z,w) in enumerate(zip(a['Z'],a['weights'])):
            for key,y in [('C',a['C'][c]),('M',abs(a['D'][c]))]:
                ref=Ridge(alpha=100.,solver='svd').fit(z,y,sample_weight=w/a['eta'][c]);np.testing.assert_allclose(np.array(model[key]['coefficients'])[mapping[c]],ref.coef_,atol=1e-10);self.assertAlmostEqual(model[key]['intercepts'][c],ref.intercept_,places=10)
            ref=LogisticRegression(C=.1,solver='lbfgs',tol=1e-11,max_iter=5000).fit(z,(a['D'][c]>0).astype(int),sample_weight=cw[c]/a['eta'][c])
            np.testing.assert_allclose(np.array(model['q']['coefficients'])[mapping[c]],ref.coef_[0],atol=1e-7);self.assertAlmostEqual(model['q']['intercepts'][c],ref.intercept_[0],places=7)

    def test_all_heads_independent_equations(self):
        a=arrays_fixture()
        for k in VARIANTS:self.assertEqual(check_model(a,k,fit_variant(a,k)),3)

    def test_alternative_optimizer_same_partial_solution(self):
        a=arrays_fixture();kind='partial';model=fit_variant(a,kind);mapping,size,_=parameter_layout(a['columns'],kind);P=reference_penalty(a['columns'],a['eta'],kind,10.)
        cw,active,_=classification_arrays(a);matrix=np.zeros((420,size+3));offset=0
        for c,z in enumerate(a['Z']):
            matrix[offset:offset+140,mapping[c]]=z;matrix[offset:offset+140,size+c]=1;offset+=140
        w=np.concatenate(cw);y=np.concatenate([x>0 for x in a['D']]).astype(float)
        def fun(theta):
            linear=matrix@theta;coef=theta[:size];value=w@(np.logaddexp(0,linear)-y*linear)+.5*coef@P@coef;grad=matrix.T@(w*(expit(linear)-y));grad[:size]+=P@coef
            return value,grad
        solved=minimize(fun,np.zeros(size+3),jac=True,method='L-BFGS-B',options={'gtol':1e-10,'ftol':1e-14,'maxiter':5000})
        self.assertTrue(solved.success);np.testing.assert_allclose(np.r_[model['q']['coefficients'],model['q']['intercepts']],solved.x,atol=2e-6)

    def test_logistic_hessian_finite_difference(self):
        a=arrays_fixture();k='partial';mapping,size,_=parameter_layout(a['columns'],k);cw,active,_=classification_arrays(a);P=penalty(a['columns'],a['eta'],k,10.);theta=np.linspace(-.1,.1,size+3)
        _,g,h=logistic_objective(theta,a,k,P,mapping,cw,active)
        eps=1e-5
        for j in range(len(theta)):
            d=np.eye(len(theta))[j]*eps
            gp=logistic_objective(theta+d,a,k,P,mapping,cw,active,False)[1];gm=logistic_objective(theta-d,a,k,P,mapping,cw,active,False)[1]
            np.testing.assert_allclose(h[:,j],(gp-gm)/(2*eps),atol=1e-7)

    def test_duplicate_channels_do_not_change_solution(self):
        a=arrays_fixture();b={key:[x for value in a[key] for x in [value,value]] for key in ['Z','C','D','columns']}
        b.update(weights=[w/2 for w in a['weights'] for _ in range(2)],eta=[x/2 for x in a['eta'] for _ in range(2)],N=a['N'])
        for k in VARIANTS:
            one=fit_variant(a,k);two=fit_variant(b,k);i,_,_=parameter_layout(a['columns'],k);j,_,_=parameter_layout(b['columns'],k)
            for head in ['C','M','q']:
                for c in range(3):
                    np.testing.assert_allclose(np.array(one[head]['coefficients'])[i[c]],np.array(two[head]['coefficients'])[j[2*c]],atol=1e-9)
                    self.assertAlmostEqual(one[head]['intercepts'][c],two[head]['intercepts'][2*c],places=9)

    def test_absent_feature_has_no_premarket_parameter(self):
        a=arrays_fixture();mapping,_,names=parameter_layout(a['columns'],'shared')
        self.assertNotIn(names.index('x'),mapping[0]);self.assertIn(names.index('x'),mapping[1])

    def test_all_one_class_exact_fallback(self):
        a=arrays_fixture();a['D']=[abs(x)+.01 for x in a['D']]
        for k in VARIANTS:
            m=fit_variant(a,k);self.assertEqual(m['q']['fallback_q'],[1.,1.,1.]);self.assertEqual(m['q']['mixed_classes'],[False]*3);np.testing.assert_array_equal(m['q']['coefficients'],np.zeros(len(m['q']['coefficients'])));check_model(a,k,m)

    def test_zero_magnitude_neutral_fallback(self):
        a=arrays_fixture();a['D']=[np.zeros_like(x) for x in a['D']];m=fit_variant(a,'partial')
        self.assertEqual(m['q']['fallback_q'],[.5]*3);np.testing.assert_allclose(m['M']['coefficients'],0);check_model(a,'partial',m)

    def test_mixed_and_degenerate_clocks(self):
        a=arrays_fixture();a['D'][0]=abs(a['D'][0])+.01;m=fit_variant(a,'partial')
        self.assertEqual(m['q']['mixed_classes'],[False,True,True]);self.assertEqual(m['q']['fallback_q'][0],1.);check_model(a,'partial',m)

    def test_amount_weights_keep_clock_mass(self):
        a=arrays_fixture();w,_,_=classification_arrays(a)
        for before,after in zip(a['weights'],w):self.assertAlmostEqual(sum(before),sum(after),places=10)

    def panel_fixture(self):
        rng=np.random.default_rng(930);days=pd.bdate_range('2025-01-02',periods=127);fields=list(dict.fromkeys(f for c in CLOCKS for f in columns(c)));ids=[];dates=[];clocks=[];reg=[]
        for day in days:
            block=[pd.Timestamp(str(day.date())+' '+c,tz=core.ET).tz_convert('UTC') for c in CLOCKS];ids+=block;dates += [day]*8;clocks+=CLOCKS
            reg.append({'date':day,'members':block,'feature_ready':True,'release':max(block)+pd.Timedelta(days=1)})
        p=pd.DataFrame(rng.normal(size=(len(ids),len(fields))),index=pd.DatetimeIndex(ids),columns=fields)
        p['date']=dates;p['clock']=clocks;p['feature_ready']=True;p['C']=rng.normal(0,.01,len(p));p['D']=rng.normal(0,.05,len(p));p['M']=abs(p.D);p['terminal_release']=p.index+pd.Timedelta(days=1)
        return p,reg,ids[-8]

    def test_future_targets_do_not_change_today_fit(self):
        p,reg,at=self.panel_fixture();a=fit_base_day(p,reg,at);changed=p.copy();changed.loc[changed.date>=p.loc[at,'date'],['C','D']]=999
        b=fit_base_day(changed,reg,at);self.assertEqual(json_ready(a),json_ready(b))

    def test_today_later_inputs_not_used_at_midnight(self):
        p,reg,at=self.panel_fixture();a=fit_base_day(p,reg,at);changed=p.copy();changed.loc[changed.index>at,columns(CLOCKS[-1])]=999
        b=fit_base_day(changed,reg,at);self.assertEqual(json_ready(a),json_ready(b));self.assertEqual(predict_base(p,at,a,'partial')[0],predict_base(changed,at,b,'partial')[0])

    def test_date_weights_not_multiplied_by_clock_count(self):
        p,reg,at=self.panel_fixture();parents,_=base_pool(reg,at);meta,_=prepare_arrays(p,parents)
        self.assertAlmostEqual(sum(meta['row_weights']),len(parents));self.assertAlmostEqual(sum(meta['eta']),1.)

    def test_constant_columns_scale_to_one(self):
        p,reg,at=self.panel_fixture();p[columns(CLOCKS[0])[0]]=3.;parents,_=base_pool(reg,at);meta,_=prepare_arrays(p,parents)
        for fields,scales in zip(meta['columns'],meta['scale']):self.assertEqual(scales[fields.index(columns(CLOCKS[0])[0])],1.)

    def test_missing_late_clock_minimum_keeps_not_ready(self):
        p,reg,at=self.panel_fixture();parents=deepcopy(reg[:120]);parents[0]['members']=parents[0]['members'][:-3]
        meta,_=prepare_arrays(p,parents);self.assertFalse(meta['ready']);self.assertEqual(meta['clock_rows'][-1],119)

    def test_no_skip_unknown_prior_parent(self):
        p,reg,at=self.panel_fixture();reg[5]['release']=None;parents,fb=base_pool(reg,at)
        self.assertEqual(len(parents),5);self.assertEqual(fb['first_pending_date'],reg[5]['date']);self.assertTrue(fb['expired_unresolved_prefix'])

    def test_unready_current_features_abstain(self):
        p,reg,at=self.panel_fixture();base=fit_base_day(p,reg,at);p.loc[at,'feature_ready']=False
        self.assertEqual(predict_base(p,at,base,'partial'),({},None))

    def test_json_model_roundtrip_prediction(self):
        p,reg,at=self.panel_fixture();base=fit_base_day(p,reg,at);decoded=json.loads(json.dumps(json_ready(base),default=str))
        self.assertEqual(predict_base(p,at,base,'partial')[0],predict_base(p,at,decoded,'partial')[0])

    def test_contract_and_negative_month_gate_unchanged(self):
        p=read(SOURCE/'protocol.json');old=read(core.ROOT/'research/soxl_soxs_14d5pct_v56/protocol.json')
        for key in ['acceptance','capital','contract','development','cases']:self.assertEqual(p[key],old[key])
        stats={'completed_trades':30,'monthly':[{'month':m,'return':x,'mark_complete':True} for m,x in zip(p['development']['months'],[.3,.3,-.01,.3])]}
        self.assertFalse(development_gate(stats,True,p)['passed'])


if __name__=='__main__':unittest.main()
