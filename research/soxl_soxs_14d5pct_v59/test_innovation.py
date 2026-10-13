"""Causality, missing sessions, independent equations and representation tests."""
import unittest,json
from sklearn.linear_model import Ridge
from innovation_learning import *
from audit_state_features import check_record
from run_planning import json_ready


def canonical(x):return json.loads(json.dumps(json_ready(x),default=str))


def fixture(n=70,clock=CLOCKS[0]):
    days=pd.bdate_range('2025-01-02',periods=n);idx=pd.DatetimeIndex([pd.Timestamp(str(d.date())+' '+clock,tz=core.ET).tz_convert('UTC') for d in days]);rng=np.random.default_rng(95)
    p=pd.DataFrame(rng.normal(size=(n,len(columns(clock)))),columns=columns(clock),index=idx);p['date']=days;p['clock']=clock;p['feature_ready']=True;p['C']=rng.normal(0,.01,n);p['D']=rng.normal(0,.05,n);p['M']=abs(p.D);p['terminal_release']=idx+pd.Timedelta(days=10)
    return p,days


class InnovationTests(unittest.TestCase):
    def test_minimum_sixty_complete_transitions(self):
        p,days=fixture(63);_,rec=build_state_features(p,days)
        self.assertFalse(rec[60]['fit']['ready']);self.assertTrue(rec[61]['fit']['ready']);self.assertEqual(rec[61]['fit']['n'],60)

    def test_independent_design_equations(self):
        p,days=fixture();out,recs=build_state_features(p,days)
        for i in [0,20,60,61,69]:
            value,n=check_record(p,days,p.index[i],canonical(recs[i]));self.assertEqual(n,3 if i>=61 else 0)
            if value is not None:np.testing.assert_allclose(value,out.iloc[i][INNOV+DIFF].to_numpy(float),atol=1e-10)

    def test_sklearn_multivariate_reference(self):
        p,days=fixture();pairs=state_pairs(p,days);fit=fit_state(p,[v for v in pairs[:-1] if v['ready']]);x=(p.loc[fit['lag_members'],STATES].to_numpy()-fit['input_center'])/fit['input_scale'];y=p.loc[fit['members'],STATES].to_numpy()
        ref=Ridge(alpha=20.,solver='svd').fit(x,y,sample_weight=fit['weights']);np.testing.assert_allclose(ref.coef_.T,fit['coefficients'],atol=1e-10);np.testing.assert_allclose(ref.intercept_,fit['output_center'],atol=1e-10)

    def test_future_append_preserves_past(self):
        p,days=fixture(75);a,ra=build_state_features(p.iloc[:70],days);b,rb=build_state_features(p,days)
        pd.testing.assert_frame_equal(a,b.iloc[:70]);self.assertEqual(canonical(ra),canonical(rb[:70]))

    def test_future_labels_do_not_affect_state_model(self):
        p,days=fixture();_,a=build_state_features(p,days);changed=p.copy();changed[['C','D','M']]=999.;changed['terminal_release']=pd.NaT
        _,b=build_state_features(changed,days);self.assertEqual(canonical(a),canonical(b))

    def test_today_state_changes_output_not_fit(self):
        p,days=fixture();_,a=build_state_features(p,days);q=p.copy();q.loc[q.index[-1],STATES]=99.;_,b=build_state_features(q,days)
        self.assertEqual(canonical(a[-1]['fit']),canonical(b[-1]['fit']));self.assertNotEqual(a[-1]['query'],b[-1]['query'])

    def test_future_inputs_do_not_change_prior_feature(self):
        p,days=fixture(75);a,ra=build_state_features(p,days);q=p.copy();q.loc[q.index[-4:],STATES]=999.;b,rb=build_state_features(q,days)
        pd.testing.assert_frame_equal(a.iloc[:-4],b.iloc[:-4]);self.assertEqual(canonical(ra[:-4]),canonical(rb[:-4]))

    def test_missing_previous_session_not_bridged(self):
        p,days=fixture();missing=p.index[-2];q=p.drop(index=missing);out,recs=build_state_features(q,days)
        self.assertFalse(recs[-1]['pair']['ready']);self.assertEqual(recs[-1]['pair']['lag_at'],missing);self.assertFalse(out.iloc[-1].state_ready)

    def test_halfday_late_clock_not_bridged(self):
        p,days=fixture(clock=CLOCKS[-1]);q=p.drop(index=p.index[-2]);out,recs=build_state_features(q,days)
        self.assertFalse(out.iloc[-1].state_ready);self.assertEqual(recs[-1]['pair']['lag_at'],p.index[-2]);check_record(q,days,q.index[-1],canonical(recs[-1]))

    def test_missing_current_state_kept_unknown(self):
        p,days=fixture();p.loc[p.index[-1],STATES[0]]=np.nan;out,recs=build_state_features(p,days)
        self.assertFalse(out.iloc[-1].feature_ready);self.assertTrue(out.iloc[-1][INNOV+DIFF].isna().all())

    def test_missing_old_pair_retained_as_ineligible(self):
        p,days=fixture(75);p.loc[p.index[5],STATES[0]]=np.nan;_,recs=build_state_features(p,days)
        self.assertNotIn(p.index[5],recs[-1]['fit']['members']);self.assertNotIn(p.index[6],recs[-1]['fit']['members']);self.assertEqual(recs[-1]['fit']['n'],71)

    def test_constant_state_finite_and_zero_surprise(self):
        p,days=fixture();p[STATES]=3.;out,recs=build_state_features(p,days);self.assertEqual(recs[-1]['fit']['input_scale'],[1.]*3);self.assertEqual(recs[-1]['fit']['residual_scale'],[1.]*3);self.assertEqual(recs[-1]['fit']['difference_scale'],[1.]*3);np.testing.assert_allclose(out.iloc[-1][INNOV+DIFF].to_numpy(float),0,atol=1e-12)

    def test_weights_have_one_date_unit(self):
        p,days=fixture();_,recs=build_state_features(p,days);fit=recs[-1]['fit'];self.assertAlmostEqual(sum(fit['weights']),fit['n'],places=10)

    def test_only_previous_clock_inputs_used(self):
        p,days=fixture();later=p.copy();later.index=later.index+pd.Timedelta(hours=1);later['clock']='10:25:30';both=pd.concat([p,later]).sort_index()
        # Only official registered clocks are supported, and unrelated later data
        # cannot enter the exact same-clock pairs passed to the state fit.
        pairs=state_pairs(both,days);ours=[x for x in pairs if x['clock']==CLOCKS[0] and x['date']<days[-1] and x['ready']]
        before=fit_state(both,ours);both.loc[later.index,STATES]=999;after=fit_state(both,ours);self.assertEqual(before,after)

    def test_three_variants_same_current_readiness(self):
        p,days=fixture();derived,_=build_state_features(p,days)
        self.assertEqual([len(columns(CLOCKS[0],k)) for k in VARIANTS],[36,36,33]);self.assertEqual([len(columns(CLOCKS[-1],k)) for k in VARIANTS],[48,48,45]);self.assertTrue(set(STATES)<=set(columns(CLOCKS[0])))
        self.assertEqual(int(derived.feature_ready.sum()),9)

    def test_fixed_transform_affine_span(self):
        p,days=fixture();out,recs=build_state_features(p,days);fit=recs[-1]['fit'];now=p.iloc[-1][STATES].to_numpy(float);delta=out.iloc[-1][DIFF].to_numpy(float)
        lag=now-delta*np.array(fit['difference_scale']);pred=np.array(fit['output_center'])+((lag-fit['input_center'])/fit['input_scale'])@fit['coefficients'];want=(now-pred)/fit['residual_scale'];np.testing.assert_allclose(want,out.iloc[-1][INNOV].to_numpy(float),atol=1e-12)

    def test_unsorted_input_rejected(self):
        p,days=fixture()
        with self.assertRaises(AssertionError):build_state_features(p.iloc[::-1],days)

    def test_expired_base_prefix_abstains_even_when_calibration_ready(self):
        at=pd.Timestamp('2026-01-05 14:25:30Z');info={k:{'ready':True,'point':[.03,-.01],'lower':[.01,-.03]} for k in VARIANTS};daily={k:{'feedback':{'expired_unresolved_prefix':False}} for k in VARIANTS};fits={k:{'feedback':{'expired_unresolved_prefix':True}} for k in VARIANTS};cash={'ready':True,'action':'CASH','allocation':0.}
        es=events(at,pd.Timestamp('2026-01-05'),CLOCKS[0],info,daily,fits,cash,cash)
        for name in ['innovation_band','innovation_point','difference_band','matched_band']:self.assertFalse(es[name]['ready']);self.assertEqual(es[name]['action'],'CASH')

    def test_contract_and_acceptance_preserved(self):
        p=read(SOURCE/'protocol.json');old=read(core.ROOT/'research/soxl_soxs_14d5pct_v58/protocol.json')
        for key in ['acceptance','capital','contract','development','cases']:self.assertEqual(p[key],old[key])
        s={'completed_trades':30,'monthly':[{'month':m,'return':v,'mark_complete':True} for m,v in zip(p['development']['months'],[.3,.3,-.01,.3])]};self.assertFalse(development_gate(s,True,p)['passed'])


if __name__=='__main__':unittest.main()
