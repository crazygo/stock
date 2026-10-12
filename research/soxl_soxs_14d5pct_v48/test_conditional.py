"""Causal daily training, factorial controls and independent tree audit."""
import unittest
import tempfile
import copy
from threadpoolctl import threadpool_limits
from conditional_options import *
from audit_conditional_models import weighted_equations
from test_options import fixture, wire, raw_fixture
from audit_option_features import audit_option_features
from unittest.mock import patch


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.limits=threadpool_limits(1)
    @classmethod
    def tearDownClass(cls):cls.limits.restore_original_limits()
    def test_independent_base_and_tree_equations(self):
        p=fixture();at=p.index[-1];models,fits,raws=fit_bases(p,at)
        for k in VARIANTS:
            info=fits[k]['fit'];ids=list(pd.to_datetime(info['members'],utc=True));w=np.asarray(info['parent_weights'])
            values,n=weighted_equations(p,ids,at,models[k],info,columns(CLOCKS[0]),w,k)
            self.assertGreaterEqual(n,3)
            np.testing.assert_allclose(raws[k],[values[s] for s in core.ETFS],atol=1e-12)
    def test_future_inputs_and_labels_do_not_rewrite_past(self):
        p=fixture(150);at=p.index[140];a=fit_bases(p,at)
        p.loc[p.index>=at,['C','D']]=999
        p.loc[p.index>at,q_columns(CLOCKS[0],'tree')+option_columns(5)]=999
        b=fit_bases(p,at);self.assertEqual(wire(a[1:]),wire(b[1:]))
        for k in ['tree','delayed','old_tree']:
            for aa,bb in zip(a[0][k]['heads']['q']._predictors,b[0][k]['heads']['q']._predictors):
                np.testing.assert_array_equal(aa[0].nodes,bb[0].nodes)
    def test_unknown_prefix_not_skipped(self):
        p=fixture();p.loc[p.index[50],'terminal_release']=pd.NaT
        models,fits,_=fit_bases(p,p.index[-1]);self.assertTrue(all(m is None for m in models.values()))
        self.assertTrue(all(len(fits[k]['fit']['members'])==50 for k in VARIANTS))
    def test_119_dates_not_ready(self):
        p=fixture(120);models,_,_=fit_bases(p,p.index[-1]);self.assertTrue(all(m is None for m in models.values()))
    def test_C_M_coefficients_and_queries_unchanged(self):
        p=fixture();_,fits,_=fit_bases(p,p.index[-1])
        for k in VARIANTS:
            for h in ['C','M']:
                self.assertEqual(wire(fits[k]['fit']['heads'][h]),wire(fits['baseline']['fit']['heads'][h]))
                self.assertEqual(fits[k]['raw'][h],fits['baseline']['raw'][h])
    def test_singleclass_and_zero_magnitude(self):
        p=fixture()
        for val,q in [(.04,1.),(0.,.5)]:
            p['D']=val;models,fits,_=fit_bases(p,p.index[-1])
            for k in VARIANTS:self.assertNotIn('q',models[k]['heads']);self.assertEqual(fits[k]['raw']['q'],q)
    def test_tree_leaf_tamper_is_rejected(self):
        p=fixture();at=p.index[-1];models,fits,_=fit_bases(p,at);k='tree';info=fits[k]['fit']
        nodes=models[k]['heads']['q']._predictors[0][0].nodes
        nodes['value'][np.flatnonzero(nodes['is_leaf'])[0]]+=.01
        with self.assertRaises(AssertionError):weighted_equations(p,list(pd.to_datetime(info['members'],utc=True)),at,models[k],info,columns(CLOCKS[0]),np.asarray(info['parent_weights']),k)
    def test_current_missing_features_abstain(self):
        p=fixture();p.loc[p.index[-1],'feature_ready']=False;_,_,raws=fit_bases(p,p.index[-1]);self.assertTrue(all(x is None for x in raws.values()))
    def test_uniform_payoff_scaling_preserves_q(self):
        p=fixture();at=p.index[-1];a=fit_bases(p,at);p[['C','D']]*=2;b=fit_bases(p,at)
        for k in VARIANTS:self.assertAlmostEqual(a[1][k]['raw']['q'],b[1][k]['raw']['q'],places=12)
    def test_failed_primary_cannot_open_cases(self):
        from run_conditional import authorize_cases
        p=read(SOURCE/'protocol.json')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stats={'primary_average':0.,'completed_trades':0,'all_months_nonnegative':True,
                'monthly':[{'month':m,'return':0.,'mark_complete':True} for m in p['development']['months']]}
            dump(root/'registration.json',{'phase':'development','protocol_sha256':sha(SOURCE/'protocol.json')})
            dump(root/'results.json',{'primary_name':p['primary'],'primary':stats,'routes':{p['primary']:stats},'coverage':{'complete':True}})
            dump(root/'verification.json',{'passed':True,'results_sha256':sha(root/'results.json')})
            with self.assertRaisesRegex(ValueError,'Development primary failed'):authorize_cases(root,p)
    def test_v47_linear_controls_identical(self):
        import option_learning as old
        p=fixture();at=p.index[-1];a=fit_bases(p,at);b=old.fit_bases(p,at)
        for k in ['options','baseline']:self.assertEqual(wire(a[1][k]),wire(b[1][k]));np.testing.assert_array_equal(a[2][k],b[2][k])
    def test_option_absence_control_columns(self):
        for clock in CLOCKS:
            self.assertEqual(q_columns(clock,'old_tree'),q_columns(clock,'baseline'))
            self.assertEqual(q_columns(clock,'tree'),q_columns(clock,'options'))
            self.assertEqual(q_columns(clock,'delayed')[-8:],option_columns(5))
    def feature_audit(self,tamper=False):
        raw,cal,f=raw_fixture();added,_,receipts=build_features(f,raw,cal['sessions'])
        panel=f.join(added);panel['clock']=CLOCKS[0];panel['feature_ready']=panel.option_ready
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory);source=run/'source';source.mkdir();panel[['feature_ready']].to_parquet(source/'financial_panel.parquet')
            if tamper:panel.loc[panel[panel.option_ready].index[-1],option_columns(2)[0]]+=.01
            panel.to_parquet(run/'option_panel.parquet');added.to_parquet(run/'option_features.parquet')
            dump(run/'option_receipts.json',receipts);dump(run/'input_manifest.json',{'source':str(source),'option_calendar':cal,'option_source':{}})
            with patch('audit_option_features.load_raw',return_value=(raw,{})):return audit_option_features(run)
    def test_independent_raw_windows_handle_mixed_frame(self):self.assertTrue(self.feature_audit()['passed'])
    def test_corrupted_feature_rejected(self):
        with self.assertRaises(AssertionError):self.feature_audit(True)
    def test_positive_relative_direction_is_not_profitable(self):
        e=decide({'ready':True,'point':[-.01,-.08],'lower':[-.02,-.09]}, {'expired_unresolved_prefix':False},True)
        self.assertEqual(e['action'],'CASH')


if __name__=='__main__':unittest.main()
