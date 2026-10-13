"""Numerical loss units, feature ablation and causal readiness regressions."""
import unittest
from history_learning import *
from reference_fit import solve


def sample(n=145):
    rng=np.random.default_rng(77);X=rng.normal(size=(n*3,18));Y=rng.normal(size=(n*3,2))*.02
    pairs=np.repeat(['semiconductor','nasdaq','sp500'],n);dw=2.**(-np.arange(n)[::-1]/126);w=np.tile(dw/3,3)
    return X,Y,pairs,w


class HistoryTests(unittest.TestCase):
    def test_normal_equations_in_different_weight_units(self):
        X,Y,p,w=sample()
        for kind in ['short_price','long_full','long_price','long_native']:
            m=fit_state(X,Y,p,w,145,120,kind);ref=solve(X,Y,p,w,145,120,kind)
            for f in ['centers','scales','intercepts','coefficients','weights','penalty']:
                np.testing.assert_allclose(m[f],ref[f],atol=1e-11,rtol=1e-10)

    def test_native_changes_penalty_strength_not_weighted_mean(self):
        X,Y,p,w=sample();a=fit_state(X,Y,p,w,145,120,'long_price');b=fit_state(X,Y,p,w,145,120,'long_native')
        np.testing.assert_allclose(a['intercepts'],b['intercepts'],atol=1e-13)
        np.testing.assert_allclose(a['penalty'],b['penalty']*145/120,atol=1e-13)
        self.assertGreater(np.max(abs(a['coefficients']-b['coefficients'])),1e-8)

    def test_equal_mass_reproduces_original_linear_fit(self):
        X,Y,p,w=sample();a=fit_state(X,Y,p,w,145,145,'long_full');b=old.fit(X,Y,p,w,145)
        for f in ['centers','scales','intercepts','coefficients','weights','penalty']:np.testing.assert_array_equal(a[f],b[f])

    def test_activity_values_cannot_change_price_model(self):
        X,Y,p,w=sample();a=fit_state(X,Y,p,w,145,120,'long_price');changed=X.copy();changed[:,11:16]+=100
        b=fit_state(changed,Y,p,w,145,120,'long_price')
        np.testing.assert_array_equal(a['coefficients'],b['coefficients'])
        np.testing.assert_allclose(a['coefficients'][:,11:16],0,atol=1e-13)
        self.assertTrue(np.array_equal(mask_features(X,'long_price')[:,[10,16,17]],X[:,[10,16,17]]))

    def test_no_missing_coordinate_imputation(self):
        X,Y,p,w=sample();X[0,11]=np.nan
        with self.assertRaises(ValueError):fit_state(X,Y,p,w,145,120,'long_price')
        for mass in [0,-1,np.nan]:
            with self.assertRaises(ValueError):fit_state(np.nan_to_num(X),Y,p,w,145,mass,'long_price')

    def test_row_replication_does_not_add_information(self):
        X,Y,p,w=sample();a=fit_state(X,Y,p,w,145,120,'long_price')
        b=fit_state(np.repeat(X,2,axis=0),np.repeat(Y,2,axis=0),np.repeat(p,2),np.repeat(w/2,2),145,120,'long_price')
        np.testing.assert_allclose(a['coefficients'],b['coefficients'],atol=1e-12,rtol=1e-10)
        self.assertAlmostEqual(sum(b['weights']),145)

    def test_feature_unready_query_never_issues(self):
        row=pd.Series({'feature_ready':False});self.assertIsNone(raw_prediction({'variant':'long_mean','mean_cd':[.1,.2]},row,'long_mean'))
        self.assertIsNone(raw_prediction(None,row,'long_price'))

    def test_full_maturity_stops_before_future_or_unknown(self):
        at=pd.Timestamp('2026-01-20 09:25:30',tz=core.ET).tz_convert('UTC')
        rows=[{'date':pd.Timestamp(d),'members':[],'feature_ready':True,'release':pd.Timestamp(rel,tz='UTC') if rel else None}
              for d,rel in [('2026-01-01','2026-01-14'),('2026-01-02',None),('2026-01-03','2026-01-19')]]
        parents,info=mature_prefix(rows,at);self.assertEqual(len(parents),1);self.assertTrue(info['expired_unresolved_prefix'])

    def test_variant_feedback_and_primary_controls(self):
        from run_history import events
        at=pd.Timestamp('2026-01-20 09:25:30',tz=core.ET).tz_convert('UTC')
        row=pd.Series({'date':pd.Timestamp('2026-01-20'),'clock':CLOCKS[0]})
        estimates={k:{'ready':True,'point':[.03,.04],'lower':[-.01,.02]} for k in VARIANTS}
        daily={k:{'feedback':{'expired_unresolved_prefix':False}} for k in VARIANTS}
        feedback={'long':{'expired_unresolved_prefix':True},'short':{'expired_unresolved_prefix':False}}
        e=events(at,row,estimates,daily,feedback,None)
        self.assertEqual(set(e),set(ROUTES));self.assertEqual(e['short_price_band']['action'],'SOXS')
        self.assertFalse(e['long_price_band']['ready']);self.assertEqual(e['premarket_only'],e['long_price_band'])
        self.assertEqual(e['double_cost'],e['long_price_band']);self.assertEqual(e['short_full_band']['action'],'CASH')
        feedback['long']['expired_unresolved_prefix']=False
        e=events(at,row,estimates,daily,feedback,None);self.assertEqual(e['long_price_band']['action'],'SOXS')
        daily['long_price']['feedback']['expired_unresolved_prefix']=True
        self.assertEqual(events(at,row,estimates,daily,feedback,None)['long_price_band']['action'],'CASH')

    def test_price_query_ignores_activity_values(self):
        X,Y,p,w=sample();m=fit_state(X,Y,p,w,145,120,'long_price')
        row=pd.Series({'feature_ready':True,**dict(zip(FIELDS,X[0]))});altered=row.copy()
        altered.loc[ACTIVITY_FIELDS]=[100,200,-20,5,-6]
        np.testing.assert_array_equal(raw_prediction(m,row,'long_price'),raw_prediction(m,altered,'long_price'))


if __name__=='__main__':unittest.main()
