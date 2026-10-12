import unittest
from regime_learning import *
from audit_regime import reference_states,check_estimate


class RegimeTests(unittest.TestCase):
    def features(self,n=90):
        days=pd.bdate_range('2025-01-02',periods=n);rng=np.random.default_rng(29)
        ats=[pd.Timestamp(str(d.date())+' 09:25:30',tz=core.ET).tz_convert('UTC') for d in days]
        return pd.DataFrame({'date':days,'clock':CLOCKS[0],'SOXX_ret1':rng.normal(0,.02,n),'QQQ_ret1':rng.normal(0,.01,n),
            'SOXX_vol20':.02,'QQQ_vol20':.01,'breadth_ret1_positive_fraction':rng.integers(0,9,n)/8},index=ats),days

    def test_incremental_parameters_match_suffix_sufficient_statistics(self):
        f,days=self.features();a=filter_states(f,days);b=reference_states(f,days)
        for day in days:np.testing.assert_allclose(a[day]['posterior'],b[day]['posterior'],atol=1e-12)

    def test_append_future_cannot_change_old_posteriors(self):
        f,days=self.features();a=filter_states(f.iloc[:40],days[:40]);f.iloc[40:,f.columns.get_loc('SOXX_ret1')]=999
        b=filter_states(f,days)
        for day in days[:40]:np.testing.assert_array_equal(a[day]['posterior'],b[day]['posterior'])

    def test_four_clocks_do_not_create_four_observations(self):
        f,days=self.features();other=f.copy();other['clock']=CLOCKS[1];other.index+=pd.Timedelta(minutes=40);other['SOXX_ret1']=999
        a=filter_states(f,days);b=filter_states(pd.concat([f,other]).sort_index(),days)
        for day in days:np.testing.assert_array_equal(a[day]['posterior'],b[day]['posterior'])

    def test_missing_official_date_resets_without_bridging(self):
        f,days=self.features();f=f.drop(f.index[40]);s=filter_states(f,days)
        self.assertFalse(s[days[40]]['ready']);self.assertIsNone(s[days[40]]['posterior'])
        self.assertFalse(s[days[41]]['ready']);self.assertEqual(s[days[41]]['consecutive_observations'],1)

    def test_filter_detects_a_synthetic_mean_shift(self):
        f=RunFilter()
        for _ in range(100):old=f.update([0,0,0])
        for _ in range(5):new=f.update([4,4,2])
        self.assertGreater(sum(new[:5]),sum(old[:5]));self.assertLess(new@np.arange(len(new)),old@np.arange(len(old)))

    def test_partial_missing_coordinate_resets_and_preserves_missingness(self):
        f,days=self.features();f.iloc[40,f.columns.get_loc('breadth_ret1_positive_fraction')]=np.nan
        a=filter_states(f,days);b=reference_states(f,days)
        for day in days:
            np.testing.assert_allclose(a[day]['coordinates'],b[day]['coordinates'],equal_nan=True)
            if a[day]['posterior'] is None:self.assertIsNone(b[day]['posterior'])
            else:np.testing.assert_allclose(a[day]['posterior'],b[day]['posterior'],atol=1e-12)
        self.assertFalse(a[days[40]]['ready']);self.assertEqual(a[days[41]]['consecutive_observations'],1)

    def test_zero_run_mass_is_hazard_not_change_detection_score(self):
        f=RunFilter()
        for x in [[0,0,0],[4,4,2],[-5,-5,-2]]:
            p=f.update(x);self.assertAlmostEqual(p[0],1/63);self.assertAlmostEqual(p.sum(),1.)

    def test_membership_uses_observation_dates_and_normalizes_weights(self):
        f,days=self.features();states=filter_states(f,days);p=f.copy();ids=list(f.index[:65])
        for mode in ['regime','hazard']:
            w,m,ages=memory_weights(ids,p,states,days[-1],mode)
            self.assertAlmostEqual(w.sum(),65);self.assertTrue(((m>=0)&(m<=1)).all());self.assertTrue((np.diff(m)>=-1e-12).all())
            np.testing.assert_array_equal(ages,np.arange(89,24,-1))

    def test_weighted_error_models_match_independent_equations(self):
        f,days=self.features();states=filter_states(f,days);p=f.copy();p['feature_ready']=True
        rng=np.random.default_rng(29);p['C']=rng.normal(0,.02,len(p));p['D']=rng.normal(0,.03,len(p))
        h={t:rng.normal(0,.02,2) for t in p.index};at=p.index[-1];ids=list(p.index[:65])
        for mode in ['regime','hazard']:
            info=estimate(p,h,states,at,h[at],ids,mode);self.assertEqual(check_estimate(p,h,states,ids,at,h[at],mode,info),400)

    def test_route_metadata_restores_runtime_timestamps(self):
        at=pd.Timestamp('2026-01-05 14:25:30Z');day=pd.Timestamp('2026-01-05');fb={'expired_unresolved_prefix':False}
        model={'ready':True,'point':[.04,0.],'lower':[.01,-.03]};decision={'feedback':fb,'estimates':{'regime':model,'hazard':model}}
        old={'at':str(at),'date':str(day),'ready':True,'action':'SOXL','allocation':1.}
        e=route_events(at,day,CLOCKS[0],decision,old)
        self.assertIsInstance(e['source_v27']['date'],pd.Timestamp);self.assertEqual(e['source_v27']['at'],at)
        self.assertEqual(e['double_cost'],e['regime_band'])

    def test_new_regime_with_no_mature_examples_cannot_reuse_old_confidence(self):
        f,days=self.features();states=filter_states(f,days);p=f.copy();p['feature_ready']=True;p['C']=.01;p['D']=.02
        h={t:np.array([.03,-.01]) for t in p.index};at=p.index[-1];ids=list(p.index[:65])
        states[days[-1]]['posterior']=np.array([1/63,62/63])
        info=estimate(p,h,states,at,h[at],ids,'regime');self.assertFalse(info['ready'])
        self.assertEqual(info['current_regime_mature_mass'],0);self.assertIsNone(info['point'])


if __name__=='__main__':unittest.main()
