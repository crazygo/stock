import unittest
import numpy as np
import pandas as pd
from paired_entry import (core, BASE_COLUMNS, REGIME_COLUMNS, factor_frame, PANEL,
                          fit, paired_calibration, decision)


class PairedTests(unittest.TestCase):
    def test_regret_is_payoff_weighted_not_win_frequency(self):
        d = np.array([.04, .05, -.10]); q = np.average(d>0, weights=np.abs(d))
        self.assertGreater((d>0).mean(), .5); self.assertLess(q, .5)
        self.assertAlmostEqual(np.abs(d).mean()*(2*q-1), d.mean())

    def test_future_features_and_volume_cannot_change_today(self):
        dates = pd.bdate_range('2025-01-01', periods=35)
        f = pd.DataFrame(index=dates[20:].tz_localize(core.ET)+pd.Timedelta(hours=9, minutes=25, seconds=30))
        f['date'] = dates[20:]
        for c in ['SOXX_vol20','QQQ_vol20']: f[c]=.02
        for c in ['SOXX_ret1','SOXX_ret5','SOXX_ret20','QQQ_ret5','SOXX_today_return',
                  'breadth_today_mean','sector_relative_5','sector_relative_20']: f[c]=.01
        for c in ['breadth_ret5_positive_fraction','breadth_ret20_positive_fraction','breadth_today_positive_fraction']:f[c]=.75
        for s in PANEL: f['leader_'+s+'_today']=.01; f['leader_'+s+'_ret5']=.05
        v = pd.DataFrame(100.,index=dates,columns=PANEL); v.loc[dates[27]]=200.
        before = factor_frame(f,v); self.assertTrue(before.feature_ready.iloc[7])
        self.assertAlmostEqual(before.volume_log_mean.iloc[7],np.log(2))
        self.assertEqual(before.volume_direction_diffusion.iloc[7],1.)
        f.loc[f.index[8]:, f.columns!='date'] = 1e6; v.loc[dates[28]:]=1e12
        after=factor_frame(f,v)
        pd.testing.assert_frame_equal(before.iloc[:8], after.iloc[:8])
        v.loc[dates[26],'AMD']=np.nan
        self.assertFalse(factor_frame(f,v).feature_ready.iloc[7])

    def test_paired_calibration_uses_only_past_mature_forecasts(self):
        dates = pd.bdate_range('2025-01-01', periods=65)
        history=[{'at':d.tz_localize('UTC'),'date':d,'raw':{'C':i/1000,'D':i/500}}
                 for i,d in enumerate(dates)]
        p=pd.DataFrame({'C':-np.arange(65)/1000,'D':-np.arange(65)/500},index=[e['at'] for e in history])
        cal=paired_calibration(history,p,list(dates[:60])); self.assertTrue(cal['ready'])
        self.assertLess(cal['maps']['D']['coefficient'],0.)
        p.iloc[60:]=1e9; history[60]['raw']['D']=1e9
        self.assertEqual(cal,paired_calibration(history,p,list(dates[:60])))
        e=decision({'C':.03,'D':.1},cal,'direction')
        self.assertEqual(e['action'],'SOXS')
        cal['ready']=False
        for method in ['paired','raw','direction']: self.assertEqual(decision({'C':1.,'D':1.},cal,method)['action'],'CASH')

    def test_fit_scaler_uses_only_members_and_weights_normalized(self):
        dates=pd.bdate_range('2025-01-01',periods=6)
        p=pd.DataFrame({'date':dates,'feature_ready':True,'C':-.01,'D':[.04,.05,-.10,.02,.03,9.]},index=dates)
        for i,c in enumerate(BASE_COLUMNS+REGIME_COLUMNS):p[c]=np.arange(6)+i
        m,info=fit(p,list(dates[:5]),'regret_regime',minimum=5)
        self.assertAlmostEqual(sum(info['classification_weights']),5.)
        self.assertAlmostEqual(sum(info['parent_weights']),5.)
        self.assertLess(m['scaler'].mean_[0],3.)
        p.iloc[-1,p.columns.get_loc(BASE_COLUMNS[0])]=1e12
        n,_=fit(p,list(dates[:5]),'regret_regime',minimum=5)
        np.testing.assert_array_equal(m['heads']['q'].coef_,n['heads']['q'].coef_)
        p['D']=0.
        z,_=fit(p,list(dates[:5]),'regret_regime',minimum=5)
        self.assertEqual(z['q'],.5); self.assertNotIn('q',z['heads'])


if __name__=='__main__':unittest.main()
