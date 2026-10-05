"""Causal regression checks on calendar gaps, missingness and publication lag."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from features import measurements,lagged,real_turnover

class CausalDailyTests(unittest.TestCase):
    def test_lag_and_future_perturbation(self):
        grid=['2026-05-21','2026-05-22','2026-05-26','2026-05-27','2026-05-28','2026-05-29']
        f=pd.DataFrame({'source_day':grid,'iv':[10.,20.,30.,40.,50.,60.],'hv':[5.]*6})
        x,obs=measurements('volatility',f,grid=grid);a=lagged('volatility',x,obs,2,grid)
        self.assertEqual(a.iloc[2].f2_v_source_day,'2026-05-21');self.assertEqual(a.iloc[2].f2_v_iv,10.)
        self.assertEqual(a.iloc[2].f2_v_available_day,'2026-05-26')
        f.loc[f.source_day>='2026-05-26','iv']=100000
        x,obs=measurements('volatility',f,grid=grid);b=lagged('volatility',x,obs,2,grid)
        pd.testing.assert_frame_equal(a.iloc[:4],b.iloc[:4])
    def test_missing_provider_day_does_not_become_observation(self):
        grid=[f'2026-05-{d:02}' for d in [4,5,6,7,8,11]]
        f=pd.DataFrame({'source_day':[grid[0],grid[2]],'iv':[10.,30.],'hv':[5.,6.]})
        x,obs=measurements('volatility',f,grid=grid);a=lagged('volatility',x,obs,2,grid)
        self.assertTrue(np.isnan(x.loc[grid[1],'iv']));self.assertEqual(a.iloc[3].f2_v_source_day,grid[0]);self.assertEqual(a.iloc[3].f2_v_stale_sessions,1.)
    def test_turnover_same_source_day_and_missing(self):
        grid=['2026-05-04','2026-05-05','2026-05-06']
        f=pd.DataFrame({'source_day':grid,**{n:[10.,20.,30.] for n in ['in_flow','super_in_flow','big_in_flow','mid_in_flow','sml_in_flow','main_in_flow']}})
        x,_=measurements('capital',f,pd.Series([100.,0.],index=grid[:2]),grid)
        self.assertEqual(x.iloc[0].in_flow,.1);self.assertTrue(x.iloc[1:].in_flow.isna().all())
    def test_oi_not_backdated_and_five_session_delay(self):
        grid=[f'2026-05-{d:02}' for d in [4,5,6,7,8,11]]
        f=pd.DataFrame({'source_day':grid,**{n:[100.]*6 for n in ['call_volume','put_volume','call_open_interest','put_open_interest','put_call_volume_ratio','put_call_open_interest_ratio','option_volume','option_open_interest']}})
        x,obs=measurements('options',f,grid=grid);a=lagged('options',x,obs,5,grid)
        self.assertTrue(a.iloc[:5].f5_o_source_day.isna().all());self.assertEqual(a.iloc[5].f5_o_source_day,grid[0])
    def test_real_turnover_requires_complete_matching_half_day(self):
        day='2025-11-28';minutes=np.arange(570,780,5)
        raw=pd.DataFrame({'start':[pd.Timestamp(day)+pd.Timedelta(minutes=int(m)) for m in minutes],'day':day,'minute':minutes,'open':5.,'high':6.,'low':4.,'close':5.,'volume':10.})
        archive=raw.drop(columns=['day','minute']).copy();archive['start_at_et']=archive.start.dt.tz_localize('America/New_York').astype(str);archive['price_basis']='NONE';archive['turnover']=50.
        with tempfile.TemporaryDirectory() as directory,patch('features.ROOT',Path(directory)),patch('features.OUT',Path(directory)/'v6'):
            root=Path(directory);a=root/'market_data/us_5m/X/2026.parquet';b=root/'market_data/model_training_history_v1/parts/X/2025-11/bars.parquet';a.parent.mkdir(parents=True);b.parent.mkdir(parents=True)
            wrong=archive.copy();wrong['open']=7.;wrong.to_parquet(a,index=False);archive.to_parquet(b,index=False)
            t,_,_=real_turnover('X',raw);self.assertEqual(t[day],2100.)
            archive.iloc[:-1].to_parquet(b,index=False);t,_,_=real_turnover('X',raw);self.assertTrue(t.empty)

if __name__=='__main__':unittest.main()
