import unittest
import numpy as np
import pandas as pd
from sequential import META, core, meta_state, neighbors, decision, realized_suffix, append_day


class SequentialTest(unittest.TestCase):
    def setUp(self):
        self.protocol={'window_dates':126,'minimum_dates':80,'neighbors':40}
        self.at=pd.Timestamp('2026-05-01 14:05:30Z')
        idx=pd.bdate_range('2025-09-01 14:05:30',periods=140,tz='UTC')
        self.pool=pd.DataFrame({k:np.zeros(len(idx)) for k in META},index=idx)
        self.pool['buy_return']=.01;self.pool['wait_return']=.005
        self.pool['release']=idx+pd.Timedelta(hours=6)
        self.query={**{k:0. for k in META},'symbol':'SOXL'}

    def test_current_future_and_late_releases_do_not_enter_history(self):
        chosen,audit=neighbors(self.pool,self.query,self.at.normalize(),self.protocol)
        self.assertEqual(len(audit['eligible']),126);self.assertEqual(len(chosen),40)
        altered=self.pool.copy();today=self.at-pd.Timedelta(minutes=20);future=self.at+pd.Timedelta(days=3)
        for at in [today,future]:
            altered.loc[at]=self.pool.iloc[-1]
            altered.loc[at,'release']=self.pool.release.iloc[0]
            altered.loc[at,'buy_return']=1000.
        later,other=neighbors(altered,self.query,self.at.normalize(),self.protocol)
        pd.testing.assert_frame_equal(chosen,later,check_freq=False);self.assertEqual(audit,other)
        altered.loc[self.pool.index[-1],'release']=self.at+pd.Timedelta(days=2)
        _,other=neighbors(altered,self.query,self.at.normalize(),self.protocol)
        self.assertNotIn(self.pool.index[-1],other['eligible'])

    def test_wait_competes_with_buy_and_no_wait_is_a_real_control(self):
        self.pool['wait_return']=.02
        wait,_=decision(self.query,self.pool,self.at,True,{'wait':True,'penalty':1},self.protocol)
        buy,_=decision(self.query,self.pool,self.at,True,{'wait':False,'penalty':1},self.protocol)
        self.assertEqual(wait['action'],'WAIT');self.assertEqual(buy['action'],'SOXL')
        last,_=decision(self.query,self.pool,self.at,False,{'wait':True,'penalty':1},self.protocol)
        self.assertEqual(last['action'],'SOXL');self.assertEqual(last['wait_value'],0.)

    def test_uncertainty_can_abstain_when_positive_mean_is_small(self):
        # Tied distances choose earliest eligible timestamps, predictably.
        self.pool['buy_return']=-.016
        self.pool.loc[self.pool.index[14:24],'buy_return']=.049
        risk,_=decision(self.query,self.pool,self.at,False,{'wait':False,'penalty':1},self.protocol)
        mean,_=decision(self.query,self.pool,self.at,False,{'wait':False,'penalty':0},self.protocol)
        self.assertGreater(mean['buy_mean'],0.)
        self.assertEqual(mean['action'],'SOXL');self.assertEqual(risk['action'],'CASH')

    def test_unknown_observations_are_not_zero(self):
        bad=self.pool.copy();bad.loc[bad.index[-1],'buy_return']=np.nan
        with self.assertRaises(ValueError):
            neighbors(bad,self.query,self.at.normalize(),self.protocol)
        self.pool['wait_return']=np.nan
        event,_=decision(self.query,self.pool,self.at,True,{'wait':True,'penalty':1},self.protocol)
        self.assertEqual(event['reason'],'unknown_continuation');self.assertFalse(event['ready'])

    def test_warmup_abstains_and_ties_do_not_buy(self):
        event,_=decision(self.query,self.pool.tail(79),self.at,True,{'wait':True,'penalty':1},self.protocol)
        self.assertEqual(event['action'],'WAIT');self.assertFalse(event['ready'])
        self.pool['buy_return']=0.;self.pool['wait_return']=0.
        event,_=decision(self.query,self.pool,self.at,False,{'wait':True,'penalty':1},self.protocol)
        self.assertEqual(event['action'],'CASH')

    def test_suffix_uses_first_saved_buy_not_best_future_trade(self):
        a=self.at;b=a+pd.Timedelta(minutes=30);c=a+pd.Timedelta(hours=3)
        tables={s:pd.DataFrame({'net_return':[-.016,.049]},index=[b,c]) for s in core.ETFS}
        events=[{'at':b,'action':'SOXL'},{'at':c,'action':'SOXS'}]
        self.assertAlmostEqual(realized_suffix(events,tables,a,[a,b,c]),np.log1p(-.016))
        tables['SOXS'].loc[c,'net_return']=1000.
        self.assertAlmostEqual(realized_suffix(events,tables,a,[a,b,c]),np.log1p(-.016))
        self.assertTrue(np.isnan(realized_suffix(events[:1],tables,a,[a,b,c])))
        events=[{'at':b,'action':'WAIT'},{'at':c,'action':'CASH'}]
        self.assertEqual(realized_suffix(events,tables,a,[a,b,c]),0.)

    def test_selected_state_calibrates_even_negative_ranked_scores(self):
        row=pd.Series({'regular_context_ready':1.,'SOXX_vol20':.02,'x_session_return':.01,
                       'x_vwap_distance':.005,'q_recent15':-.003})
        ensemble=pd.Series({'ready':True,'SOXL_adjusted':-.02,'SOXS_adjusted':-.01})
        a=pd.Series({'SOXL_adjusted':-.01,'SOXS_adjusted':0.})
        b=pd.Series({'SOXL_adjusted':-.03,'SOXS_adjusted':-.02})
        result=meta_state(row,ensemble,a,b)
        self.assertEqual(result['symbol'],'SOXS');self.assertEqual(result['direction'],-1.)
        self.assertEqual(result['signed_sector_session_return'],-.01)
        ensemble['ready']=False;self.assertIsNone(meta_state(row,ensemble,a,b))

    def test_appended_end_of_day_labels_still_cannot_train_same_day(self):
        b=self.at+pd.Timedelta(minutes=30)
        events=[{'at':self.at,'action':'WAIT'},{'at':b,'action':'SOXL'}]
        tables={s:pd.DataFrame({'net_return':[.01,.02]},index=[self.at,b]) for s in core.ETFS}
        clocks=[t.tz_convert(core.ET).strftime('%H:%M:%S') for t in [self.at,b]]
        pools={c:pd.DataFrame(index=pd.DatetimeIndex([],tz='UTC')) for c in clocks}
        append_day(pools,{self.at:self.query,b:self.query},events,tables,self.at+pd.Timedelta(hours=6),[self.at,b])
        _,audit=neighbors(pools[clocks[0]],self.query,self.at.normalize(),{'window_dates':126,'minimum_dates':1,'neighbors':1})
        self.assertEqual(audit['eligible'],[])


if __name__=='__main__':
    unittest.main()
