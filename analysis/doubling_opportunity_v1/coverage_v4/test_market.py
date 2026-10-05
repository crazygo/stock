import unittest
import sys
from pathlib import Path
import numpy as np,pandas as pd
from market import normalized,price_features,labels,PRICE_FEATURES
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from scripts.model_history_calendar import calendar

class SplitCausalTests(unittest.TestCase):
    def sessions(self,start,end):
        return pd.DatetimeIndex([r['session_date'] for r in calendar(start,end)['sessions']])
    def frame(self,index,close):
        return pd.DataFrame({'code':'US.TEST','time_key':index.strftime('%Y-%m-%d 00:00:00'),
            'open':close,'high':np.asarray(close)*1.01,'low':np.asarray(close)*.99,'close':close,'volume':1_000_000.})

    def test_future_split_does_not_change_any_past_feature_or_absolute_price(self):
        idx=self.sessions('2023-01-02','2024-03-01');raw=self.frame(idx,np.linspace(10.,30.,len(idx)))
        spy={n:pd.Series(0.,index=idx) for n in (20,60)};expected=pd.Series(1.,index=idx).rolling('183D').count()
        empty=pd.DataFrame(columns=['ex_div_date','split_ratio']);events=pd.DataFrame([{'ex_div_date':'2024-02-01','split_ratio':.1}])
        a=price_features(raw,idx,empty,'2024-03-01',spy,expected)[0];b=price_features(raw,idx,events,'2024-03-01',spy,expected)[0]
        before=idx<'2024-02-01'
        pd.testing.assert_frame_equal(a.loc[before,PRICE_FEATURES],b.loc[before,PRICE_FEATURES],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(b.price,np.log(raw.close));np.testing.assert_allclose(a.dollar_volume_actual,b.dollar_volume_actual,equal_nan=True)

    def test_reverse_split_is_not_a_doubling_and_real_economic_doubling_is(self):
        idx=self.sessions('2024-01-02','2024-04-30');prices=np.where(idx<pd.Timestamp('2024-01-16'),10.,50.)
        raw=self.frame(idx,prices);events=pd.DataFrame([{'ex_div_date':'2024-01-16','split_ratio':5.}])
        _,adj,_,unknown,_,_,_=normalized(raw,idx,events,'2024-04-30')
        q=labels(adj,idx,unknown,'2024-04-30',60)
        self.assertEqual(q.loc['2024-01-02','y60'],0)
        adj.loc['2024-01-17','high']=101.
        q=labels(adj,idx,unknown,'2024-04-30',60)
        self.assertEqual(q.loc['2024-01-02','y60'],1);self.assertEqual(q.loc['2024-01-02','first_touch60'],pd.Timestamp('2024-01-17'))

    def test_missing_prefix_and_unsupported_actions_remain_unknown(self):
        idx=self.sessions('2024-01-02','2024-04-30');raw=self.frame(idx,np.full(len(idx),10.))
        events=pd.DataFrame([{'ex_div_date':'2024-01-16','spin_off_ratio':.2}])
        _,adj,_,unknown,feature_unknown,_,_=normalized(raw,idx,events,'2024-04-30')
        q=labels(adj,idx,unknown,'2024-04-30',60)
        self.assertTrue(pd.isna(q.loc['2024-01-02','y60']));self.assertEqual(q.loc['2024-01-02','label_status60'],'unsupported_corporate_action')
        self.assertFalse(feature_unknown[idx.get_loc('2024-01-12')]);self.assertTrue(feature_unknown[idx.get_loc('2024-01-16')])
        adj.loc['2024-01-17','close']=np.nan;adj.loc['2024-01-17','high']=np.nan
        q=labels(adj,idx,np.zeros(len(idx),bool),'2024-04-30',60)
        self.assertTrue(pd.isna(q.loc['2024-01-02','y60']))

    def test_weekend_maturity_uses_final_official_session_and_no_early_touch(self):
        idx=self.sessions('2024-01-02','2024-03-29');raw=self.frame(idx,np.full(len(idx),10.))
        raw.loc[1,'high']=25.
        _,adj,_,unknown,_,_,_=normalized(raw,idx,pd.DataFrame(),'2024-03-29')
        immature=labels(adj,idx,unknown,'2024-01-05',5)
        self.assertEqual(immature.loc['2024-01-02','y5'],1) # expiry Sunday, last session Friday already complete
        q=labels(adj,idx,unknown,'2024-01-04',5)
        self.assertTrue(pd.isna(q.loc['2024-01-02','y5'])) # observed touch does not waive full maturity

    def test_no_fake_holiday_and_invalid_ohlc_cannot_mature_success(self):
        idx=self.sessions('2024-01-02','2024-04-30');self.assertNotIn(pd.Timestamp('2024-01-15'),idx)
        raw=self.frame(idx,np.full(len(idx),10.));raw.loc[1,'high']=25.
        # A later contradictory bar invalidates the whole future prefix even after a real touch.
        raw.loc[3,'low']=11.
        _,adj,_,unknown,_,_,_=normalized(raw,idx,pd.DataFrame(),'2024-04-30')
        q=labels(adj,idx,unknown,'2024-04-30',60)
        self.assertTrue(pd.isna(q.loc['2024-01-02','y60']))
        self.assertEqual(q.loc['2024-01-02','label_status60'],'missing_or_invalid_future_session')

if __name__=='__main__':unittest.main()
