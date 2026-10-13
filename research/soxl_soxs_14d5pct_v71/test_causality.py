"""Prevent future/missing observations and duplicate pair rows changing eligibility."""
import copy
import unittest
from transfer_learning import *

INPUTS=ROOT/'.cache/soxl_soxs_14d5pct_v71/inputs/development'


class CausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.market=load_market()
        cls.prefixes={s:pd.read_parquet(INPUTS/(s+'_prefix.parquet')) for s in sorted({s for p in PAIRS.values() for s in p})}
        cls.at=pd.Timestamp('2026-01-02 09:25:30',tz=core.ET).tz_convert('UTC')

    def test_future_prices_and_prefixes_do_not_change_today(self):
        want,_=feature_row(self.market,self.prefixes,'semiconductor',self.at)
        market=copy.copy(self.market);market.daily={s:f.copy() for s,f in self.market.daily.items()}
        prefixes={s:f.copy() for s,f in self.prefixes.items()}
        for f in market.daily.values():f.loc[f.index>=pd.Timestamp('2026-01-02'),'close']*=100
        for f in prefixes.values():f.loc[f.index>self.at,['amount','quote','quote15']]*=100
        got,_=feature_row(market,prefixes,'semiconductor',self.at)
        np.testing.assert_array_equal(got,want)

    def test_missing_official_day_is_not_skipped(self):
        market=copy.copy(self.market);market.daily=dict(self.market.daily)
        market.daily['SOXX']=market.daily['SOXX'].drop(pd.Timestamp('2025-12-30'))
        got,_=feature_row(market,self.prefixes,'semiconductor',self.at);self.assertIsNone(got)

    def test_missing_historical_activity_is_not_skipped(self):
        prefixes=dict(self.prefixes);prefixes['SOXL']=prefixes['SOXL'].copy()
        old=pd.Timestamp('2025-12-30 09:25:30',tz=core.ET).tz_convert('UTC')
        prefixes['SOXL'].loc[old,'prefix_ready']=False
        got,_=feature_row(self.market,prefixes,'semiconductor',self.at);self.assertIsNone(got)

    def test_missing_prefix_bar_invalidates_entire_current_prefix(self):
        market=copy.copy(self.market);market.bars=dict(self.market.bars)
        market.bars['SOXX']=market.bars['SOXX'].drop(self.at-pd.Timedelta(seconds=30,minutes=10))
        table=prefix_table(market,pd.DatetimeIndex([self.at]));self.assertFalse(table['SOXX'].iloc[0].prefix_ready)

    def test_unresolved_training_date_cannot_be_jumped(self):
        rows=[{'date':pd.Timestamp(d),'members':[],'feature_ready':True,'release':r} for d,r in [
             ('2025-12-01',pd.Timestamp('2025-12-15',tz='UTC')),('2025-12-02',None),
             ('2025-12-03',pd.Timestamp('2025-12-17',tz='UTC'))]]
        selected,feedback=mature_prefix(rows,self.at)
        self.assertEqual(len(selected),1);self.assertTrue(feedback['expired_unresolved_prefix'])


if __name__=='__main__':unittest.main()
