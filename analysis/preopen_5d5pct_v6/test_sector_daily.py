import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
from sector_daily import peer_daily
from common import DATES

class PriorPeerContext(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([dict(symbol=s,day=d,d_return_5=v,d_return_20=v*2) for d in DATES[20:40] for s,v in [('A',.1),('B',.2),('C',-.1)]])
    def compute(self,f):
        with patch('sector_daily.GROUP_OF',{'A':'g','B':'g','C':'g'}),patch('sector_daily.GROUPS',{'g':['A','B','C']}):return peer_daily(f)
    def test_target_excluded_from_peer_statistics(self):
        f=self.fixture();r=self.compute(f);a=r[r.symbol=='A'];self.assertTrue(np.allclose(a.gd_mean_5,.05));self.assertTrue(np.allclose(a.gd_breadth_5,.5))
    def test_future_rows_do_not_change_prior_context(self):
        f=self.fixture();before=self.compute(f);f.loc[f.day>DATES[30],'d_return_5']=99.;after=self.compute(f)
        pd.testing.assert_frame_equal(before[before.day<=DATES[30]],after[after.day<=DATES[30]])
if __name__=='__main__':unittest.main()
