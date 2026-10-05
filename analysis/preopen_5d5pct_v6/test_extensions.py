import unittest
import numpy as np
import pandas as pd
from research_extensions import peers_exact

class ExactPeerClock(unittest.TestCase):
 def test_missing_clock_does_not_become_six_observations(self):
  rows=[]
  for minute in [300,310,315,320,325,330,335]:
   for symbol,value in [('AMD',.03),('MRVL',.02)]:rows.append(dict(symbol=symbol,day='2026-09-01',minute=minute,i15_ret=value,i30_ret=value,rth_ret=value,pre_gap=value))
  result=peers_exact(pd.DataFrame(rows))
  self.assertTrue(result.loc[result.minute==335,'b_leader_persist_30'].isna().all())
  self.assertTrue((result.loc[result.minute==330,'b_leader_persist_30']==1).all())
 def test_future_peer_rows_do_not_change_earlier_prefix(self):
  rows=[]
  for minute in range(300,365,5):
   for symbol,value in [('AMD',.03),('MRVL',.02)]:rows.append(dict(symbol=symbol,day='2026-09-01',minute=minute,i15_ret=value,i30_ret=value,rth_ret=value,pre_gap=value))
  original=pd.DataFrame(rows);changed=original.copy();changed.loc[changed.minute>330,['i15_ret','i30_ret','rth_ret','pre_gap']]=100
  cols=['b_positive_change_30','b_leader_change_30','b_leader_persist_30']
  a=peers_exact(original);b=peers_exact(changed)
  np.testing.assert_allclose(a.loc[a.minute<=330,cols],b.loc[b.minute<=330,cols],equal_nan=True)
if __name__=='__main__':unittest.main()
