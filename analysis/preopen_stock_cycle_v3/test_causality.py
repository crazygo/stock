"""Meaningful future-mutation and missingness checks for minute-window inputs."""
import json,unittest
import numpy as np
import pandas as pd
import build as b
class CausalTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.f,cls.sources,cls.audit=b.legacy.load('AAOI');cls.original=b.legacy.load
  cls.calendar={r['session_date']:r for r in json.loads((b.legacy.HISTORY/'calendar.json').read_text())['sessions']}
  cls.date='2026-08-05';cls.base=cls.run_frame(cls.f)
 @classmethod
 def run_frame(cls,f):
  b.legacy.load=lambda s:(f.copy(),cls.sources,cls.audit)
  try:
   rows,contexts,*_=b.make_symbol('AAOI',cls.calendar,write_raw=False)
   return {r['day']:r for r in rows},{r['day']:r for r in contexts}
  finally:b.legacy.load=cls.original
 def compare(self,a,z):
  self.assertEqual(set(a),set(z))
  for k in a:
   if isinstance(a[k],float):self.assertTrue(np.isclose(a[k],z[k],equal_nan=True),k)
   else:self.assertEqual(a[k],z[k],k)
 def test_future_price_volume_cannot_change_prefix_features(self):
  f=self.f.copy();m=(f.day==self.date)&(f.minute>=570);f.loc[m,'high']*=2;f.loc[m,'volume']*=10
  altered=self.run_frame(f);self.compare(self.base[1][self.date],altered[1][self.date]);self.assertNotEqual(self.base[0][self.date]['mfe'],altered[0][self.date]['mfe'])
 def test_future_missing_label_cannot_change_current_context(self):
  f=self.f[~((self.f.day==self.date)&(self.f.minute==600))];altered=self.run_frame(f)
  self.compare(self.base[1][self.date],altered[1][self.date]);self.assertNotIn(self.date,altered[0])
 def test_available_after_decision_is_excluded(self):
  f=self.f.copy();m=(f.day==self.date)&(f.minute==560);f.loc[m,'available']=pd.Timestamp(self.date+' 09:26')
  altered=self.run_frame(f);self.assertNotIn(self.date,altered[1])
 def test_quiet_night_is_not_missing_night(self):
  f=self.f.copy();d=pd.Timestamp(self.date);m=(f.start>=d-pd.Timedelta(hours=4))&(f.start<d+pd.Timedelta(hours=4));f.loc[m,'volume']=0
  altered=self.run_frame(f);self.assertIn(self.date,altered[0]);self.assertEqual(altered[1][self.date]['night_quiet'],1);self.assertEqual(altered[1][self.date]['night_missing'],0)
 def test_missing_night_keeps_pre_and_missing_flag(self):
  d=pd.Timestamp(self.date);f=self.f[~((self.f.start>=d-pd.Timedelta(hours=4))&(self.f.start<d+pd.Timedelta(hours=4)))];altered=self.run_frame(f)
  self.assertIn(self.date,altered[0]);self.assertEqual(altered[1][self.date]['night_missing'],1)
 def test_peer_context_is_label_independent(self):
  row=self.base[0][self.date].copy();context=self.base[1][self.date].copy();context['symbol']='COHR'
  panel=b.enrich_peer(pd.DataFrame([row]),pd.DataFrame([row,context]));self.assertEqual(panel.peer_count.iloc[0],1);self.assertTrue(np.isfinite(panel.peer_pre_gap.iloc[0]))
 def test_one_stock_day_is_one_sample(self):
  rows=list(self.base[0].values());self.assertEqual(len(rows),len(set(r['day'] for r in rows)))
 def test_same_day_macro_close_cannot_change_inputs(self):
  import enrich_macro as macro
  calendar=list(self.calendar.values());base=macro.macro_frame(calendar);original=pd.read_csv
  def changed(path,*a,**k):
   f=original(path,*a,**k);mask=f.DATE.eq('08/05/2026')
   for col in ['CLOSE','VVIX']:
    if col in f:f.loc[mask,col]*=3
   return f
  pd.read_csv=changed
  try:altered=macro.macro_frame(calendar)
  finally:pd.read_csv=original
  pd.testing.assert_series_equal(base.loc[self.date],altered.loc[self.date]);self.assertNotEqual(base.loc['2026-08-06','env_vix'],altered.loc['2026-08-06','env_vix'])
if __name__=='__main__':unittest.main()
