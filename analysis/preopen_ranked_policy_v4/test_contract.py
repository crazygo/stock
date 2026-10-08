import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import data,policy,store

class Contract(unittest.TestCase):
 def raw(self):
  starts=pd.date_range('2026-08-03 09:30',periods=78,freq='5min');f=pd.DataFrame(dict(start=starts,end=starts+pd.Timedelta(minutes=5),available=starts+pd.Timedelta(minutes=5,seconds=1),open=100.,high=101.,low=99.,close=100.,volume=100000.,day='2026-08-03',minute=np.arange(570,960,5)))
  ctx=pd.DataFrame([dict(day='2026-08-03',prior_close=100.,prior_volume=7800000.,**{k:.01 for k in data.STATIC})]);return f,ctx
 def test_future_price_volume_do_not_change_features(self):
  f,c=self.raw();original=data.intraday(f,c,'AAOI');mut=f.copy();mut.loc[mut.minute>=600,['open','high','low','close','volume']]=[500.,600.,400.,500.,9e9];changed=data.intraday(mut,c,'AAOI');cols=[k for k in original if k not in data.LABELS]
  pd.testing.assert_frame_equal(original[original.minute<=600][cols].reset_index(drop=True),changed[changed.minute<=600][cols].reset_index(drop=True))
 def test_future_missing_keeps_present_prediction(self):
  f,c=self.raw();full=data.intraday(f,c,'AAOI');partial=data.intraday(f[f.minute<600],c,'AAOI');cols=[k for k in full if k not in data.LABELS]
  pd.testing.assert_frame_equal(full[full.minute<=600][cols].reset_index(drop=True),partial[cols].reset_index(drop=True));self.assertTrue(partial.y.isna().all())
 def test_late_bar_excluded(self):
  f,c=self.raw();f.loc[0,'available']=pd.Timestamp('2026-08-03 09:36');p=data.intraday(f,c,'AAOI');self.assertNotIn(575,p.minute.values)
 def test_next_open_is_after_decision(self):
  f,c=self.raw();p=data.intraday(f,c,'AAOI');r=p.iloc[0];self.assertEqual(r.minute,575);self.assertEqual(r.entry_minute,580);self.assertAlmostEqual(r.entry,100.1)
 def predictions(self):
  rows=[]
  for t in [575,580,585,590]:
   for s,score in [('AAOI',.9),('ALAB',.8)]:rows.append(dict(symbol=s,day='2026-08-03',minute=t,entry_minute=t+5,reference=100.,score=score,baseline=.5,ex_action=0,last_volume=1e6,feature_available='2026-08-03 09:35:01',entry=100.,exit=103.,exit_minute=590,y=1,net=.03,mae=-.01))
  return pd.DataFrame(rows)
 def test_only_first_qualified_and_one_position(self):
  p=policy.replay(self.predictions(),.7);self.assertEqual(p['trades'][0]['symbol'],'AAOI');self.assertEqual([d['reason'] for d in p['decisions'][1:3]],['position_open','position_open']);self.assertEqual(len(p['trades']),2);self.assertEqual(p['trades'][1]['symbol'],'ALAB')
 def test_first_liquidity_failure_does_not_buy_second(self):
  f=self.predictions();f.loc[f.symbol=='AAOI','last_volume']=1.;p=policy.replay(f,.7);self.assertEqual(len(p['trades']),0);self.assertTrue(all(d['reason']=='top1_liquidity_rejected' for d in p['decisions']))
 def test_no_threshold_no_operation(self):self.assertEqual(policy.replay(self.predictions(),None)['metrics']['n'],0)
 def test_unknown_outcome_remains_in_denominator(self):
  f=self.predictions();f.loc[f.symbol=='AAOI',['entry','exit','y','net','mae']]=np.nan;p=policy.replay(f,.7);self.assertEqual(p['metrics']['n'],1);self.assertEqual(p['metrics']['unknown'],1);self.assertEqual(p['metrics']['precision'],0.)
 def test_settlement_cash_not_recycled_same_day(self):
  f=self.predictions();p=policy.replay(f,.7);self.assertAlmostEqual(p['decisions'][-1]['cash_before'],100000.-p['trades'][0]['budget']);self.assertGreater(p['equity'][-1]['unsettled'],0)
 def test_database_constraints(self):
  with tempfile.TemporaryDirectory() as t:
   c=store.connect(Path(t)/'test.sqlite');values=('2026-08-03','same_decision','regular','no_action','test','{}')
   with c:
    c.execute('INSERT OR IGNORE INTO observations(observed_at,decision_key,market_state,action,reason,payload) VALUES(?,?,?,?,?,?)',values);c.execute('INSERT OR IGNORE INTO observations(observed_at,decision_key,market_state,action,reason,payload) VALUES(?,?,?,?,?,?)',values)
   self.assertEqual(c.execute('SELECT COUNT(*) FROM observations').fetchone()[0],1);c.close()

if __name__=='__main__':unittest.main()
