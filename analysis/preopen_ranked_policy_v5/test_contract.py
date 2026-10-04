import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import data,policy,store

class Contract(unittest.TestCase):
 def raw(self):
  starts=pd.date_range('2026-08-03 04:00',periods=144,freq='5min');f=pd.DataFrame(dict(start=starts,end=starts+pd.Timedelta(minutes=5),available=starts+pd.Timedelta(minutes=5,seconds=1),open=100.,high=101.,low=99.,close=100.,volume=100000.,day='2026-08-03',minute=np.arange(240,960,5)))
  ctx=pd.DataFrame([dict(day='2026-08-03',prior_close=100.,prior_volume=7800000.,**{k:.01 for k in data.STATIC})]);return f,ctx
 def test_future_price_volume_do_not_change_features(self):
  f,c=self.raw();original=data.intraday(f,c,'AAOI');mut=f.copy();mut.loc[mut.minute>=600,['open','high','low','close','volume']]=[500.,600.,400.,500.,9e9];changed=data.intraday(mut,c,'AAOI');cols=[k for k in original if k not in data.LABELS]
  pd.testing.assert_frame_equal(original[original.minute<=600][cols].reset_index(drop=True),changed[changed.minute<=600][cols].reset_index(drop=True))
 def test_future_missing_keeps_present_prediction(self):
  f,c=self.raw();full=data.intraday(f,c,'AAOI');partial=data.intraday(f[f.minute<600],c,'AAOI');cols=[k for k in full if k not in data.LABELS]
  pd.testing.assert_frame_equal(full[full.minute<=600][cols].reset_index(drop=True),partial[cols].reset_index(drop=True));self.assertTrue(partial.y.isna().all())
 def test_late_bar_excluded(self):
  f,c=self.raw();f.loc[0,'available']=pd.Timestamp('2026-08-03 04:06');p=data.intraday(f,c,'AAOI');self.assertNotIn(245,p.minute.values)
 def test_next_open_is_after_decision(self):
  f,c=self.raw();p=data.intraday(f,c,'AAOI');r=p.iloc[0];self.assertEqual(r.minute,245);self.assertEqual(r.entry_minute,250);self.assertAlmostEqual(r.entry,100.1)
 def test_future_pre_bars_do_not_leak_into_early_signal(self):
  f,c=self.raw();original=data.intraday(f,c,'AAOI');mut=f.copy();mut.loc[(mut.minute>=480)&(mut.minute<570),['open','high','low','close','volume']]=[500.,600.,400.,500.,9e9];changed=data.intraday(mut,c,'AAOI');cols=[k for k in original if k not in data.LABELS]
  pd.testing.assert_frame_equal(original[original.minute<=480][cols].reset_index(drop=True),changed[changed.minute<=480][cols].reset_index(drop=True))
 def test_pre_entry_changes_price_reference(self):
  f,c=self.raw();f.loc[f.minute<570,['open','high','low','close']]=[95.,96.,94.,95.];p=data.intraday(f,c,'AAOI');r=p.iloc[0];self.assertAlmostEqual(r.entry,95.095);self.assertEqual(r.y,1);self.assertEqual(r.exit_minute,575)
 def test_pre_touch_does_not_replace_rth_label(self):
  f,c=self.raw();f.loc[(f.minute>=250)&(f.minute<570),'high']=110.;r=data.intraday(f,c,'AAOI').iloc[0];self.assertEqual(r.y,0)
 def test_delayed_night_not_available_in_early_pre(self):
  f,c=self.raw();c['night_available']='2026-08-03 09:20';p=data.intraday(f,c,'AAOI');self.assertTrue(np.isnan(p.iloc[0].night_ret));self.assertEqual(p.iloc[0].night_missing,1.)
 def test_phase_predict_handles_single_phase(self):
  from phase import predict
  class Model:
   def predict_proba(self,x):return np.tile([.2,.8],(len(x),1))
  class Cal:
   def predict_proba(self,x):
    if not len(x):raise ValueError('empty phase')
    return np.tile([.2,.8],(len(x),1))
  a=dict(model=Model(),features=['value'],phase_calibrators={'pre':Cal(),'regular':Cal()});self.assertEqual(list(predict(a,pd.DataFrame({'minute':[245,250],'value':[1.,1.]}))),[.8,.8]);self.assertEqual(list(predict(a,pd.DataFrame({'minute':[575],'value':[1.]}))),[.8])
 def predictions(self):
  rows=[]
  for t in [575,580,585,590]:
   for s,score in [('AAOI',.9),('ALAB',.8)]:rows.append(dict(symbol=s,day='2026-08-03',minute=t,entry_minute=t+5,reference=100.,score=score,baseline=.5,ex_action=0,last_volume=1e6,feature_available='2026-08-03 09:35:01',entry=100.,exit=103.,exit_minute=590,y=1,net=.03,mae=-.01))
  return pd.DataFrame(rows)
 def test_only_first_qualified_and_one_position(self):
  p=policy.replay(self.predictions(),.7);self.assertEqual(p['trades'][0]['symbol'],'AAOI');self.assertEqual([d['reason'] for d in p['decisions'][1:3]],['position_open','position_open']);self.assertEqual(len(p['trades']),2);self.assertEqual(p['trades'][1]['symbol'],'ALAB')
 def test_rank_first_only_after_liquidity_qualification(self):
  f=self.predictions();f.loc[f.symbol=='AAOI','last_volume']=1.;p=policy.replay(f,.7);self.assertEqual(p['trades'][0]['symbol'],'ALAB');self.assertNotIn('AAOI',p['decisions'][0]['eligible_symbols']);self.assertEqual(p['decisions'][0]['top_symbol'],'ALAB')
 def test_no_threshold_no_operation(self):self.assertEqual(policy.replay(self.predictions(),None)['metrics']['n'],0)
 def test_capacity_sizing_preserves_top_rank_and_minimum(self):
  f=self.predictions();f.loc[f.symbol=='AAOI','last_volume']=1500.
  fixed=policy.replay(f,.7);small=policy.replay(f,.7,capacity_capped=True)
  self.assertEqual(fixed['trades'][0]['symbol'],'ALAB');self.assertEqual(small['trades'][0]['symbol'],'AAOI');self.assertEqual(small['trades'][0]['qty'],15)
  self.assertGreaterEqual(small['trades'][0]['budget'],1000.);self.assertLessEqual(small['trades'][0]['qty'],1500*.01)
  f.loc[f.symbol=='AAOI','last_volume']=900.;self.assertEqual(policy.replay(f,.7,capacity_capped=True)['trades'][0]['symbol'],'ALAB')
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
 def test_live_exit_missing_path_cannot_become_failure(self):
  from server import live_exit_event
  from common import ET
  f,_=self.raw();now=pd.Timestamp('2026-08-03 16:05',tz=ET).to_pydatetime()
  self.assertEqual(live_exit_event(f,'2026-08-03T04:10:00-04:00',103.,now)['hit'],0)
  gap=f[f.minute!=600];self.assertIsNone(live_exit_event(gap,'2026-08-03T04:10:00-04:00',103.,now))
  gap=gap.copy();gap.loc[gap.minute==610,'high']=104.;self.assertIsNone(live_exit_event(gap,'2026-08-03T04:10:00-04:00',103.,now))
  f.loc[f.minute==580,'high']=104.;self.assertEqual(live_exit_event(f[f.minute!=600],'2026-08-03T04:10:00-04:00',103.,now)['hit'],1)

if __name__=='__main__':unittest.main()
