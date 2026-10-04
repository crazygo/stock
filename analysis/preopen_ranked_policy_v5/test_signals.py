import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import signals
import signal_runtime
from store import connect
from test_maturity import raw,now,DAY

class Signals(unittest.TestCase):
 def frame(self):
  rows=[]
  for i in range(20):
   t=575+i*5
   for s,p in [(f'S{i}',.85 if i<10 else .4),(f'L{i}',.1)]:
    rows.append(dict(day='2026-08-03',minute=t,symbol=s,score=p,ex_action=0,feature_available=str(pd.Timestamp('2026-08-03')+pd.Timedelta(minutes=t,seconds=1)),baseline=.5,reference=100.,entry=100.1,exit=103.,exit_minute=960,y=int(i<8),net=.03,mae=-.01,label_end='2026-08-03 16:00',cash=0.,position='held',last_volume=0.))
  return pd.DataFrame(rows)
 def test_only_issued_valid_signals_in_precision(self):
  sim=signals.replay(self.frame(),{'pre':.8,'regular':.8});m=sim['metrics'];self.assertEqual((m['n'],m['tp'],m['fp'],m['abstentions']),(10,8,2,10));self.assertEqual(m['precision'],.8);self.assertEqual(m['coverage'],.5)
 def test_validity_independent_of_future_and_account(self):
  f=self.frame();a=signals.replay(f,{'pre':.8,'regular':.8});f[['y','entry','exit','net','mae']]=np.nan;f['cash']=1e9;f['position']=None;b=signals.replay(f,{'pre':.8,'regular':.8});self.assertEqual([(e['day'],e['symbol'],e['minute']) for e in a['events']],[(e['day'],e['symbol'],e['minute']) for e in b['events']]);self.assertEqual(b['metrics']['n'],10);self.assertEqual(b['metrics']['pending'],10);self.assertIsNone(b['metrics']['precision']);self.assertEqual(b['metrics']['conservative_precision'],0.)
 def test_late_top_rank_and_same_stock_day_dedup(self):
  f=self.frame().iloc[:4].copy();f['symbol']=['A','B','A','B'];f['score']=[.95,.85,.95,.85];f.loc[f.index[0],'feature_available']='2026-08-03 10:00';s=signals.replay(f,{'pre':.8,'regular':.8});self.assertEqual([e['symbol'] for e in s['events']],['B','A'])
 def test_disabled_threshold_is_abstention_not_accuracy(self):
  m=signals.replay(self.frame(),{'pre':None,'regular':None})['metrics'];self.assertEqual(m['n'],0);self.assertEqual(m['abstentions'],20);self.assertIsNone(m['precision'])
 def test_high_confidence_runner_up_is_not_emitted_sample(self):
  f=self.frame().iloc[:2].copy();f['score']=[.9,.85];m=signals.replay(f,{'pre':.8,'regular':.8})['metrics'];self.assertEqual(m['n'],1)
 def test_prospective_signal_maturity_without_fills(self):
  with tempfile.TemporaryDirectory() as td:
   db=Path(td)/'test.sqlite';con=connect(db);e=dict(day=DAY,symbol='A',minute=595,phase='regular',score=.8,threshold=.75,baseline=.5,reference=100.,feature_available=DAY+' 09:55:01',feature_row={})
   with con:
    signals.put_run(con,'live','route','prospective_signal',{'frozen':'one'});signals.put_event(con,'live',e);signals.put_event(con,'live',e);signals.put_tick(con,'live',dict(day=DAY,minute=595,symbol='A',reason='valid_signal',eligible=1))
   con.close();f=raw();f.loc[f.minute==650,'high']=105.
   with patch.object(signals,'connect',side_effect=lambda:connect(db)):
    self.assertEqual(signals.settle(lambda _:f[f.minute!=950],now('16:01')),0);self.assertEqual(signals.settle(lambda _:f,now('16:01')),1);self.assertEqual(signals.settle(lambda _:f,now('16:02')),0)
   con=connect(db);m=signals.summary(con,'live');self.assertEqual(m['n'],1);self.assertEqual(m['precision'],1.);self.assertEqual(con.execute('SELECT count(*) FROM live_fills').fetchone()[0],0);con.close()
 def test_live_validity_and_repeat_independent_of_cash(self):
  with tempfile.TemporaryDirectory() as td:
   db=Path(td)/'test.sqlite';dep=dict(admitted=True,route='route',frozen_at='frozen',protocol_sha256='hash',all_routes=[dict(route='route',thresholds={'pre':.8,'regular':.8})]);r=self.frame().iloc[0].to_dict();r.update(day=DAY,minute=595,feature_available=DAY+' 09:55:01');row=dict(r,feature_row=r)
   with patch.object(signal_runtime,'connect',side_effect=lambda:connect(db)):
    first=signal_runtime.evaluate([row],DAY,595,'route',True,dep);self.assertTrue(first['valid']);self.assertEqual(first,signal_runtime.evaluate([row],DAY,595,'route',True,dep));later=signal_runtime.evaluate([row],DAY,600,'route',True,dep);self.assertFalse(later['valid'])
   con=connect(db);self.assertEqual(con.execute('SELECT count(*) FROM signal_events').fetchone()[0],1);self.assertEqual(con.execute('SELECT count(*) FROM live_fills').fetchone()[0],0);self.assertNotIn('y',json.loads(con.execute('SELECT payload FROM signal_events').fetchone()[0])['feature_row']);con.close()

if __name__=='__main__':unittest.main()
