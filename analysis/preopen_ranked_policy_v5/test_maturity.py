import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from common import ET
import maturity
from store import connect,dump,record_predictions

DAY='2026-10-05'
def raw():
 minute=np.arange(240,960,5);start=pd.Timestamp(DAY)+pd.to_timedelta(minute,unit='m')
 return pd.DataFrame(dict(day=DAY,minute=minute,start=start,end=start+pd.Timedelta(minutes=5),open=100.,high=101.,low=99.,close=100.,volume=100000.))
def now(clock):return pd.Timestamp(DAY+' '+clock,tz=ET).to_pydatetime()

class Maturity(unittest.TestCase):
 def test_no_early_maturity_or_pre_only_hit(self):
  f=raw();f.loc[f.minute<570,'high']=120
  self.assertIsNone(maturity.outcome(f,DAY,250,now('15:59:59')))
  r=maturity.outcome(f,DAY,250,now('16:00:01'));self.assertEqual(r['hit'],0);self.assertAlmostEqual(r['entry'],100.1)
 def test_future_missing_remains_pending(self):
  f=raw();self.assertIsNone(maturity.outcome(f[f.minute!=950],DAY,600,now('16:01')))
 def test_regular_hit_matches_entry_target(self):
  f=raw();f.loc[f.minute==650,'high']=105
  r=maturity.outcome(f,DAY,600,now('16:01'));self.assertEqual(r['hit'],1);self.assertAlmostEqual(r['net'],.02897)
 def test_non_buy_forecasts_mature_idempotently_without_trades(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);(p/'phase_deployment.json').write_text(json.dumps(dict(observe_from=DAY)));db=p/'test.sqlite';con=connect(db)
   payload=dict(reference_day=DAY,entry_at=DAY+'T10:00:00-04:00',current_probability=True,rows=[dict(symbol='A',score=.8),dict(symbol='B',score=.4)])
   con.execute('INSERT INTO observations(observed_at,decision_key,market_state,action,reason,payload) VALUES(?,?,?,?,?,?)',(DAY+'T09:55:30-04:00','one','regular','no_action','model_not_admitted',dump(payload)))
   record_predictions(con,1,payload['rows'],DAY,595)
   payload['current_probability']=False
   con.execute('INSERT INTO observations(observed_at,decision_key,market_state,action,reason,payload) VALUES(?,?,?,?,?,?)',(DAY+'T15:50:30-04:00','late','regular','no_action','market_closed',dump(payload)))
   con.commit();con.close();f=raw();f.loc[f.minute==650,'high']=105;bad=f[f.minute!=950]
   with patch.object(maturity,'OUT',p),patch.object(maturity,'connect',side_effect=lambda:connect(db)):
    self.assertEqual(maturity.settle(lambda s:f if s=='A' else bad,now('16:01')),1)
    self.assertEqual(maturity.settle(lambda s:f if s=='A' else bad,now('16:02')),0)
    con=connect(db);summary=maturity.summary(con);self.assertEqual(summary['prediction_rows'],2);self.assertEqual(summary['mature'],1);self.assertEqual(summary['pending'],1);self.assertAlmostEqual(summary['phases']['regular']['brier'],.04);self.assertEqual(con.execute('SELECT count(*) FROM trades').fetchone()[0],0);con.close()

if __name__=='__main__':unittest.main()
