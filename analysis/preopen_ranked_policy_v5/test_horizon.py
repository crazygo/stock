"""Horizon boundaries, censoring and temporal purge contracts."""
import unittest
import pandas as pd
import numpy as np
from horizon import label_rows,window_end,split_dates,DATES,SESSIONS,replay

def raw_days(days,high=101.):
 rows=[]
 for day in days:
  for minute in range(570,570+SESSIONS[day]['duration_minutes'],5):
   start=pd.Timestamp(day)+pd.Timedelta(minutes=minute);rows.append(dict(day=day,minute=minute,start=start,end=start+pd.Timedelta(minutes=5),open=100.,high=high,low=99.,close=100.,volume=1000.))
 return pd.DataFrame(rows)

class HorizonContracts(unittest.TestCase):
 def test_weekend_holiday_and_entry_day_count(self):
  self.assertEqual(window_end('2026-05-22','5d5pct'),'2026-05-29 16:00:00')
  self.assertEqual(window_end('2026-06-29','5d5pct'),'2026-07-06 16:00:00')
 def test_half_day_close(self):
  self.assertEqual(window_end('2026-11-20','5d5pct'),'2026-11-27 13:00:00')
 def test_before_entry_high_cannot_win(self):
  days=[d for d in DATES if '2026-05-04'<=d<='2026-05-08'];raw=raw_days(days);raw.loc[(raw.day=='2026-05-04')&(raw.minute==570),'high']=120
  result=label_rows(raw,pd.DataFrame([dict(day='2026-05-04',minute=575)]),'5d5pct');self.assertEqual(result.y.iloc[0],0);self.assertAlmostEqual(result.entry.iloc[0],100.1)
 def test_last_window_bar_counts_and_later_bar_does_not(self):
  days=[d for d in DATES if '2026-05-04'<=d<='2026-05-11'];raw=raw_days(days);raw.loc[(raw.day=='2026-05-11')&(raw.minute==570),'high']=120;keys=pd.DataFrame([dict(day='2026-05-04',minute=575)])
  self.assertEqual(label_rows(raw,keys,'5d5pct').y.iloc[0],0)
  raw.loc[(raw.day=='2026-05-08')&(raw.minute==955),'high']=110;self.assertEqual(label_rows(raw,keys,'5d5pct').y.iloc[0],1)
 def test_pending_hit_is_not_early_success(self):
  raw=raw_days(['2026-09-30'],high=120);r=label_rows(raw,pd.DataFrame([dict(day='2026-09-30',minute=575)]),'5d5pct');self.assertTrue(pd.isna(r.y.iloc[0]));self.assertEqual(r.future_status.iloc[0],'pending_window')
 def test_future_missing_bar_not_failure_or_success(self):
  days=[d for d in DATES if '2026-05-04'<=d<='2026-05-08'];raw=raw_days(days,high=120);raw=raw[~((raw.day=='2026-05-07')&(raw.minute==700))];r=label_rows(raw,pd.DataFrame([dict(day='2026-05-04',minute=575)]),'5d5pct');self.assertTrue(pd.isna(r.y.iloc[0]));self.assertEqual(r.future_status.iloc[0],'missing_future_bars')
 def test_future_action_is_unknown(self):
  days=[d for d in DATES if '2026-05-04'<=d<='2026-05-08'];r=label_rows(raw_days(days),pd.DataFrame([dict(day='2026-05-04',minute=575)]),'5d5pct',actions={'2026-05-06'});self.assertEqual(r.future_status.iloc[0],'corporate_action_window')
 def test_invalid_future_ohlc_not_a_failure(self):
  days=[d for d in DATES if '2026-05-04'<=d<='2026-05-08'];raw=raw_days(days);raw.loc[(raw.day=='2026-05-06')&(raw.minute==700),'high']=np.nan;r=label_rows(raw,pd.DataFrame([dict(day='2026-05-04',minute=575)]),'5d5pct');self.assertTrue(pd.isna(r.y.iloc[0]));self.assertEqual(r.future_status.iloc[0],'missing_future_bars')
 def test_purge_every_boundary_for_both_targets(self):
  for month in [f'2026-{m:02d}' for m in range(5,11)]:
   td,cd,sd=split_dates(month,set(DATES))
   for target in ['5d5pct','10d10pct']:
    self.assertLess(window_end(td[-1],target)[:10],cd[0]);self.assertLess(window_end(cd[-1],target)[:10],sd[0]);self.assertLess(window_end(sd[-1],target)[:10],month+'-01')
 def test_unknown_top_kept_and_no_second_replacement(self):
  rows=[]
  for minute in [575,580]:
   for symbol,score,y in [('AAA',.9,np.nan),('BBB',.8,1)]:rows.append(dict(day='2026-05-04',minute=minute,symbol=symbol,score=score,y=y,ex_action=0,feature_available=f'2026-05-04 {minute//60:02d}:{minute%60:02d}:01',baseline=.5,reference=100.,entry=100.1,target=105.105,label_end='2026-05-08 16:00:00',future_status='missing_future_bars' if symbol=='AAA' else 'mature'))
  sim=replay(pd.DataFrame(rows),{'pre':.7,'regular':.7},'5d5pct');self.assertEqual([e['symbol'] for e in sim['events']],['AAA','BBB']);self.assertEqual(sim['metrics']['pending'],1);self.assertEqual(sim['metrics']['n'],2)

if __name__=='__main__':unittest.main()
