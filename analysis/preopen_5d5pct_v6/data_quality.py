"""Separate acquisition bounds, valid sessions and actually scoreable 5-day windows."""
from collections import Counter
import json
import numpy as np
import pandas as pd
from common import OUT,OLD,DATES,SESSIONS,sha,write,now

def inspect(path):
 f=pd.read_parquet(path).sort_values('start');p=f[['open','high','low','close']].to_numpy(float);v=f.volume.to_numpy(float)
 good=np.isfinite(p).all(1)&(p>0).all(1)&np.isfinite(v)&(v>=0)&(f.high>=p.max(1))&(f.low<=p.min(1))
 valid=f[good];complete=[];gaps=[];sessions=Counter()
 for day,g in valid.groupby('day',sort=True):
  if day not in SESSIONS:continue
  close=570+SESSIONS[day]['duration_minutes'];r=g[(g.minute>=570)&(g.minute<close)]
  if np.array_equal(r.minute.to_numpy(),np.arange(570,close,5)):complete.append(day)
  else:gaps.append(dict(day=day,expected=SESSIONS[day]['duration_minutes']//5,valid_bars=len(r)))
  sessions.update({'night':int(((g.minute<240)|(g.minute>=1200)).sum()),'pre':int(((g.minute>=240)&(g.minute<570)).sum()),'regular':len(r),'post':int(((g.minute>=close)&(g.minute<1200)).sum())})
 full=set(complete);five=[d for d in complete if DATES.index(d)+4<len(DATES) and set(DATES[DATES.index(d):DATES.index(d)+5])<=full and DATES[DATES.index(d)+4]<='2026-09-30']
 return dict(path=str(path.relative_to(OUT)) if path.is_relative_to(OUT) else str(path.relative_to(OUT.parents[1])),sha256=sha(path),bars=len(f),valid_bars=int(good.sum()),invalid_bars=int((~good).sum()),duplicate_starts=int(f.start.duplicated().sum()),
  first=str(f.start.min()),last=str(f.end.max()),valid_first=str(valid.start.min()),valid_last=str(valid.end.max()),regular_complete_days=len(complete),regular_incomplete_days=len(gaps),
  complete_rth_range=[complete[0],complete[-1]] if complete else None,full_five_day_windows=len(five),scoreable_window_range=[five[0],five[-1]] if five else None,
  sessions=dict(sessions),company_action_adjustment='not applied; raw NONE. Label eligibility additionally requires action audit.',gap_examples=gaps[:3])

def main():
 rows=[]
 for folder in [OLD/'raw',OUT/'cache/acquired',OUT/'cache/backfill']:
  for path in sorted(folder.glob('*.parquet')):
   try:rows.append(dict(symbol=path.stem,**inspect(path)))
   except Exception as e:rows.append(dict(symbol=path.stem,path=str(path),error=repr(e),status='quality_unavailable'))
 write(OUT/'data_quality.json',dict(at=now(),requested=['2024-10-04','2026-09-30'],status='partial_history_not_universe_complete',records=rows,
  note='Date bounds never imply complete sessions. Five-day scoreability is complete RTH-price coverage only; actions and exact entry must also pass. Sparse zero placeholders are invalid, never candles.',
  provider_timestamp_probe=dict(symbol='AAPL',day='2026-10-02',regular_first_time_key='09:35',daily_open_equals_first_5m_open=True,regular_last_time_key='16:00',normalization='time_key is bar end; start=end−5m',source='read-only OpenD K_DAY/K_5M Session.RTH; cache/provider_timestamp_probe.json')))
 print(json.dumps(dict(symbol_files=len(rows),invalid_bars=sum(r.get('invalid_bars',0) for r in rows),quality_errors=sum('error'in r for r in rows))),flush=True)
if __name__=='__main__':main()
