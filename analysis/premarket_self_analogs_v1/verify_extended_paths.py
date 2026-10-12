"""Check displayed future paths against captured bars, including unfinished examples."""
from pathlib import Path
import json,math
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
data=json.loads((OUT/'results.json').read_text());before=json.loads((OUT/'runs/20261008_0900_before_3session_overlay/results.json').read_text())
projection=json.loads(json.dumps(data));projection.pop('curve_extension')
for stock in projection['stocks']:
 for sample in stock.get('samples',[]):sample.pop('forward_3_sessions')
assert projection==before,'The view extension changed matching or original outcome statistics'
a=json.loads((OUT/'acquisition.json').read_text());axis=data['curve_extension'];cal=json.loads((OUT/'trading_days.json').read_text())['days'];days=sorted(x['time'] for x in cal);bydate={d:i for i,d in enumerate(days)}
assert len(axis['prefix_x'])==31 and axis['prefix_x'][-1]==axis['boundary_x']==29
assert axis['reference_sessions']==['2026-10-08','2026-10-09','2026-10-12']
checked=0;partial=0;points=0;actual_sunday_night_points=0;selected_complete=0
for stock in data['stocks']:
 if stock['status']!='ready':continue
 item=next(x for x in a['datasets'] if x['code']==stock['code']);frame=pd.read_parquet(ROOT/item['path']);bars=frame.set_index('time_key').to_dict('index')
 assert len(stock['current_curve'])==len(axis['prefix_x'])
 for sample in stock['samples']:
  path=sample['forward_3_sessions'];assert path['sessions']==days[bydate[sample['date']]:bydate[sample['date']]+3]
  start=bars[sample['feature_keys'][0]]['open'];assert len(path['points'])==len(path['keys'])
  for (x,y),k in zip(path['points'],path['keys']):
   assert x>axis['boundary_x']
   if k>axis['current_cutoff_et']:assert y is None
   if y is not None:
    assert abs(y-100*math.log(bars[k]['close']/start))<1e-6;points+=1
    if pd.Timestamp(k).dayofweek==6:actual_sunday_night_points+=1
  assert path['missing_count']==sum(p[1] is None for p in path['points'])
  outcome=path['outcome'];reference=bars[sample['input_end']]['close'];assert abs(outcome['reference_price']-reference)<1e-8
  observed=[(k,bars[k]) for k,point in zip(path['keys'],path['points']) if point[1] is not None and all(math.isfinite(float(bars[k][f])) and float(bars[k][f])>0 for f in ['close','high','low'])]
  assert outcome['observed_bars']==len(observed) and outcome['expected_bars']==len(path['keys'])
  if observed:
   actual={'close_pct':(observed[-1][1]['close']/reference-1)*100,'high_pct':(max(bar['high'] for _,bar in observed)/reference-1)*100,'low_pct':(min(bar['low'] for _,bar in observed)/reference-1)*100}
   for metric,value in actual.items():
    assert abs(outcome['observed_'+metric]-value)<1e-6
    if outcome['complete']:assert abs(outcome[metric]-value)<1e-6
    else:assert outcome[metric] is None
  assert outcome['complete']==(path['complete'] and len(observed)==len(path['keys']))
  if not path['complete']:partial+=1
  if sample['date'] in stock['selected_dates'] and path['complete']:selected_complete+=1
  checked+=1
result={'passed':True,'historical_samples_checked':checked,'observed_points_checked':points,'unfinished_or_missing_samples_retained':partial,'real_sunday_night_points_checked':actual_sunday_night_points,'selected_complete_3_session_paths':selected_complete,'original_matching_and_statistics_unchanged':True,'three_session_close_high_low_verified_from_raw_ohlc':True,'current_curve_end_x':29,'full_axis_end_x':axis['end_x'],'current_future_points':0,'reference_sessions':axis['reference_sessions']}
(OUT/'extended_path_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
