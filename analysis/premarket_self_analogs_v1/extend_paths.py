"""Append observed historical 3-session paths; never use them in analog selection."""
from pathlib import Path
from datetime import date,timedelta
import json,math,hashlib,shutil
import pandas as pd
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
PRE=['05:00:00','06:00:00','07:00:00','08:00:00','09:00:00','09:30:00']
FULL_RTH=['10:30:00','11:30:00','12:30:00','13:30:00','14:30:00','15:30:00','16:00:00']
HALF_RTH=['10:30:00','11:30:00','12:30:00','13:00:00']
NIGHT=['21:00:00','22:00:00','23:00:00'];DAWN=['00:00:00','01:00:00','02:00:00','03:00:00','04:00:00']

def hour(label):
 h,m,_=map(int,label.split(':'));return h+m/60

def extend_results(result,acquisition,calendar):
 days=sorted(x['time'][:10] for x in calendar);types={x['time'][:10]:x.get('trade_date_type','WHOLE') for x in calendar};lookup={d:i for i,d in enumerate(days)}
 cutoff=result['protocol']['cutoff_et'];clock=hour(cutoff);boundary=24+clock-4;end=boundary+48+20-clock
 batch_end=result['protocol']['reference_date_et']+' '+cutoff
 def schedule(day):
  ds=days[lookup[day]:lookup[day]+3];slots=[]
  for i,d in enumerate(ds):
   half=types[d]!='WHOLE';rth=HALF_RTH if half else FULL_RTH;post=['14:00:00','15:00:00','16:00:00','17:00:00'] if half else ['17:00:00','18:00:00','19:00:00','20:00:00']
   if i:
    previous_calendar=(date.fromisoformat(d)-timedelta(days=1)).isoformat()
    slots += [(boundary+24*i+hour(t)-24-clock,previous_calendar+' '+t) for t in NIGHT]
    slots += [(boundary+24*i+hour(t)-clock,d+' '+t) for t in DAWN]
   for t in PRE+rth+post:
    if i==0 and hour(t)<=clock:continue
    slots.append((boundary+24*i+hour(t)-clock,d+' '+t))
  return ds,slots
 # The input template has a complete prior session; compress only known market closures.
 prefix_x=[0]+[hour(t)-4 for t in PRE+FULL_RTH+['17:00:00','18:00:00','19:00:00','20:00:00']]+[hour(t)-4 for t in NIGHT]+[20+hour(t) for t in DAWN]+[20+hour(t) for t in PRE if hour(t)<=clock]
 refdays,_=schedule(result['protocol']['reference_date_et'])
 result['curve_extension']={'version':'historical_overlay_3_sessions_v1','day1_rule':'reference_day_remainder_is_day1; through_day3_postmarket_close','current_cutoff_et':batch_end,'reference_sessions':refdays,'prefix_x':prefix_x,'boundary_x':boundary,'end_x':end,'future_used_in_similarity':False,'normalization':'100*ln(close/input_start_open), identical across prefix and extension','missing_rule':'null breaks line; structural market closures have no fabricated prices'}
 for stock in result['stocks']:
  if stock['status']!='ready':continue
  item=next(x for x in acquisition['datasets'] if x['code']==stock['code']);path=ROOT/item['path']
  assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
  frame=pd.read_parquet(path);frame=frame[frame.time_key<=batch_end];bars=frame.set_index('time_key').to_dict('index')
  for sample in stock['samples']:
   ds,slots=schedule(sample['date']);start=float(bars[sample['feature_keys'][0]]['open']);points=[];keys=[];missing=[]
   for x,k in slots:
    bar=bars.get(k);price=float(bar['close']) if bar is not None else None
    good=price is not None and math.isfinite(price) and price>0
    points.append([x,round(100*math.log(price/start),7) if good else None]);keys.append(k)
    if not good:missing.append(k)
   sample['forward_3_sessions']={'sessions':ds,'points':points,'keys':keys,'complete':len(ds)==3 and not missing,'missing_count':len(missing),'missing_keys':missing,'last_real_at_et':next((k for k,p in reversed(list(zip(keys,points))) if p[1] is not None),None),'target_end_at_et':keys[-1] if keys else None}
   path=sample['forward_3_sessions'];reference=float(bars[sample['input_end']]['close']);observed=[]
   assert abs(reference-float(sample['reference_price']))<1e-6  # Stored result prices are rounded to 7 decimals.
   for k,point in zip(keys,points):
    if point[1] is None:continue
    bar=bars[k]
    if all(math.isfinite(float(bar[field])) and float(bar[field])>0 for field in ['close','high','low']):observed.append((k,bar))
   metrics_complete=path['complete'] and len(observed)==len(keys)
   def change(value):return round((float(value)/reference-1)*100,7)
   observed_close=change(observed[-1][1]['close']) if observed else None
   observed_high=change(max(float(bar['high']) for _,bar in observed)) if observed else None
   observed_low=change(min(float(bar['low']) for _,bar in observed)) if observed else None
   path['outcome']={'complete':metrics_complete,'reference_at_et':sample['input_end'],'reference_price':reference,'terminal_at_et':observed[-1][0] if observed else None,'close_pct':observed_close if metrics_complete else None,'high_pct':observed_high if metrics_complete else None,'low_pct':observed_low if metrics_complete else None,'observed_close_pct':observed_close,'observed_high_pct':observed_high,'observed_low_pct':observed_low,'observed_bars':len(observed),'expected_bars':len(keys)}
 return result

def main():
 result=json.loads((OUT/'results.json').read_text());a=json.loads((OUT/'acquisition.json').read_text());cal=json.loads((OUT/'trading_days.json').read_text())['days']
 backup=OUT/'runs/20261008_0900_before_3session_overlay';backup.mkdir(parents=True,exist_ok=True)
 for name in ['results.json','index.html','delivery_manifest.json']:
  if (OUT/name).exists() and not (backup/name).exists():shutil.copy2(OUT/name,backup/name)
 result=extend_results(result,a,cal)
 (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 (OUT/'curve_extension.json').write_text(json.dumps(result['curve_extension'],ensure_ascii=False,indent=2))
 print('extended',sum(len(s.get('samples',[])) for s in result['stocks']),'historical stock-days; today stops at',result['curve_extension']['current_cutoff_et'])

if __name__=='__main__':main()
