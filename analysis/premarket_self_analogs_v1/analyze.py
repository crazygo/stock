"""Session-aligned hourly self-analog research, without outcome-based selection."""
from pathlib import Path
from datetime import date,datetime,timedelta
from zoneinfo import ZoneInfo
import json,hashlib,math,collections
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage,fcluster
from scipy.spatial.distance import squareform
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
TODAY=date(2026,10,8);CUTOFF='09:00:00';MIN=TODAY-timedelta(days=360)
PRE=['05:00:00','06:00:00','07:00:00','08:00:00','09:00:00','09:30:00']
RTH=['10:30:00','11:30:00','12:30:00','13:30:00','14:30:00','15:30:00','16:00:00']
POST=['17:00:00','18:00:00','19:00:00','20:00:00']
NIGHT=['21:00:00','22:00:00','23:00:00'];DAWN=['00:00:00','01:00:00','02:00:00','03:00:00','04:00:00']

def key(day,h):return f'{day} {h}'
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
 if isinstance(x,(np.integer,)):return int(x)
 if isinstance(x,(np.floating,float)):return round(float(x),7) if math.isfinite(x) else None
 return x

def metric(samples,field,base):
 pairs=[(s,s.get(field)) for s in samples if s.get(field) is not None]
 if not pairs:return {'n':0,'unknown':len(samples),'mean':None,'median':None,'up_fraction':None}
 vals=[v['close_pct'] for _,v in pairs];ups=sum(v>0 for v in vals);downs=sum(v<0 for v in vals)
 def summary(name):
  x=[v[name] for _,v in pairs]
  return {'mean':float(np.mean(x)),'median':float(np.median(x)),'p25':float(np.percentile(x,25)),'p75':float(np.percentile(x,75))}
 result={'n':len(vals),'unknown':len(samples)-len(vals),'up':ups,'down':downs,'flat':len(vals)-ups-downs,'up_fraction':ups/len(vals),'close':summary('close_pct'),'high':summary('high_pct'),'low':summary('low_pct'),'up_close_mean':float(np.mean([v for v in vals if v>0])) if ups else None,'down_close_mean':float(np.mean([v for v in vals if v<0])) if downs else None,'closed_before_24h':sum(s.get('future_24h_market_closes',False) for s,v in pairs)}
 result['mapped_prices']={name:base*(1+result[name]['median']/100) for name in ['close','high','low']}
 if field=='future_rth':result['open_to_close_mean']=float(np.mean([v['open_to_close_pct'] for _,v in pairs]));result['open_to_close_median']=float(np.median([v['open_to_close_pct'] for _,v in pairs]))
 return result

def main():
 proto=json.loads((OUT/'protocol.json').read_text());acq=json.loads((OUT/'acquisition.json').read_text());pos=json.loads((OUT/'positions.json').read_text());snap=json.loads((OUT/'snapshot.json').read_text());metadata={x['code']:x for x in json.loads((OUT/'basicinfo.json').read_text())}
 cal=json.loads((OUT/'trading_days.json').read_text())['days'];days=sorted(date.fromisoformat(x['time'][:10]) for x in cal)
 sessions={d:False for d in days}
 old=json.loads((ROOT/'market_data/calendars/nasdaq_sessions_2026_v1.json').read_text())
 for s in old['sessions']:
  d=date.fromisoformat(s['session_date'])
  if d in sessions:sessions[d]=s['is_early_close']
 # OpenD trade_date_type distinguishes half-days outside the stored official-calendar subset.
 for x in cal:
  if str(x.get('trade_date_type','WHOLE')).upper() != 'WHOLE':sessions[date.fromisoformat(x['time'][:10])]=True
 idx={d:i for i,d in enumerate(days)}
 def rthslots(d):return ['10:30:00','11:30:00','12:30:00','13:00:00'] if sessions[d] else RTH
 def inputkeys(day):
  prev=days[idx[day]-1]
  if sessions[prev]:return None,prev
  before=day-timedelta(days=1)
  return [key(prev,h) for h in PRE+RTH+POST]+[key(before,h) for h in NIGHT]+[key(day,h) for h in DAWN+PRE[:5]],prev
 def forwardkeys(day):
  end=day+timedelta(days=1)
  post=['14:00:00','15:00:00','16:00:00','17:00:00'] if sessions[day] else POST
  keys=[key(day,h) for h in PRE[5:]+rthslots(day)+post]
  if end in sessions:keys += [key(day,h) for h in NIGHT]+[key(end,h) for h in DAWN+PRE[:5]]
  return keys,end not in sessions
 def valid(bar):return bar is not None and all(math.isfinite(float(bar[v])) and float(bar[v])>0 for v in ['open','close','high','low'])
 def fetch(bars,keys):
  missing=[k for k in keys if not valid(bars.get(k))]
  return ([bars[k] for k in keys] if not missing else None),missing
 def outcome(seq,base,keys):
  return {'reference_price':base,'close_pct':100*(seq[-1]['close']/base-1),'high_pct':100*(max(x['high'] for x in seq)/base-1),'low_pct':100*(min(x['low'] for x in seq)/base-1),'terminal_at_et':keys[-1],'open_to_close_pct':100*(seq[-1]['close']/seq[0]['open']-1),'curve':[100*(x['close']/base-1) for x in seq],'labels':[x[11:16] for x in keys],'ohlc':[[100*(x[k]/base-1) for k in ['open','high','low','close']] for x in seq]}
 records=[];audit={'source_hashes_checked':0,'future_never_used_for_selection':True,'bar_labels':'end_at_ET','current_bar_never_used':True,'missing_never_interpolated':True,'stocks':[]}
 for position in pos['positions']:
  code=position['code'];quote=next((x for x in snap['snapshots'] if x['code']==code),{})
  rec={'code':code,'name':position['stock_name'],'qty':position['qty'],'quote':{k:quote.get(k) for k in ['pre_price','pre_change_rate','pre_volume','update_time','last_price']},'status':'not_applicable' if not code.startswith('US.') else 'pending','warnings':[]}
  if code.startswith('US.') and quote.get('update_time'):
   try:
    last=datetime.fromisoformat(quote['update_time']).replace(tzinfo=ZoneInfo('America/New_York'));received=datetime.fromisoformat(snap['received_at']);age=(received-last).total_seconds()/60
    rec['quote']['update_age_minutes']=max(0,age)
    if age>5:rec['warnings'].append(f'盘前报价最后更新距获取时点{age:.1f}分钟；价格可能陈旧。')
   except (ValueError,TypeError):rec['warnings'].append('盘前报价更新时间无法核实。')
  if not code.startswith('US.'):
   rec['reason']='港股交易时段不同，不并入美股盘前窗口。';records.append(rec);continue
  dataset=next((d for d in acq['datasets'] if d['code']==code),None)
  if not dataset:rec['status']='data_unavailable';records.append(rec);continue
  p=ROOT/dataset['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==dataset['sha256'];audit['source_hashes_checked']+=1
  f=pd.read_parquet(p);f=f[f.time_key<=key(TODAY,CUTOFF)].copy()
  bars={x['time_key']:x for x in f.to_dict('records')}
  ik,prev=inputkeys(TODAY);current,miss=fetch(bars,ik)
  rec['coverage']={'requested_start':str(MIN),'first':str(f.time_key.min()),'last':str(f.time_key.max()),'rows':len(f),'listing_date':metadata.get(code,{}).get('listing_date')}
  if current is None:rec['status']='current_incomplete';rec['missing_current']=miss;records.append(rec);continue
  start=current[0]['open'];y=np.r_[0,100*np.log(np.array([b['close'] for b in current])/start)]
  rng=float(np.ptp(y));base=float(current[-1]['close']);rec.update(status='ready',reference_price=base,current_curve=y.tolist(),current_labels=['前日04:00']+['前日'+k[11:16] if i<17 else '夜盘'+k[11:16] if i<25 else '本日'+k[11:16] for i,k in enumerate(ik)],current_keys=ik,window={'start':key(prev,'04:00:00'),'end':key(TODAY,CUTOFF),'trading_session_hours':29,'calendar_hours':(datetime.fromisoformat(key(TODAY,CUTOFF))-datetime.fromisoformat(key(prev,'04:00:00'))).total_seconds()/3600,'price_points':len(y),'range_log_pp':rng,'zero_volume_bars':sum(b['volume']==0 for b in current)})
  rec['phase_returns']={'previous_RTH_pct':100*(current[12]['close']/current[6]['open']-1),'previous_postmarket_pct':100*(current[16]['close']/current[12]['close']-1),'overnight_pct':100*(current[24]['close']/current[16]['close']-1),'current_premarket_pct':100*(current[-1]['close']/current[24]['close']-1),'at_09_vs_previous_RTH_close_pct':100*(base/current[12]['close']-1)}
  if rng<.10:rec['warnings'].append('当前曲线幅度很小，相关系数容易受少量成交影响。')
  if sum(b['volume']==0 for b in current)>8:rec['warnings'].append('当前窗口存在较多零成交量小时，提供方沿用价格不代表成交活跃。')
  if str(f.time_key.min())[:10]>str(MIN):rec['warnings'].append('行情历史不足360天。')
  samples=[];rejected=collections.Counter();rejected_days=[];daily=[]
  for day in days:
   if day>=TODAY or day<MIN or idx[day]==0:continue
   keys,previous=inputkeys(day)
   if keys is None:rejected['previous_half_day']+=1;rejected_days.append({'date':str(day),'reason':'previous_half_day'});continue
   seq,missing=fetch(bars,keys)
   if seq is None:rejected['missing_input']+=1;rejected_days.append({'date':str(day),'reason':'missing_input','missing_count':len(missing)});continue
   curve=np.r_[0,100*np.log(np.array([b['close'] for b in seq])/seq[0]['open'])]
   r=float(np.corrcoef(y,curve)[0,1]) if rng>1e-8 and np.ptp(curve)>1e-8 else None
   amp=float(np.ptp(curve)/rng) if rng>1e-8 else None
   b=float(seq[-1]['close']);fk,closed=forwardkeys(day);future,missing_future=fetch(bars,fk);rk=[key(day,h) for h in rthslots(day)];regular,missing_rth=fetch(bars,rk)
   sample={'date':str(day),'previous_date':str(previous),'r':r,'amplitude_ratio':amp,'amplitude_pass':amp is not None and .5<=amp<=2,'curve':curve.tolist(),'reference_price':b,'input_start':key(previous,'04:00:00'),'input_end':key(day,CUTOFF),'feature_keys':keys,'calendar_hours':(datetime.fromisoformat(key(day,CUTOFF))-datetime.fromisoformat(key(previous,'04:00:00'))).total_seconds()/3600,'zero_volume_bars':sum(x['volume']==0 for x in seq),'future_24h':outcome(future,b,fk) if future else None,'future_rth':outcome(regular,b,rk) if regular else None,'future_24h_market_closes':closed,'future_24h_missing':missing_future,'future_rth_missing':missing_rth}
   sample['future_24h_comparable']=sample['future_24h'] if not closed else None
   samples.append(sample)
  for day in days:
   if day>TODAY:continue
   rk=[key(day,h) for h in rthslots(day)];regular,missing=fetch(bars,rk)
   if regular:daily.append({'date':str(day),'open':regular[0]['open'],'high':max(x['high'] for x in regular),'low':min(x['low'] for x in regular),'close':regular[-1]['close']})
  selected=[s for s in samples if s['r'] is not None and s['r']>=proto['threshold_N'] and s['amplitude_pass']]
  if len(selected)>1:
   arr=np.array([s['curve'] for s in selected]);corr=np.corrcoef(arr);dist=np.clip(1-corr,0,2);np.fill_diagonal(dist,0)
   groups=fcluster(linkage(squareform(dist,checks=False),method='complete'),t=.10,criterion='distance')
  else:groups=np.ones(len(selected),dtype=int)
  for s,g in zip(selected,groups):s['cluster']=int(g)
  clusters=[]
  for group in sorted(set(int(x) for x in groups)):
   subset=[s for s in selected if s['cluster']==group];center=np.median([s['curve'] for s in subset],axis=0)
   medoid=min(subset,key=lambda s:float(np.sum((np.array(s['curve'])-center)**2)))['date']
   clusters.append({'id':group,'n':len(subset),'medoid_date':medoid,'dates':[s['date'] for s in subset],'future_24h':metric(subset,'future_24h',base),'future_rth':metric(subset,'future_rth',base)})
  rec.update(eligible_input_days=len(samples),excluded_days=dict(rejected),excluded_date_evidence=rejected_days,matched_n=len(selected),selected_dates=[s['date'] for s in selected],samples=sorted(samples,key=lambda s:(s['r'] if s['r'] is not None else -2),reverse=True),clusters=clusters,daily=daily,stats={'future_24h':metric(selected,'future_24h',base),'future_24h_comparable':metric([s for s in selected if not s['future_24h_market_closes']],'future_24h_comparable',base),'future_rth':metric(selected,'future_rth',base)},baseline={h:metric(samples,h,base) for h in ['future_24h','future_rth']},sensitivity=[])
  for threshold in proto['threshold_sensitivity']:
   ss=[s for s in samples if s['r'] is not None and s['r']>=threshold and s['amplitude_pass']]
   rec['sensitivity'].append({'threshold':threshold,'n':len(ss),'future_24h':metric(ss,'future_24h',base),'future_rth':metric(ss,'future_rth',base)})
  if len(selected)<12:rec['warnings'].append('相似样本不足12个，只能查看个案，不能作稳定方向判断。')
  for s in samples:
   assert max(s['feature_keys'])==key(date.fromisoformat(s['date']),CUTOFF)
   assert s['date']<str(TODAY)
   if s['future_24h']:assert s['future_24h']['terminal_at_et']>s['input_end']
  audit['stocks'].append({'code':code,'features':len(samples),'matched':len(selected),'clusters':len(clusters),'unknown_24h':rec['stats']['future_24h']['unknown'],'unknown_rth':rec['stats']['future_rth']['unknown'],'excluded':dict(rejected)})
  records.append(rec)
 result={'version':proto['version'],'protocol':proto,'snapshot_received_at':snap['received_at'],'position_received_at':pos['received_at'],'generated_at':datetime.now(ZoneInfo('UTC')).isoformat(),'stocks':records,'notes':['上涨比例是历史条件频率，不是校准后的未来概率。','相邻历史窗口与未来24小时路径存在重叠；样本非独立。','高低价中位数分别取每个历史窗口后续最高/最低价收益率，再映射到当前09:00价格，不是保障范围。','主24h严格按日历24小时；遇到周末或假日提前停止有交易价格。另列完整延伸至次日盘前的可比子集。','港股单列；SOXS为反向杠杆ETF，历史结果不与普通股票合并。','全序列QFQ用于形态和收益率；历史美元价格不用于模拟实际成交。']}
 from extend_paths import extend_results
 result=extend_results(result,acq,cal)
 (OUT/'results.json').write_text(json.dumps(clean(result),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 audit['generated_at']=result['generated_at'];audit['passed']=audit['source_hashes_checked']==19
 (OUT/'verification.json').write_text(json.dumps(clean(audit),ensure_ascii=False,indent=2)+'\n')
 for rec in records:
  if rec['status']=='ready':
   a=rec['stats']['future_24h'];r=rec['stats']['future_rth'];print(rec['code'],'matched',rec['matched_n'],'24h_n',a['n'],'median',a.get('close',{}).get('median'),'high',a.get('high',{}).get('median'),'low',a.get('low',{}).get('median'),'RTHmedian',r.get('close',{}).get('median'),'clusters',len(rec['clusters']))
  else:print(rec['code'],rec['status'])

if __name__=='__main__':main()
