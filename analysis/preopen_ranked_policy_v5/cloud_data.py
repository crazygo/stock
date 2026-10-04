"""Read-only Local -> R2 -> optional reachable OpenD for the frozen registry."""
from __future__ import annotations
import json,os,socket,time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT,OUT,ET,SESSIONS,save
from data import read_raw

def normalize_archive(f):
 if {'start','end','available','day','minute','open','high','low','close','volume'}.issubset(f):n=f.copy()
 elif {'start_at_et','end_at_et','price_basis','open','high','low','close','volume'}.issubset(f) and set(f.price_basis.dropna())=={'NONE'}:
  n=f[['open','high','low','close','volume']].copy()
  for dst,src in [('start','start_at_et'),('end','end_at_et')]:n[dst]=pd.to_datetime(f[src],utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
  n['available']=n.end+pd.Timedelta(seconds=1);n['day']=n.start.dt.strftime('%Y-%m-%d');n['minute']=n.start.dt.hour*60+n.start.dt.minute
 else:raise ValueError('Unknown time/price basis; requires explicit NONE 5m bars')
 for c in ['open','high','low','close','volume']:n[c]=pd.to_numeric(n[c],errors='raise').astype(float)
 for c in ['start','end','available']:n[c]=pd.to_datetime(n[c])
 prices=n[['open','high','low','close']].to_numpy(float);valid=np.isfinite(prices).all(1)&(prices>0).all(1)&np.isfinite(n.volume)&(n.volume>=0)&(n.high>=prices.max(1))&(n.low<=prices.min(1))&(n.end-n.start==pd.Timedelta(minutes=5))
 return n[valid].drop_duplicates('start',keep='last').sort_values('start').reset_index(drop=True)

def covered(f,day,cut):
 if f.empty:return False
 today=f[(f.day==day)&(f.minute>=240 if cut<=570 else f.minute>=570)&(f.minute<cut)]
 prefix=np.array_equal(today.minute.to_numpy(),np.arange(240 if cut<=570 else 570,cut,5))
 # At least 30 preceding complete sessions needed for long rolling features.
 close={d:570+r['duration_minutes'] for d,r in SESSIONS.items()};prior=f[(f.day<day)&(f.minute>=570)&(f.minute<f.day.map(close))]
 count=sum(np.array_equal(g.minute.to_numpy(),np.arange(570,close[d],5)) for d,g in prior.groupby('day') if d in close)
 return bool(prefix and count>=30)

class MarketData:
 def __init__(self,directory=None):
  self.directory=Path(directory) if directory else OUT/'raw';self.use_archive=directory is None;self.cache={};self.audit=[];self.errors=[]
 def raw(self,s):
  if s not in self.cache:
   p=self.directory/f'{s}.parquet'
   empty=pd.DataFrame(columns=['start','end','available','day','minute','open','high','low','close','volume'])
   self.cache[s]=normalize_archive(pd.read_parquet(p)) if p.exists() else normalize_archive(read_raw(s) if self.use_archive else empty)
  return self.cache[s]
 def merge(self,s,n,source,now,received_at=None):
  f=pd.concat([self.raw(s),n],ignore_index=True).drop_duplicates('start',keep='last').sort_values('start').reset_index(drop=True);save(f,self.directory/f'{s}.parquet');self.cache[s]=f
  self.audit.append(dict(symbol=s,source=source,received_at=received_at or datetime.now(timezone.utc).isoformat(),loaded_at=datetime.now(timezone.utc).isoformat(),latest_bar_end=str(f.end.max()),rows=len(f)))
 def refresh(self,symbols,day,cut,futu_host=None,futu_port=11111,progress=None):
  from scripts.r2_client import R2Client
  missing=[s for s in symbols if not covered(self.raw(s),day,cut)]
  self.audit+=[dict(symbol=s,source='local',status='covered',received_at=None) for s in symbols if s not in missing]
  if not missing:return
  if progress:progress(f'本地不足 {len(missing)} 只，检查 R2（只读）')
  try:
   client=R2Client();inventory=client.list_objects('us_5m/')+client.list_objects('model_training_history_v1/parts/')
   begin=str((pd.Timestamp(day)-pd.Timedelta(days=110)).date());wanted=[]
   for obj in inventory:
    key=obj['key'];parts=key.split('/');s=parts[1] if key.startswith('us_5m/') and len(parts)==3 else parts[2] if key.startswith('model_training_history_v1/parts/') and len(parts)>=5 else None
    if s not in missing or not key.endswith('.parquet'):continue
    if key.startswith('us_5m/') and parts[2][:4]!=day[:4]:continue
    if key.startswith('model_training_history_v1/') and not begin[:7]<=parts[3]<=day[:7]:continue
    wanted.append((s,obj))
   def get(item):
    s,obj=item;key=obj['key'];p=self.directory/'r2_cache'/key;stamp=p.with_suffix('.etag.json');old=json.loads(stamp.read_text()) if stamp.exists() else {}
    if not p.exists() or old.get('etag')!=obj['etag']:
     p.parent.mkdir(parents=True,exist_ok=True);client.get_object(key,p);stamp.write_text(json.dumps({'etag':obj['etag'],'received_at':datetime.now(timezone.utc).isoformat()}))
    receipt=json.loads(stamp.read_text()).get('received_at') if stamp.exists() else None
    return s,normalize_archive(pd.read_parquet(p)),key,receipt
   # Network reads only; broker fallback remains sequential and throttled.
   with ThreadPoolExecutor(max_workers=4) as pool:
    futures=[pool.submit(get,x) for x in wanted]
    for future in futures:
     try:s,n,key,receipt=future.result();self.merge(s,n,'R2:'+key,datetime.now(ET),received_at=receipt)
     except Exception as e:self.errors.append(dict(source='R2',reason=type(e).__name__,detail=str(e)[:160]))
  except Exception as e:self.errors.append(dict(source='R2',reason=type(e).__name__,detail=str(e)[:160]))
  missing=[s for s in symbols if not covered(self.raw(s),day,cut)]
  if not missing:return
  host=futu_host or os.getenv('PREOPEN_FUTU_HOST','127.0.0.1');port=int(os.getenv('PREOPEN_FUTU_PORT',str(futu_port)))
  try:
   with socket.create_connection((host,port),timeout=2):pass
  except OSError:
   self.errors.append(dict(source='OpenD',reason='not_reachable',missing=len(missing),detail='Configure PREOPEN_FUTU_HOST/PORT for an accessible quote gateway; no orders sent.'));return
  if progress:progress(f'R2 后仍不足 {len(missing)} 只，使用 OpenD 行情接口')
  import futu as ft
  ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host=host,port=port)
  try:
   for i,s in enumerate(missing,1):
    from acquire import normalize
    f=self.raw(s);past=f[f.day<=day];start=max(str((pd.Timestamp(day)-pd.Timedelta(days=110)).date()),str(past.day.max()) if len(past) and covered(past,day,cut) else str((pd.Timestamp(day)-pd.Timedelta(days=110)).date()));pages=[];key=None
    # Recent live subscription first when the 30-session context already exists.
    ret=None
    if len(past) and day==datetime.now(ET).date().isoformat():
     ret,_=q.subscribe(['US.'+s],[ft.SubType.K_5M],subscribe_push=False,extended_time=True,session=ft.Session.ALL)
     if ret==ft.RET_OK:
      ret,recent=q.get_cur_kline('US.'+s,1000,ktype=ft.KLType.K_5M,autype=ft.AuType.NONE)
      if ret==ft.RET_OK:self.merge(s,normalize(recent),'OpenD_current',datetime.now(ET))
    if not covered(self.raw(s),day,cut):
     while True:
      ret,bars,key=q.request_history_kline('US.'+s,start=start,end=day,ktype=ft.KLType.K_5M,autype=ft.AuType.NONE,fields=ft.KL_FIELD.ALL,max_count=1000,page_req_key=key,extended_time=True,session=ft.Session.ALL)
      if ret!=ft.RET_OK:break
      pages.append(bars)
      if not key:break
      time.sleep(.25)
     if ret==ft.RET_OK and pages:self.merge(s,normalize(pd.concat(pages,ignore_index=True)),'OpenD_history',datetime.now(ET))
     else:self.errors.append(dict(symbol=s,source='OpenD',reason='history_unavailable'))
    if progress and i%10==0:progress(f'OpenD 已检查 {i}/{len(missing)}')
    time.sleep(1.2)
  finally:q.close()
