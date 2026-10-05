"""Read-only Local -> R2 -> optional reachable OpenD for the frozen registry."""
from __future__ import annotations
import json,os,socket,time,subprocess,sys,tempfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT,OUT,ET,SESSIONS,save,write
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
  self.directory=Path(directory) if directory else OUT/'raw';self.use_archive=directory is None;self.cache={};self.audit=[];self.errors=[];self._audit_path=None
 def checkpoint(self):
  if self._audit_path:
   p=Path(self._audit_path);temp=p.with_suffix('.tmp');temp.write_text(json.dumps(dict(attempts=self.audit,errors=self.errors),ensure_ascii=False));temp.replace(p)
 def refresh_bounded(self,symbols,day,cut,futu_host=None,futu_port=11111,progress=None,budget_seconds=30):
  """Run network/SDK work in a killable child; a hung gateway cannot hide the report."""
  if budget_seconds<=0:raise ValueError('Data acquisition budget must be positive')
  if progress:progress(f'只读补数据预算 {budget_seconds:g} 秒；配置从仓库 config/r2_storage.json 自动读取')
  with tempfile.TemporaryDirectory(prefix='preopen-acquire-') as td:
   request=Path(td)/'request.json';audit=Path(td)/'audit.json'
   request.write_text(json.dumps(dict(symbols=list(symbols),day=day,cut=cut,directory=None if self.use_archive else str(self.directory.resolve()),futu_host=futu_host,futu_port=futu_port,budget_seconds=budget_seconds,audit_path=str(audit))))
   process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--refresh-worker',str(request)],cwd=ROOT,stdout=subprocess.DEVNULL)
   timed_out=False
   try:process.wait(timeout=budget_seconds)
   except subprocess.TimeoutExpired:
    timed_out=True;process.kill();process.wait()
   finally:
    if process.poll() is None:process.kill();process.wait()
   if audit.exists():
    result=json.loads(audit.read_text());self.audit+=result['attempts'];self.errors+=result['errors']
   if timed_out:
    self.errors.append(dict(source='acquisition',reason='time_budget_exceeded',seconds=budget_seconds,detail='R2/OpenD acquisition stopped; completed local files remain usable.'))
    if progress:progress('补数据达到时限，保留已下载文件并继续生成明确标注的参考报告')
   elif process.returncode:
    self.errors.append(dict(source='acquisition',reason='worker_failed',exit_code=process.returncode))
   # The child only writes complete archives atomically; reload them after it exits.
   for s in symbols:self.cache.pop(s,None)
 def raw(self,s):
  if s not in self.cache:
   p=self.directory/f'{s}.parquet'
   empty=pd.DataFrame(columns=['start','end','available','day','minute','open','high','low','close','volume'])
   self.cache[s]=normalize_archive(pd.read_parquet(p)) if p.exists() else normalize_archive(read_raw(s) if self.use_archive else empty)
  return self.cache[s]
 def merge(self,s,n,source,now,received_at=None):
  f=pd.concat([self.raw(s),n],ignore_index=True).drop_duplicates('start',keep='last').sort_values('start').reset_index(drop=True);save(f,self.directory/f'{s}.parquet');self.cache[s]=f
  self.audit.append(dict(symbol=s,source=source,received_at=received_at or datetime.now(timezone.utc).isoformat(),loaded_at=datetime.now(timezone.utc).isoformat(),latest_bar_end=str(f.end.max()),rows=len(f)))
  self.checkpoint()
 def refresh(self,symbols,day,cut,futu_host=None,futu_port=11111,progress=None,budget_seconds=30):
  from scripts.r2_client import R2Client
  missing=[s for s in symbols if not covered(self.raw(s),day,cut)]
  self.audit+=[dict(symbol=s,source='local',status='covered',received_at=None) for s in symbols if s not in missing]
  self.checkpoint()
  if not missing:return
  if progress:progress(f'本地不足 {len(missing)} 只，检查 R2（只读）')
  try:
   client=R2Client(timeout=5,deadline=time.monotonic()+budget_seconds)
   begin=str((pd.Timestamp(day)-pd.Timedelta(days=110)).date())
   def get(item):
    s,obj=item;key=obj['key'];p=self.directory/'r2_cache'/key;stamp=p.with_suffix('.etag.json');old=json.loads(stamp.read_text()) if stamp.exists() else {}
    if not p.exists() or old.get('etag')!=obj['etag']:
     p.parent.mkdir(parents=True,exist_ok=True);client.get_object(key,p);write(stamp,{'etag':obj['etag'],'received_at':datetime.now(timezone.utc).isoformat()})
    receipt=json.loads(stamp.read_text()).get('received_at') if stamp.exists() else None
    return s,normalize_archive(pd.read_parquet(p)),key,receipt
   # Load the common archive before listing the larger training-history prefix.
   for prefix in ['us_5m/','model_training_history_v1/parts/']:
    if progress:progress(f'R2 正在列出 {prefix}（每次请求最多 5 秒）')
    inventory=client.list_objects(prefix);wanted=[]
    for obj in inventory:
     key=obj['key'];parts=key.split('/');s=parts[1] if key.startswith('us_5m/') and len(parts)==3 else parts[2] if key.startswith('model_training_history_v1/parts/') and len(parts)>=5 else None
     if s not in missing or not key.endswith('.parquet'):continue
     if key.startswith('us_5m/') and parts[2][:4]!=day[:4]:continue
     if key.startswith('model_training_history_v1/') and not begin[:7]<=parts[3]<=day[:7]:continue
     wanted.append((s,obj))
    if progress:progress(f'R2 找到 {len(wanted)} 个候选文件，最多四路只读下载')
    with ThreadPoolExecutor(max_workers=4) as pool:
     futures=[pool.submit(get,x) for x in wanted]
     for future in as_completed(futures):
      try:s,n,key,receipt=future.result();self.merge(s,n,'R2:'+key,datetime.now(ET),received_at=receipt)
      except Exception as e:self.errors.append(dict(source='R2',reason=type(e).__name__,detail=str(e)[:160]));self.checkpoint()
    missing=[s for s in symbols if not covered(self.raw(s),day,cut)]
    if not missing:break
  except Exception as e:self.errors.append(dict(source='R2',reason=type(e).__name__,detail=str(e)[:160]))
  self.checkpoint()
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
  finally:q.close();self.checkpoint()

if __name__=='__main__':
 if len(sys.argv)!=3 or sys.argv[1]!='--refresh-worker':raise SystemExit('Internal acquisition worker; use recommend.py')
 request=json.loads(Path(sys.argv[2]).read_text());market=MarketData(request.pop('directory'));market._audit_path=request.pop('audit_path')
 try:market.refresh(**request,progress=lambda msg:print(msg,file=sys.stderr,flush=True))
 except Exception as e:market.errors.append(dict(source='acquisition',reason=type(e).__name__,detail=str(e)[:160]));raise
 finally:market.checkpoint()
