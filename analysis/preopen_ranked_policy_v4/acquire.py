"""Coverage-driven Local -> R2 -> OpenD. Quote context only, no cloud writes."""
from __future__ import annotations
import argparse,fcntl,json,re,time
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from common import *
from scripts.r2_client import R2Client

def universe(q,ft):
 old=json.loads((OUT/'universe.json' if (OUT/'universe.json').exists() else ROOT/'analysis/preopen_intraday_v2/universe.json').read_text())
 ret,groups=q.get_user_security_group()
 if ret!=ft.RET_OK:raise RuntimeError('watchlist groups unavailable: '+str(groups)[:100])
 names=set(groups.group_name)
 if '特别关注' not in names:raise RuntimeError('当前分组没有特别关注，不能假定 Favorites 别名')
 records={}
 for name in ['特别关注','全部','ETF']:
  ret,f=q.get_user_security(name)
  if ret!=ft.RET_OK:raise RuntimeError(name+' query failed: '+str(f)[:100])
  records[name]=f.to_dict('records');time.sleep(.4)
 # ETF files retain their own source timestamp; no pretend PIT or fresh issuer download.
 for row in old['members'].values():row['groups']=[g for g in row['groups'] if g not in ['特别关注','futud自选','自选ETF']]
 for name,group in [('特别关注','特别关注'),('全部','futud自选'),('ETF','自选ETF')]:
  for r in records[name]:
   if not r['code'].startswith('US.') or r['stock_type'] not in ['STOCK','ETF']:continue
   s=r['code'][3:]
   if not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,9}',s):continue
   m=old['members'].setdefault(s,dict(symbol=s,groups=[]));m.update(name=r['name'],stock_type=r['stock_type'])
   if group not in m['groups']:m['groups'].append(group)
 old['watchlist_observed_at']=datetime.now(timezone.utc).isoformat();old['favorite_group']='特别关注';old['favorites']=records['特别关注'];old['all_watchlist']=records['全部'];old['etf_watchlist']=records['ETF']
 old['members']={s:m for s,m in old['members'].items() if m['groups']}
 write(OUT/'universe.json',old);return old

def normalize(f):
 end=pd.to_datetime(f.time_key).dt.tz_localize('America/New_York',ambiguous='raise',nonexistent='raise').dt.tz_localize(None)
 out=pd.DataFrame(dict(start=end-pd.Timedelta(minutes=5),end=end,available=end+pd.Timedelta(seconds=1)))
 for k in ['open','high','low','close','volume']:out[k]=pd.to_numeric(f[k],errors='raise').to_numpy(float)
 out['day']=out.start.dt.strftime('%Y-%m-%d');out['minute']=out.start.dt.hour*60+out.start.dt.minute
 prices=out[['open','high','low','close']].to_numpy();valid=np.isfinite(prices).all(1)&(prices>0).all(1)&np.isfinite(out.volume)&(out.volume>=0)&(out.high>=prices.max(1))&(out.low<=prices.min(1))
 return out[valid].drop_duplicates('start').sort_values('start').reset_index(drop=True)

def covered(f,day,cutoff=None):
 if day not in SESSIONS:return True
 if f is None or f.empty:return False
 end=570+SESSIONS[day]['duration_minutes'];end=min(end,cutoff) if cutoff is not None else end
 g=f[(f.day==day)&(f.minute>=570)&(f.minute<end)]
 return np.array_equal(g.minute.to_numpy(),np.arange(570,end,5))

def refresh(symbols=None,end=None,live=False,status=None):
 import futu as ft
 from data import read_raw
 now=datetime.now(ET);day=end or now.date().isoformat();days=[d for d in SESSIONS if d<=day]
 if day not in SESSIONS or now.hour*60+now.minute<570:day=days[-1] if day not in SESSIONS else days[-2]
 current=(live and day==now.date().isoformat() and 570<=now.hour*60+now.minute<570+SESSIONS[day]['duration_minutes'])
 cutoff=(now.hour*60+now.minute)//5*5 if current else None
 ft.SysConfig.enable_proto_encrypt(False)
 with open('/tmp/stock_futu_acquisition.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
  try:
   u=universe(q,ft)
   if symbols is None:
    coverage=json.loads((OUT/'coverage.json').read_text()) if (OUT/'coverage.json').exists() else None
    available=set(p.name for p in (ROOT/'market_data/model_training_history_v1/parts').iterdir())|set(p.stem for p in (ROOT/'analysis/preopen_stock_cycle_v3/raw').glob('*.parquet'))
    symbols=[c['symbol'] for c in coverage['coverage']] if coverage else sorted((set(u['members'])|{'QQQ','SOXX','IGV'})&available)
   missing=[];receipt=datetime.now(timezone.utc).isoformat();attempts=[]
   for s in symbols:
    f=read_raw(s)
    if not covered(f,day,cutoff):missing.append((s,f))
    else:attempts.append(dict(symbol=s,source='local',session=day,status='covered',received_at=None))
   client=R2Client();inventory={};r2_error=None
   if missing:
    try:inventory={x['key']:x for x in client.list_objects('us_5m/')}
    except Exception as e:r2_error=type(e).__name__
   for i,(s,f) in enumerate(missing):
    if status:status.update(phase='补齐行情',done=i,total=len(missing),symbol=s)
    if r2_error is None:
     for key in inventory:
      if not key.startswith(f'us_5m/{s}/'):continue
      p=OUT/'raw/r2'/key;p.parent.mkdir(parents=True,exist_ok=True)
      if not p.exists():client.get_object(key,p)
      rf=pd.read_parquet(p)
      if {'start_at_et','end_at_et','price_basis'}.issubset(rf) and set(rf.price_basis.dropna())=={'NONE'}:
       n=rf[['open','high','low','close','volume']].copy();n['start']=pd.to_datetime(rf.start_at_et,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None);n['end']=pd.to_datetime(rf.end_at_et,utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None);n['available']=n.end+pd.Timedelta(seconds=1);n['day']=n.start.dt.strftime('%Y-%m-%d');n['minute']=n.start.dt.hour*60+n.start.dt.minute
       f=pd.concat([f,n],ignore_index=True).drop_duplicates('start',keep='last').sort_values('start')
    source='R2'
    if current and not covered(f,day,cutoff):
     # Live subscription avoids repeatedly downloading years of history.
     ret,_=q.subscribe(['US.'+s],[ft.SubType.K_5M],subscribe_push=False,extended_time=True,session=ft.Session.ALL)
     if ret==ft.RET_OK:
      ret,recent=q.get_cur_kline('US.'+s,1000,ktype=ft.KLType.K_5M,autype=ft.AuType.NONE)
      if ret==ft.RET_OK:
       n=normalize(recent);f=pd.concat([f,n],ignore_index=True).drop_duplicates('start',keep='last').sort_values('start').reset_index(drop=True);source='OpenD_current'
     else:attempts.append(dict(symbol=s,source='subscription',status='failed',error='K_5M subscription unavailable',session=day))
    if not covered(f,day,cutoff):
     source='OpenD';start=max('2026-09-24',str(f.end.max().date()) if len(f) else '2026-08-01');pages=[];key=None
     while True:
      ret,raw,key=q.request_history_kline('US.'+s,start=start,end=day,ktype=ft.KLType.K_5M,autype=ft.AuType.NONE,fields=ft.KL_FIELD.ALL,max_count=1000,page_req_key=key,extended_time=True,session=ft.Session.ALL)
      if ret!=ft.RET_OK:attempts.append(dict(symbol=s,source=source,status='failed',error=str(raw)[:200],session=day));break
      pages.append(raw)
      if not key:break
      time.sleep(.25)
     if ret==ft.RET_OK and pages:
      n=normalize(pd.concat(pages,ignore_index=True));f=pd.concat([f,n],ignore_index=True).drop_duplicates('start',keep='last').sort_values('start').reset_index(drop=True)
     time.sleep(1.2)
    if len(f):save(f,OUT/'raw'/f'{s}.parquet')
    attempts.append(dict(symbol=s,source=source,session=day,status='covered' if covered(f,day,cutoff) else 'incomplete',last_bar=str(f.end.max()) if len(f) else None,received_at=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(attempts[-1]),flush=True)
   report=dict(checked_at=datetime.now(timezone.utc).isoformat(),session=day,cutoff=cutoff,live=current,attempts=attempts,r2_error=r2_error,requested=len(symbols),missing_before=len(missing))
   write(OUT/'acquisition.json',report)
   return report
  finally:q.close()

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--symbols',nargs='+');ap.add_argument('--end');a=ap.parse_args();print(json.dumps(clean(refresh(a.symbols,a.end)),ensure_ascii=False))
