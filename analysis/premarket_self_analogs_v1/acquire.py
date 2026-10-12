"""Read-only capture: Local -> R2 -> OpenD. No R2 upload or trading calls."""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from zoneinfo import ZoneInfo
import json,hashlib,sys,time,subprocess,argparse,concurrent.futures
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
START='2025-10-10'; END='2026-10-08'
RUN='capture_20261008T124943Z'
DEST=ROOT/'.cache/premarket_self_analogs_v1'/RUN
MARK='@@RESULT@@'

def worker(code):
 import futu as ft
 ft.SysConfig.enable_proto_encrypt(False)
 q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
 try:
  chunks=[]; key=None; calls=0
  while True:
   time.sleep(.7)
   r,d,key=q.request_history_kline(code,start=START,end=END,ktype=ft.KLType.K_60M,autype=ft.AuType.QFQ,session=ft.Session.ALL,max_count=1000,page_req_key=key)
   calls+=1
   if r!=ft.RET_OK: raise RuntimeError(str(d))
   chunks.append(d)
   if key is None: break
   if calls>20: raise RuntimeError('pagination limit')
  frame=pd.concat(chunks,ignore_index=True).drop_duplicates('time_key',keep='last').sort_values('time_key')
  p=DEST/(code+'.parquet'); p.parent.mkdir(parents=True,exist_ok=True)
  tmp=p.with_suffix('.tmp.parquet');frame.to_parquet(tmp,compression='zstd',compression_level=7);tmp.replace(p)
  print(MARK+json.dumps({'code':code,'path':str(p.relative_to(ROOT)),'source':'opend_qfq_full_series','rows':len(frame),'first':str(frame.time_key.min()),'last':str(frame.time_key.max()),'requests':calls,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}),flush=True)
 finally:q.close()

def local_r2(code):
 from scripts.r2_client import R2Client
 sym=code[3:]; attempts=[]; parts=[]
 for year in [2025,2026]:
  p=ROOT/'market_data/us_60m'/sym/f'{year}.parquet'
  d=pd.read_parquet(p) if p.exists() else None
  hit=d is not None and len(d) and str(d.time_key.min())[:10] <= (START if year==2025 else '2026-01-02') and str(d.time_key.max())[:10] >= ('2025-12-31' if year==2025 else END)
  attempts.append({'stage':'local','year':year,'status':'complete' if hit else 'incomplete' if d is not None else 'miss','rows':len(d) if d is not None else 0})
  if not hit:
   key=f'market_data/us_60m/{sym}/{year}.parquet'; dest=DEST/'r2'/sym/f'{year}.parquet'
   try:
    c=R2Client(timeout=5);head=c.head_object(key)
    if head:
     dest.parent.mkdir(parents=True,exist_ok=True);c.get_object(key,dest);r=pd.read_parquet(dest)
     d=r;attempts.append({'stage':'r2','year':year,'status':'downloaded','rows':len(r),'last':str(r.time_key.max())})
    else:attempts.append({'stage':'r2','year':year,'status':'miss'})
   except Exception as e:attempts.append({'stage':'r2','year':year,'status':'unavailable','error_type':type(e).__name__})
  if d is not None:parts.append(d)
 complete=len(parts)==2 and str(parts[0].time_key.min())[:10]<=START and str(parts[1].time_key.max())[:10]>=END
 return {'code':code,'attempts':attempts,'complete':bool(complete),'parts':parts}

def main():
 DEST.mkdir(parents=True,exist_ok=True)
 positions=json.loads((OUT/'positions.json').read_text())['positions']; codes=[p['code'] for p in positions if p['code'].startswith('US.')]
 manifest={'run_id':RUN,'requested_start':START,'requested_end':END,'created_at':datetime.now(timezone.utc).isoformat(),'autype':'QFQ','session':'ALL','r2_uploaded':False,'datasets':[],'attempts':[]}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  for item in ex.map(local_r2,codes):
   manifest['attempts'].append({k:v for k,v in item.items() if k not in ['parts','complete']})
   print('LAYER',item['code'],json.dumps(item['attempts']),flush=True)
   if item['complete']:
    f=pd.concat(item['parts']).drop_duplicates('time_key',keep='last').sort_values('time_key');p=DEST/(item['code']+'.parquet');f.to_parquet(p,compression='zstd',compression_level=7)
    manifest['datasets'].append({'code':item['code'],'path':str(p.relative_to(ROOT)),'source':'local_r2','rows':len(f),'first':str(f.time_key.min()),'last':str(f.time_key.max()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 for code in codes:
  if any(d['code']==code for d in manifest['datasets']):continue
  try:
   p=subprocess.run([sys.executable,str(Path(__file__)),'--worker',code],cwd=ROOT,capture_output=True,text=True,timeout=65)
   line=next((s[len(MARK):] for s in p.stdout.splitlines() if s.startswith(MARK)),None)
   if not line:raise RuntimeError('worker failed: '+p.stderr[-300:])
   item=json.loads(line);manifest['datasets'].append(item)
   print('CAPTURE',json.dumps(item),flush=True)
  except Exception as e:
   manifest['attempts'].append({'code':code,'stage':'opend','status':'failed','error_type':type(e).__name__,'error':str(e)[:350]});print('FAILED',code,type(e).__name__,flush=True)
  (OUT/'acquisition.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 import futu as ft
 ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
 try:
  r,d=q.request_trading_days(market=ft.Market.US,start=START,end='2026-10-15')
  if r==ft.RET_OK:
   (OUT/'trading_days.json').write_text(json.dumps({'source':'OpenD_exchange_trading_days','received_at':datetime.now(timezone.utc).isoformat(),'days':d.to_dict('records') if hasattr(d,'to_dict') else d},ensure_ascii=False,indent=2,default=str))
   print('CALENDAR',len(d),d.head(2).to_dict('records') if hasattr(d,'head') else d[:2],flush=True)
  r,d=q.get_stock_basicinfo(ft.Market.US,code_list=codes)
  if r==ft.RET_OK:(OUT/'basicinfo.json').write_text(json.dumps(d.to_dict('records'),ensure_ascii=False,indent=2,default=str))
  rehabs={}
  for code in codes:
   time.sleep(.35);r,d=q.get_rehab(code)
   rehabs[code]=d.to_dict('records') if r==ft.RET_OK else {'error':str(d)}
  (OUT/'rehabs.json').write_text(json.dumps(rehabs,ensure_ascii=False,indent=2,default=str))
 finally:q.close()
 manifest['finished_at']=datetime.now(timezone.utc).isoformat();manifest['status']='complete' if len(manifest['datasets'])==len(codes) else 'partial'
 (OUT/'acquisition.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print('DONE',manifest['status'],len(manifest['datasets']),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--worker');args=a.parse_args()
 if args.worker:worker(args.worker)
 else:main()
