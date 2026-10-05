"""Read-only per-symbol collection. Atomic files, finite pages and bounded worker."""
import sys,json,time,subprocess,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
START='2024-10-04';END='2026-09-30'
WINDOWS=[('2024-10-04','2025-09-30'),('2025-10-01','2026-09-30')]
METHODS={'capital':'get_capital_flow','options':'get_option_underlying_his_statistic','volatility':'get_option_underlying_his_volatility','short_volume':'get_daily_short_volume'}
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def transient(errors):return any(any(marker in e for marker in ['NN_ProtoRet_TimeOut','网络中断','ByDisConnOrCacnel']) for e in errors)
def worker(symbol):
 import futu as ft,pandas as pd
 ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
 records=[]
 try:
  for family,method in METHODS.items():
   output=ROOT/'cache/history'/family/f'{symbol}.parquet';meta=output.with_suffix('.json');output.parent.mkdir(parents=True,exist_ok=True)
   if meta.exists():
    previous=json.loads(meta.read_text());first_attempt=meta.with_suffix('.first_attempt.json')
    if transient(previous['errors']) and not first_attempt.exists():
     meta.rename(first_attempt)
     if output.exists():output.rename(output.with_suffix('.first_attempt.parquet'))
     rawdir=ROOT/'cache/raw_pages'/family/symbol
     if rawdir.exists():rawdir.rename(rawdir.with_name(symbol+'_first_attempt'))
    else:records.append(previous);continue
   previous_file=output.with_suffix('.first_attempt.parquet');parts=[pd.read_parquet(previous_file)] if previous_file.exists() else [];calls=[];errors=[]
   windows=[(START,END)] if family in ['capital','short_volume'] else WINDOWS
   for w,(begin,end) in enumerate(windows):
    key=None;seen=set()
    for page in range(20):
     kwargs=dict(stock_code='US.'+symbol,period_type='DAY',start=begin,end=end) if family=='capital' else dict(code='US.'+symbol,num=50,next_key=key) if family=='short_volume' else dict(code='US.'+symbol,begin_time=begin,end_time=end,page_req_key=key)
     requested=stamp()
     try:values=getattr(q,method)(**kwargs)
     except Exception as e:errors.append(repr(e));break
     received=stamp();ret=values[0]
     call=dict(window=[begin,end],page=page,ret=int(ret),requested_at=requested,received_at=received)
     if ret!=ft.RET_OK:call['error']=str(values[1])[:500];calls.append(call);errors.append(call['error']);break
     f=values[1];call['rows']=len(f);calls.append(call)
     raw=ROOT/'cache/raw_pages'/family/symbol/f'{w}_{page}.parquet';raw.parent.mkdir(parents=True,exist_ok=True)
     # Some successful responses mix numeric values and literal N/A. Keep the
     # original JSON before numeric coercion, with missing values never zeroed.
     original=raw.with_suffix('.json');original.write_text(f.to_json(orient='records',force_ascii=False)+'\n');call['original_json_sha256']=sha(original)
     text_columns={'code','name','time','source_day','last_valid_time','capital_flow_item_time','timestamp_str'}
     attrs=dict(f.attrs);f=f.copy()
     for column in f.columns:
      if column not in text_columns:f[column]=pd.to_numeric(f[column],errors='coerce')
     f.attrs=attrs;f.to_parquet(raw,index=False,compression='zstd',compression_level=7);call['raw_sha256']=sha(raw)
     parts.append(f)
     if family=='capital':key=None
     elif family=='short_volume':key=f.attrs.get('next_key');key=None if key=='-1' else key
     else:key=values[2]
     datecol='capital_flow_item_time' if family=='capital' else 'timestamp_str' if family=='short_volume' else 'time'
     time.sleep(1.05)
     if key is None or not len(f) or (family=='short_volume' and f[datecol].astype(str).min()[:10]<=START):break
     marker=repr(key)
     if marker in seen:errors.append('repeated pagination key');break
     seen.add(marker)
    else:errors.append('finite pagination limit reached')
   if parts:
    frame=pd.concat(parts,ignore_index=True);datecol='capital_flow_item_time' if family=='capital' else 'timestamp_str' if family=='short_volume' else 'time'
    frame['source_day']=frame[datecol].astype(str).str[:10];frame=frame[(frame.source_day>=START)&(frame.source_day<=END)].copy();frame=frame.sort_values('source_day').drop_duplicates('source_day',keep='first')
    tmp=output.with_suffix('.writing.parquet');frame.to_parquet(tmp,index=False,compression='zstd',compression_level=7);tmp.replace(output)
    rows=len(frame);first=frame.source_day.min() if rows else None;last=frame.source_day.max() if rows else None
   else:rows=0;first=last=None
   data=dict(symbol=symbol,family=family,method=method,requested=[START,END],at=stamp(),status='available_partial' if rows else 'unavailable',rows=rows,first=first,last=last,file=str(output.relative_to(ROOT)) if rows else None,sha256=sha(output) if output.exists() else None,calls=calls,errors=errors,availability='source period day; historical publication/receipt vintages not supplied; use preregistered 2/5-session lags only in exposed development')
   first_attempt=meta.with_suffix('.first_attempt.json')
   if first_attempt.exists():
    data['previous_attempt']=json.loads(first_attempt.read_text())
    data['previous_attempt']['archived_file']=str(previous_file.relative_to(ROOT)) if previous_file.exists() else None
    data['previous_attempt']['archived_pages']=str((ROOT/'cache/raw_pages'/family/(symbol+'_first_attempt')).relative_to(ROOT))
   tmp=meta.with_suffix('.writing.json');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(meta);records.append(data)
 finally:q.close()
 (ROOT/'cache/history'/f'{symbol}.json').write_text(json.dumps(dict(symbol=symbol,at=stamp(),records=records),ensure_ascii=False,indent=2)+'\n')

def main():
 if '--worker' in sys.argv:worker(sys.argv[-1]);return
 old=ROOT.parent.parent/'preopen_ranked_policy_v5/raw';symbols=sorted(p.stem for p in old.glob('*.parquet'))
 # Priority only changes acquisition order, never model membership.
 priority=['ALAB','AMD','MRVL','TER','TXG'];symbols=priority+[s for s in symbols if s not in priority]
 rows=[]
 for i,symbol in enumerate(symbols,1):
  meta=ROOT/'cache/history'/f'{symbol}.json'
  if meta.exists():
   prior=json.loads(meta.read_text());retry=any(transient(x['errors']) and not (ROOT/'cache/history'/x['family']/f'{symbol}.first_attempt.json').exists() for x in prior.get('records',[]))
   if retry:meta.rename(meta.with_suffix('.first_attempt.json'))
  if not meta.exists():
   try:
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',symbol],capture_output=True,text=True,timeout=240)
    with (ROOT/'logs'/f'acquire_{symbol}.log').open('a') as stream:stream.write('\nAttempt '+stamp()+'\n'+result.stdout+result.stderr)
   except subprocess.TimeoutExpired:
    with (ROOT/'logs'/f'acquire_{symbol}.log').open('a') as stream:stream.write('\nWorker exceeded finite 240s budget; completed family files retained.\n')
  row=json.loads(meta.read_text()) if meta.exists() else dict(symbol=symbol,status='worker_timeout_or_failure',at=stamp())
  rows.append(row);print(json.dumps(dict(completed=i,total=len(symbols),symbol=symbol,families=[(x['family'],x['rows']) for x in row.get('records',[])])),flush=True)
  (ROOT/'acquisition.json').write_text(json.dumps(dict(at=stamp(),status='running',fixed_symbols=symbols,records=rows),ensure_ascii=False,indent=2)+'\n')
 (ROOT/'acquisition.json').write_text(json.dumps(dict(at=stamp(),status='completed',fixed_symbols=symbols,records=rows),ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
