"""Finish missing corporate-action checks; no data uploads."""
import sys,json,time
from pathlib import Path
import pandas as pd
import build as b
sys.path.insert(0,str(b.ROOT));from scripts.r2_client import R2Client

def main():
 client=R2Client();inventory={o['key'] for o in client.list_objects('corporate_actions/')};import futu as ft
 ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111);checks=[]
 try:
  for s in ['RGTI','QBTS','SOXX','IGV']:
   roots=[b.legacy.HISTORY/'corporate_actions',b.ROOT/'market_data/corporate_actions',b.legacy.OUT/'cache/corporate_actions'];p=next((r/(s+'.parquet') for r in roots if (r/(s+'.parquet')).exists()),roots[-1]/(s+'.parquet'));source='local'
   if not p.exists():
    p.parent.mkdir(parents=True,exist_ok=True);key=f'corporate_actions/{s}.parquet'
    if key in inventory:client.get_object(key,p);source='R2'
    else:
     ret,f=q.get_rehab('US.'+s)
     if ret!=ft.RET_OK:raise RuntimeError(s+': '+str(f))
     f.to_parquet(p,index=False,compression='zstd',compression_level=7);source='OpenD';time.sleep(3.2)
   f=pd.read_parquet(p);events=f[f.ex_div_date.astype(str).str[:10].between(b.START,b.END)] if 'ex_div_date' in f else f
   checks.append(dict(symbol=s,source=source,path=str(p.relative_to(b.ROOT)),sha256=b.sha(p),events_in_window=events.astype(object).where(pd.notna(events),None).to_dict('records')))
   print(s,'events in window',len(events),flush=True)
 finally:q.close()
 (b.OUT/'supplement_action_audit.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2,default=str))
 if any(c['source']!='local' and c['events_in_window'] for c in checks):print('ACTION_REBUILD_REQUIRED',flush=True)
if __name__=='__main__':main()
