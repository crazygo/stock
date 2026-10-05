"""Temporary own-connection capability subscription; always canceled after minimum hold."""
import sys,json,time,datetime,hashlib
from pathlib import Path
import futu as ft
ROOT=Path(__file__).resolve().parent
ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
report={'requested_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'symbol':'US.AMD','scope':'this connection only','historical_training_eligible':False}
started=time.monotonic();subscribed=False
try:
 r,d=q.subscribe(['US.AMD'],[ft.SubType.ORDER_BOOK,ft.SubType.TICKER],is_first_push=False,subscribe_push=False);report['subscribe_ret']=int(r);report['subscribe_message']=str(d)[:200];subscribed=r==ft.RET_OK
 if subscribed:
  for name,kwargs in [('get_order_book',dict(code='US.AMD',num=5)),('get_rt_ticker',dict(code='US.AMD',num=20))]:
   r,d=getattr(q,name)(**kwargs);row={'ret':int(r),'received_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
   file=ROOT/'cache/probes'/f'subscribed_{name}.json'
   if r==ft.RET_OK:
    if hasattr(d,'to_dict'):payload=d.to_dict('records');row.update(rows=len(d),columns=list(d),time_range=[str(d.time.min()),str(d.time.max())] if 'time' in d else None)
    else:payload=d;row.update(keys=list(d),bid_levels=len(d.get('Bid',[])),ask_levels=len(d.get('Ask',[])),server_time_bid=d.get('svr_recv_time_bid'),server_time_ask=d.get('svr_recv_time_ask'))
    file.write_text(json.dumps(payload,ensure_ascii=False,default=str));row.update(file=str(file.relative_to(ROOT)),sha256=hashlib.sha256(file.read_bytes()).hexdigest())
   else:row['error']=str(d)[:300]
   report[name]=row
finally:
 if subscribed:
  time.sleep(max(0.,62-(time.monotonic()-started)))
  r,d=q.unsubscribe(['US.AMD'],[ft.SubType.ORDER_BOOK,ft.SubType.TICKER]);report.update(unsubscribe_ret=int(r),unsubscribe_message=str(d)[:200])
 q.close();report['completed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (ROOT/'subscription_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False),flush=True)
