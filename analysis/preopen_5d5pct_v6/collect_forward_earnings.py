"""Capture free forecasts now for future as-of research; never rewrite history."""
import argparse,json,time
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
import pandas as pd
import futu as ft
from common import OUT,write,sha,now

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--weeks',type=int,default=8);a=ap.parse_args()
    assert 1<=a.weeks<=12
    started=now();stamp=datetime.now(ZoneInfo('UTC')).strftime('%Y%m%dT%H%M%SZ');folder=OUT/'cache/forward_earnings'/stamp
    first=datetime.now(ZoneInfo('America/New_York')).date()+timedelta(days=1)
    universe={r['symbol'] for r in json.loads((OUT/'universe.json').read_text())['members']};summaries=[];errors=0
    ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
    try:
        for n in range(a.weeks):
            begin=first+timedelta(days=7*n);end=begin+timedelta(days=6);ret,data=q.get_earnings_calendar(market=ft.Market.US,begin_date=str(begin),end_date=str(end));received=now()
            records=data.to_dict('records') if isinstance(data,pd.DataFrame) else []
            file=folder/f'{begin}.json';write(file,dict(request_started_at=started,received_at=received,range=[str(begin),str(end)],records=records,source='Futu get_earnings_calendar',status='ok' if ret==ft.RET_OK else 'unavailable',reason=None if ret==ft.RET_OK else str(data)[:180]))
            summaries.append(dict(start=str(begin),end=str(end),received_at=received,rows=len(records),universe_rows=sum(str(r.get('security','')).removeprefix('US.') in universe for r in records),sha256=sha(file),status='ok' if ret==ft.RET_OK else 'unavailable'))
            errors=0 if ret==ft.RET_OK else errors+1
            if errors>=3:break
            time.sleep(3.2)
    finally:q.close()
    write(OUT/'forward_earnings_capture.json',dict(at=now(),snapshot=stamp,weeks=summaries,historical_versions_reconstructed=False,
        note='For this collector observed_available_at is actual receipt; earliest external availability is unknown. Scheduled earnings timestamps are not confirmed release times. One snapshot is not a complete revision series; no new model/admission follows. Preserve zero/NaN actuals without inventing releases.'))
    print(json.dumps(dict(snapshot=stamp,rows=sum(s['rows'] for s in summaries),status='captured_for_future_research')),flush=True)
if __name__=='__main__':main()
