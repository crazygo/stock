"""Diagnose four fixed cross-provider residuals after the bulk worker finishes."""
from pathlib import Path
import datetime, hashlib, json, time
import futu as ft
ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v76'
CASES=[('SOXL','2024-09-06'),('SOXS','2023-04-28'),('SOXX','2023-08-02'),('QQQ','2023-04-03')]


def main():
    status=json.loads((CACHE/'acquisition_status.json').read_text())
    assert len(status)==9 and all(r['complete'] for r in status)
    out=CACHE/'daily_diagnostics';assert not out.exists();out.mkdir()
    ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
    receipts=[]
    try:
        for symbol,day in CASES:
            time.sleep(1.2);started=datetime.datetime.now(datetime.timezone.utc).isoformat()
            ret,data,key=q.request_history_kline('US.'+symbol,start=day,end=day,ktype=ft.KLType.K_DAY,autype=ft.AuType.NONE,max_count=1000,session=ft.Session.RTH)
            assert ret==ft.RET_OK and key is None and len(data)==1
            path=out/f'{symbol}_{day}.parquet';data.to_parquet(path,index=False,compression='zstd',compression_level=7)
            receipts.append(dict(symbol=symbol,date=day,requested_at=started,received_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (out/'receipt.json').write_text(json.dumps(receipts,indent=2)+'\n')
    finally:q.close()


if __name__=='__main__':main()
