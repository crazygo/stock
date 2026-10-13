"""Check one discovered SPY low disagreement against finer and daily source bars."""
from pathlib import Path
import datetime, hashlib, json, sys, time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v76'


def main():
    import futu as ft
    from scripts.r2_client import R2Client
    out=CACHE/'spy_low_probe';assert not out.exists();out.mkdir()
    checked=[]
    for interval in ['us_1m','us_daily']:
        key=f'{interval}/SPY/2023.parquet';local=ROOT/'market_data'/key
        assert not local.exists(), 'Inspect existing local data first'
        found=R2Client(timeout=5,deadline=time.monotonic()+12).head_object(key)
        assert found is None, 'Inspect remote data before issuing diagnostic'
        checked.append(dict(local=str(local.relative_to(ROOT)),r2_key=key,found=False))
    ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
    records=[]
    try:
        for name,kind in [('minute',ft.KLType.K_1M),('daily',ft.KLType.K_DAY)]:
            time.sleep(1.2);before=datetime.datetime.now(datetime.timezone.utc).isoformat()
            ret,data,key=q.request_history_kline('US.SPY',start='2023-10-30',end='2023-10-30',ktype=kind,autype=ft.AuType.NONE,max_count=1000,session=ft.Session.RTH)
            assert ret==ft.RET_OK and key is None
            path=out/(name+'.parquet');data.to_parquet(path,index=False,compression='zstd',compression_level=7)
            records.append(dict(kind=name,rows=len(data),requested_at=before,received_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        (out/'receipt.json').write_text(json.dumps(dict(source_checks=checked,captures=records),indent=2)+'\n')
    finally:q.close()


if __name__=='__main__':main()
