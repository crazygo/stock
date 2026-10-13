"""Bounded Local/R2/OpenD feasibility sample; not a training dataset acceptance."""
from pathlib import Path
import sys,json,time,hashlib,datetime,subprocess
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
CACHE=ROOT/'.cache/soxl_soxs_14d5pct_v75'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def worker():
    import futu as ft
    ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
    result={'symbol':'SOXL','start':'2023-04-03','end':'2023-04-05','autype':'NONE','session':'ALL','pages':[]}
    try:
        ret,quota=q.get_history_kl_quota(get_detail=False)
        if ret!=ft.RET_OK:raise RuntimeError('Quota read failed')
        result['quota']={'used':int(quota[0]),'remaining':int(quota[1])}
        if quota[1]<1:raise RuntimeError('No new symbol quota available')
        key=None
        for page in range(2):
            time.sleep(1.2 if page==0 else .3);before=datetime.datetime.now(datetime.timezone.utc).isoformat()
            ret,raw,key=q.request_history_kline('US.SOXL',start=result['start'],end=result['end'],ktype=ft.KLType.K_5M,autype=ft.AuType.NONE,max_count=1000,page_req_key=key,extended_time=True,session=ft.Session.ALL)
            after=datetime.datetime.now(datetime.timezone.utc).isoformat()
            if ret!=ft.RET_OK:result.update(ok=False,error_type='ProviderError',error_message=str(raw)[:300]);break
            f=CACHE/'older_probe'/f'SOXL_20230403_20230405_{page}.parquet';f.parent.mkdir(parents=True,exist_ok=True)
            assert not f.exists();raw.to_parquet(f,index=False,compression='zstd',compression_level=7)
            result['pages'].append({'path':str(f.relative_to(ROOT)),'sha256':sha(f),'rows':len(raw),'first':str(raw.time_key.min()) if len(raw) else None,'last':str(raw.time_key.max()) if len(raw) else None,'requested_at':before,'received_at':after})
            if key is None:result.update(ok=True,pagination_complete=True);break
        if key is not None:result.update(ok=False,pagination_complete=False)
        (CACHE/'older_probe/opend_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:q.close()


def main():
    import pandas as pd
    from scripts.r2_client import R2Client
    inv=json.loads((CACHE/'older_history_inventory.json').read_text());rows=[r for r in inv['r2'] if r['prefix'].endswith('/SOXL/')];assert len(rows)==3 and all('objects' in r for r in rows)
    local=[r for r in inv['local'] if r['symbol']=='SOXL'];assert all(r.get('first','9999')>'2023-04-05' for r in local)
    remote=[]
    for obj in [o for row in rows for o in row['objects']]:
        f=CACHE/'older_probe/r2'/Path(obj['key']);assert not f.exists()
        R2Client(timeout=5,deadline=time.monotonic()+15).get_object(obj['key'],f)
        data=pd.read_parquet(f);remote.append({'key':obj['key'],'path':str(f.relative_to(ROOT)),'sha256':sha(f),'first':str(data.time_key.min()),'last':str(data.time_key.max())})
        assert str(data.time_key.min())>'2023-04-05'
    command=[sys.executable,'-W','error','-W','default::DeprecationWarning',str(Path(__file__)),'--worker']
    log=CACHE/'older_probe/worker.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as f:cp=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=45)
    assert cp.returncode==0,'Bounded worker failed; inspect local log'
    probe=json.loads((CACHE/'older_probe/opend_receipt.json').read_text())
    regular={}
    for entry in probe['pages']:
        data=pd.read_parquet(ROOT/entry['path']);t=pd.to_datetime(data.time_key);mask=t.dt.strftime('%H:%M').between('09:30','15:55')
        for day,count in t[mask].dt.strftime('%Y-%m-%d').value_counts().items():regular[day]=regular.get(day,0)+int(count)
    receipt={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'local_inventory_sha256':sha(CACHE/'older_history_inventory.json'),'R2_checked_contents':remote,'OpenD':probe,'observed_regular_start_bars_by_date':regular,'complete_training_dataset':False,'new_strategy_or_capital_evaluation':False,'R2_uploads':0,'source_sha256':sha(Path(__file__))}
    (CACHE/'older_probe_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt,flush=True)


if __name__=='__main__':
    if '--worker' in sys.argv:worker()
    else:main()
