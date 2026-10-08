"""Read-only Local -> R2 -> OpenD corporate-action metadata supplement."""
from pathlib import Path
import json,time,sys
import futu as ft
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scripts.r2_client import R2Client

def main():
    u=json.loads((OUT/'universe.json').read_text());symbols=sorted(s for s,m in u['members'].items() if '特别关注' in m['groups'] and m.get('stock_type')=='STOCK')
    client=R2Client()
    try:remote={o['key'] for o in client.list_objects('corporate_actions/')}
    except Exception as e:remote=set();print('R2 inventory',type(e).__name__,flush=True)
    ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
    try:
        for s in symbols:
            roots=[ROOT/'market_data/model_training_history_v1/corporate_actions',ROOT/'market_data/corporate_actions',OUT/'cache/corporate_actions']
            if any((base/(s+'.parquet')).exists() for base in roots):continue
            p=roots[-1]/(s+'.parquet');p.parent.mkdir(parents=True,exist_ok=True);key='corporate_actions/'+s+'.parquet'
            if key in remote:client.get_object(key,p);print(s,'R2',flush=True);continue
            r,d=q.get_rehab('US.'+s)
            if r==ft.RET_OK:d.to_parquet(p,index=False,compression='zstd',compression_level=7);print(s,len(d),flush=True)
            else:print(s,str(d),flush=True)
            time.sleep(3.2)
        r,d=q.get_stock_basicinfo(ft.Market.US,ft.SecurityType.STOCK,code_list=['US.CBRS','US.LIFE'])
        if r==ft.RET_OK:(OUT/'security_metadata.json').write_text(d.to_json(orient='records',force_ascii=False,indent=2))
    finally:q.close()
if __name__=='__main__':main()
