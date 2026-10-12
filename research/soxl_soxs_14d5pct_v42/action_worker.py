"""Single killable read-only corporate-action request; never opens a trade context."""
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
import futu as ft

def main():
    p=argparse.ArgumentParser();p.add_argument('--symbol',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert a.symbol in ['SPY','IWM','TLT','SHY','HYG','LQD'];assert not a.output.exists()
    ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
    try:
        start=datetime.now(timezone.utc).isoformat();ret,data=q.get_rehab('US.'+a.symbol);end=datetime.now(timezone.utc).isoformat()
        if ret!=ft.RET_OK:raise RuntimeError(str(data))
        data.to_parquet(a.output,index=False,compression='zstd',compression_level=7)
        print('@@MACRO_ACTION@@ '+json.dumps({'symbol':a.symbol,'requested_at':start,'received_at':end,'rows':len(data),'ok':True}),flush=True)
    finally:q.close()

if __name__=='__main__':main()
