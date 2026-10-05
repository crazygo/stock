"""Reproduce the two excluded supplementary probes, including mixed N/A."""
import sys,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent))
from common import write,now,sha

def main():
    import futu as ft,pandas as pd
    if (P/'extra_probes.json').exists() and '--refresh' not in sys.argv:
        print('Existing finite extra probe evidence preserved.');return
    ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111);results={}
    try:
        for method,args in [('get_macro_indicator_history',dict(indicator_id=1003000008,max_count=100)),('get_option_volatility',dict(code='US.AMD'))]:
            ret,data=getattr(q,method)(**args);r=dict(ret=int(ret),received_at=now())
            if ret==ft.RET_OK and isinstance(data,pd.DataFrame):
                path=P/'cache/probes'/f'{method}_extra.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(data.to_json(orient='records',force_ascii=False)+'\n');r.update(rows=len(data),columns=list(data),file=str(path.relative_to(P)),sha256=sha(path))
            else:r['error']=str(data)[:500]
            results[method]=r
    finally:q.close()
    write(P/'extra_probes.json',results)

if __name__=='__main__':main()
