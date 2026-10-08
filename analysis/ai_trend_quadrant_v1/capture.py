"""Read current watchlists only. No account, watchlist mutation or trading calls."""
import json
from datetime import datetime, timezone
from pathlib import Path
import futu as ft

OUT=Path(__file__).resolve().parent

def main():
    ft.SysConfig.enable_proto_encrypt(False)
    q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
    d={'observed_at':datetime.now(timezone.utc).isoformat(),'source':'OpenD get_user_security_group/get_user_security','groups':{},'errors':[]}
    try:
        ret,g=q.get_user_security_group()
        if ret!=ft.RET_OK:raise RuntimeError('group query failed')
        d['group_names']=[str(x) for x in g.group_name]
        for name in ['全部','特别关注','ETF']:
            if name not in d['group_names']:
                d['errors'].append({'group':name,'error':'actual group absent'});continue
            ret,f=q.get_user_security(group_name=name)
            if ret!=ft.RET_OK:raise RuntimeError('security query failed: '+name)
            d['groups'][name]=[{'code':str(r.code),'name':str(r.name),'kind':str(r.stock_type),'listing_date':str(r.listing_date)} for r in f.itertuples()]
    finally:q.close()
    (OUT/'watchlist.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'observed_at':d['observed_at'],'counts':{k:len(v) for k,v in d['groups'].items()},'errors':d['errors']},ensure_ascii=False))

if __name__=='__main__':main()
