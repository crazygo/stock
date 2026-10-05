"""Current group verification only; never changes the preregistered universe."""
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import OUT,write,now

def main():
    import futu as ft
    ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
    try:
        ret,groups=q.get_user_security_group();assert ret==ft.RET_OK and '特别关注' in set(groups.group_name);time.sleep(3.2)
        ret,members=q.get_user_security(group_name='特别关注');assert ret==ft.RET_OK
        snapshot=dict(observed_at=now(),groups=groups.to_dict('records'),snapshots=[dict(group='特别关注',received_at=now(),status='ok',members=members.to_dict('records'))],membership='current_retrospective_not_PIT; fixed 124 training pool unchanged')
        write(OUT/'R06/favorites_snapshot.json',snapshot);print(json.dumps(dict(status='completed',members=len(members),us_stocks=int((members.code.str.startswith('US.')&(members.stock_type=='STOCK')).sum()))),flush=True)
    finally:q.close()

if __name__=='__main__':main()
