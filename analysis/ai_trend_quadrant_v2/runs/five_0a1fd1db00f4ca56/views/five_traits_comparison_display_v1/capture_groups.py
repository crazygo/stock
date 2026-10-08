"""Read watchlist names and nonzero position membership. Never save account IDs or balances."""
from pathlib import Path
from datetime import datetime, timezone
import json
import futu as ft

OUT=Path(__file__).resolve().parent


def main():
    ft.SysConfig.enable_proto_encrypt(False)
    snapshot={'observed_at':datetime.now(timezone.utc).isoformat(), 'source':'OpenD read-only group and position queries',
              'watchlists':{},'holdings':[],'errors':[]}
    q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
    try:
        ret,groups=q.get_user_security_group()
        if ret!=ft.RET_OK:raise RuntimeError('group_names_unavailable')
        names=list(groups.group_name)
        for name in ['全部','特别关注']:
            if name not in names:
                snapshot['errors'].append({'group':name,'reason':'actual_group_absent'});continue
            ret,rows=q.get_user_security(group_name=name)
            if ret!=ft.RET_OK:
                snapshot['errors'].append({'group':name,'reason':'query_failed'});continue
            snapshot['watchlists'][name]=[{'code':str(r.code),'name':str(r.name),'kind':str(r.stock_type),
                                           'listing_date':str(r.listing_date)} for r in rows.itertuples()]
    finally:q.close()
    for firm,label in [(ft.SecurityFirm.FUTUINC,'Moomoo US'),(ft.SecurityFirm.FUTUSECURITIES,'富途证券')]:
        ctx=ft.OpenSecTradeContext(filter_trdmarket=ft.TrdMarket.US,host='127.0.0.1',port=11111,security_firm=firm)
        try:
            ret,accounts=ctx.get_acc_list()
            if ret!=ft.RET_OK:
                snapshot['errors'].append({'group':'holdings','firm':label,'reason':'account_query_failed'});continue
            for a in accounts.itertuples():
                if str(a.trd_env)!='REAL':continue
                ret,rows=ctx.position_list_query(acc_id=int(a.acc_id),trd_env=ft.TrdEnv.REAL)
                if ret!=ft.RET_OK:
                    snapshot['errors'].append({'group':'holdings','firm':label,'reason':'position_query_failed'});continue
                snapshot['holdings'].extend({'code':str(r.code),'name':str(r.stock_name),'account_type':label+' / '+str(a.acc_type)}
                                            for r in rows.itertuples() if float(r.qty)!=0)
        finally:ctx.close()
    (OUT/'group_capture.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'observed_at':snapshot['observed_at'],'watchlists':{k:len(v) for k,v in snapshot['watchlists'].items()},
                      'holdings':len(set(r['code'] for r in snapshot['holdings'])),'errors':snapshot['errors']},ensure_ascii=False))


if __name__=='__main__':main()
