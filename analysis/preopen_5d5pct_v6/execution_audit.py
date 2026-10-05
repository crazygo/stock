"""Free read-only quote capability probe; does not alter subscriptions or signals."""
import json,time
from datetime import datetime
from zoneinfo import ZoneInfo
import futu as ft
from common import OUT,write,now

def main():
    codes=['US.'+s for s in ['ALAB','AMD','MRVL','TER','TXG']]
    ft.SysConfig.enable_proto_encrypt(False);q=ft.OpenQuoteContext(host='127.0.0.1',port=11111);rows=[]
    try:
        ret,data=q.get_market_snapshot(codes);snapshot=[]
        if ret==ft.RET_OK:
            fields=[c for c in ['code','update_time','last_price','volume','suspension','pre_price','pre_timestamp','after_price','after_timestamp'] if c in data]
            snapshot=data[fields].to_dict('records')
            write(OUT/'cache/execution_snapshot_columns.json',list(data.columns))
        snapshot_status='available_current_query_only' if ret==ft.RET_OK else 'unavailable'
        for code in codes:
            ret,book=q.get_order_book(code,num=1)
            item=dict(code=code,received_at=now(),status='available_current_query_only' if ret==ft.RET_OK else 'unavailable',historical_versions=False)
            if ret==ft.RET_OK:
                for key in ['svr_recv_time_bid','svr_recv_time_ask','order_book_type']:
                    if key in book:item[key]=book[key]
                ask=book.get('Ask',[]);bid=book.get('Bid',[])
                if ask and bid and ask[0][0]>0 and bid[0][0]>0:
                    a,b=float(ask[0][0]),float(bid[0][0]);item.update(ask=a,bid=b,ask_volume=ask[0][1],bid_volume=bid[0][1],spread_bps=(a-b)/((a+b)/2)*10000)
                else:item['quality']='missing_or_zero_best_prices'
            else:item['reason']=str(book)[:180]
            rows.append(item);time.sleep(1)
    finally:q.close()
    write(OUT/'execution_audit.json',dict(at=now(),market_time=datetime.now(ZoneInfo('America/New_York')).isoformat(),snapshot_status=snapshot_status,snapshot=snapshot,books=rows,
        subscriptions_changed=False,orders_sent=False,historical_execution_proven=False,note='Current query receipt time does not establish historical signal-time quotes, executable depth, venue coverage or fills. No quote result changes signal denominators. No subscription or paid permission requested.'))
    print(json.dumps(dict(status='completed',books_available=sum(r['status'].startswith('available') for r in rows))),flush=True)
if __name__=='__main__':main()
