"""Temporary free-permission book probe on this new connection only."""
import json,time
import futu as ft
from common import OUT,write,now

def main():
    codes=['US.'+s for s in ['ALAB','AMD','MRVL','TER','TXG']];ft.SysConfig.enable_proto_encrypt(False)
    q=ft.OpenQuoteContext(host='127.0.0.1',port=11111);started=time.monotonic();rows=[];unsubscribe=None
    try:
        ret,before=q.query_subscription(is_all_conn=False)
        ret,msg=q.subscribe(codes,[ft.SubType.ORDER_BOOK],is_first_push=False,subscribe_push=False)
        subscribed=ret==ft.RET_OK
        if subscribed:
            for code in codes:
                ret,book=q.get_order_book(code,num=1);row=dict(code=code,received_at=now(),status='available' if ret==ft.RET_OK else 'unavailable')
                if ret==ft.RET_OK:
                    ask=book.get('Ask',[]);bid=book.get('Bid',[])
                    row['provider_times']={k:v for k,v in book.items() if 'time'in k}
                    if ask and bid and ask[0][0]>0 and bid[0][0]>0:
                        a,b=float(ask[0][0]),float(bid[0][0]);row.update(ask=a,bid=b,ask_volume=ask[0][1],bid_volume=bid[0][1],spread_bps=(a-b)/((a+b)/2)*10000)
                    else:row['quality']='missing_best_prices'
                else:row['reason']=str(book)[:180]
                rows.append(row);time.sleep(1)
            # OpenD requires a subscription to remain for one minute. This is
            # an asynchronous shell process, not a blocking assistant tool wait.
            while time.monotonic()-started<62:time.sleep(1)
            ret,msg=q.unsubscribe(codes,[ft.SubType.ORDER_BOOK]);unsubscribe=dict(ok=ret==ft.RET_OK,reason=str(msg))
        write(OUT/'book_probe.json',dict(at=now(),before_own_subscription=before if isinstance(before,dict) else str(before)[:180],subscribe_ok=subscribed,subscribe_reason=str(msg),books=rows,unsubscribe=unsubscribe,orders_sent=False,paid_permissions_requested=False,historical_execution_proven=False,note='This connection only; current received BBO is not historical BBO, guaranteed fill or price available at a past signal. Existing connection subscriptions untouched.'))
    finally:q.close()
    print(json.dumps(dict(books=len(rows),subscribed=subscribed,unsubscribe=unsubscribe)),flush=True)
if __name__=='__main__':main()
