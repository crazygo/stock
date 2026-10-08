import futu as ft
import pandas as pd

ft.SysConfig.enable_proto_encrypt(False)

trd_ctx = ft.OpenSecTradeContext(
    filter_trdmarket=ft.TrdMarket.US,
    host='127.0.0.1',
    port=11111,
    security_firm=ft.SecurityFirm.FUTUINC
)

acc_id = 283445330641239202 # Cash account

ret_h, h_deals = trd_ctx.history_deal_list_query(acc_id=acc_id, start='2026-10-06', end='2026-10-08')
if ret_h == ft.RET_OK:
    h_deals['create_time'] = h_deals['create_time'].astype(str)
    h_deals = h_deals.sort_values(by='create_time', ascending=False)
    print(f"Total deals from 2026-10-06: {len(h_deals)}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    
    # Calculate amount
    h_deals['amount'] = h_deals['qty'] * h_deals['price']
    
    cols = ['create_time', 'trd_side', 'code', 'stock_name', 'qty', 'price', 'amount', 'order_id', 'deal_id', 'status']
    print(h_deals[cols].to_string(index=False))

trd_ctx.close()
