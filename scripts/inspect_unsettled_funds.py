import futu as ft
import pandas as pd
import datetime

ft.SysConfig.enable_proto_encrypt(False)

trd_ctx = ft.OpenSecTradeContext(
    filter_trdmarket=ft.TrdMarket.US,
    host='127.0.0.1',
    port=11111,
    security_firm=ft.SecurityFirm.FUTUINC
)

acc_id = 283445330641239202 # Cash account

print("=== 1. Cash Account accinfo_query ===")
ret_info, info_df = trd_ctx.accinfo_query(acc_id=acc_id)
if ret_info == ft.RET_OK:
    # Print cash related fields
    for col in info_df.columns:
        val = info_df[col].iloc[0]
        if any(k in col.lower() for k in ['cash', 'unsettle', 'settle', 'power', 'asset', 'frozen', 'val']):
            print(f"  {col}: {val}")
else:
    print("accinfo_query failed:", info_df)

print("\n=== 2. Today's Deals (deal_list_query) ===")
ret_d, deals = trd_ctx.deal_list_query(acc_id=acc_id)
if ret_d == ft.RET_OK:
    print(f"Today deals: {len(deals)}")
    if len(deals) > 0:
        for idx, row in deals.iterrows():
            amt = row['qty'] * row['price']
            print(f"  [{row['create_time']}] {row['trd_side']:4s} {row['code']:9s} ({row.get('stock_name','')}) | Qty:{row['qty']} | Price:{row['price']} | Amt:{amt:.2f} | Order:{row['order_id']}")
else:
    print("deal_list_query failed:", deals)

print("\n=== 3. Deals from 2026-10-06 to 2026-10-08 ===")
ret_h, h_deals = trd_ctx.history_deal_list_query(acc_id=acc_id, start='2026-10-06', end='2026-10-08')
if ret_h == ft.RET_OK:
    print(f"Total deals: {len(h_deals)}")
    h_deals['create_time'] = h_deals['create_time'].astype(str)
    h_deals = h_deals.sort_values(by='create_time', ascending=False)
    for idx, row in h_deals.iterrows():
        amt = row['qty'] * row['price']
        print(f"  [{row['create_time']}] {row['trd_side']:4s} {row['code']:9s} ({row.get('stock_name',''):15s}) | Qty:{row['qty']:5.1f} | Price:{row['price']:8.4f} | Amt:{amt:9.2f} | Order:{row['order_id']} | Deal:{row['deal_id']}")
else:
    print("history_deal_list_query failed:", h_deals)

print("\n=== 4. Check Orders from 2026-10-06 to 2026-10-08 ===")
ret_o, h_orders = trd_ctx.history_order_list_query(acc_id=acc_id, start='2026-10-06', end='2026-10-08')
if ret_o == ft.RET_OK:
    print(f"Total orders: {len(h_orders)}")
    h_orders['updated_time'] = h_orders['updated_time'].astype(str)
    h_orders = h_orders.sort_values(by='updated_time', ascending=False)
    for idx, row in h_orders.iterrows():
        print(f"  [{row['updated_time']}] {row['order_status']:15s} {row['trd_side']:4s} {row['code']:9s} ({row.get('stock_name',''):12s}) | Qty:{row['qty']:5.1f} | DealtQty:{row['dealt_qty']:5.1f} | DealtAvgPrice:{row['dealt_avg_price']:8.4f} | OrderID:{row['order_id']}")

trd_ctx.close()
