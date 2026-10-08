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

ret_h, h_deals = trd_ctx.history_deal_list_query(acc_id=acc_id, start='2026-10-05', end='2026-10-08')
if ret_h == ft.RET_OK:
    h_deals['create_time'] = pd.to_datetime(h_deals['create_time'])
    h_deals = h_deals.sort_values(by='create_time', ascending=True)
    h_deals['amount'] = h_deals['qty'] * h_deals['price']
    
    # Let's group by trade date
    # Note: US Eastern Time is UTC-4. HKT is UTC+8 (12 hours ahead of EDT).
    # Convert HKT to US Eastern Time (EDT)
    h_deals['create_time_hkt'] = h_deals['create_time']
    h_deals['create_time_edt'] = h_deals['create_time'] - pd.Timedelta(hours=12)
    h_deals['us_trade_date'] = h_deals['create_time_edt'].dt.date
    
    print("=== SUMMARY BY US TRADE DATE ===")
    for dt, group in h_deals.groupby('us_trade_date'):
        buys = group[group['trd_side'] == 'BUY']
        sells = group[group['trd_side'] == 'SELL']
        buy_amt = buys['amount'].sum()
        sell_amt = sells['amount'].sum()
        net_amt = sell_amt - buy_amt
        print(f"\nUS Trade Date: {dt} (Total {len(group)} deals)")
        print(f"  买入总计: {len(buys)}笔 | ${buy_amt:.2f}")
        print(f"  卖出总计: {len(sells)}笔 | ${sell_amt:.2f}")
        print(f"  净资金流入/流出 (Sells - Buys): ${net_amt:+.2f}")
        
    print("\n=== DETAILED LIST OF DEALS (CHRONOLOGICAL) ===")
    cols = ['create_time_hkt', 'create_time_edt', 'us_trade_date', 'trd_side', 'code', 'stock_name', 'qty', 'price', 'amount', 'order_id']
    for idx, r in h_deals.iterrows():
        edt_str = r['create_time_edt'].strftime('%Y-%m-%d %H:%M:%S')
        hkt_str = r['create_time_hkt'].strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{edt_str} EDT | {hkt_str} HKT] {r['trd_side']:4s} {r['code']:8s} {r.get('stock_name',''):10s} | {r['qty']:4.0f}股 @ ${r['price']:7.2f} = ${r['amount']:8.2f} | Order:{r['order_id']}")

trd_ctx.close()
