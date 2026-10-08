#!/usr/bin/env python3
"""
Moomoo US 现金账户未交收资金穿透与逐笔清算分析工具
- 绕过 get_acc_cash_flow 无法查询美股现金流水的限制
- 结合 accinfo_query、history_deal_list_query、order_list_query
- 按照美国 SEC T+1 交收规则与时区换算 (HKT - 12h = EDT) 自动重构未交收与已交收资金明细
"""

import sys
import datetime
import pandas as pd
import futu as ft

def analyze_unsettled_funds(acc_id: int = 283445330641239202, days: int = 7):
    ft.SysConfig.enable_proto_encrypt(False)

    trd_ctx = ft.OpenSecTradeContext(
        filter_trdmarket=ft.TrdMarket.US,
        host='127.0.0.1',
        port=11111,
        security_firm=ft.SecurityFirm.FUTUINC
    )

    print("=" * 70)
    print(f"🚀 Moomoo US 资金与未交收清算分析 (账户 ID: {acc_id})")
    print("=" * 70)

    # 1. 账户资产与现金现状
    ret_info, info_df = trd_ctx.accinfo_query(acc_id=acc_id)
    if ret_info != ft.RET_OK:
        print(f"❌ 获取账户资产失败: {info_df}")
        trd_ctx.close()
        return

    row = info_df.iloc[0]
    us_cash = row.get('us_cash', 0.0)
    us_avl_with = row.get('us_avl_withdrawal_cash', 0.0)
    total_assets = row.get('usd_assets', row.get('total_assets', 0.0))
    frozen_cash = row.get('frozen_cash', 0.0)
    unsettled_diff = us_cash - us_avl_with

    print(f"\n📊 资金总览 (实时快照):")
    print(f"  • 美元总资产:       ${total_assets:,.2f} USD")
    print(f"  • 账面总现金 (Cash): ${us_cash:,.2f} USD")
    print(f"  • 可提现金额 (Avl):  ${us_avl_with:,.2f} USD")
    print(f"  • 现金购买力 (Power): ${row.get('usd_net_cash_power', us_avl_with):,.2f} USD")
    print(f"  • 待清算/规费差额:   ${unsettled_diff:,.2f} USD (港币冻结字段: HK${frozen_cash:,.2f})")

    # 2. 当前挂单 (Order List) 检查
    ret_o, orders = trd_ctx.order_list_query(acc_id=acc_id)
    active_orders = []
    if ret_o == ft.RET_OK and len(orders) > 0:
        for _, o in orders.iterrows():
            if o['order_status'] in ['SUBMITTED', 'WAITING_SUBMIT', 'SUBMITTING']:
                active_orders.append(o)
    
    print(f"\n📋 当前未成交挂单: {len(active_orders)} 笔")
    for o in active_orders:
        print(f"  • [{o['order_status']}] {o['trd_side']} {o['code']} ({o.get('stock_name','')}) | 数量: {o['qty']} | 挂单价: ${o['price']}")

    # 3. 历史成交 (Deal List) 与 T+1 交收时序重构
    start_date = (datetime.date.today() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.date.today().strftime('%Y-%m-%d')

    ret_h, h_deals = trd_ctx.history_deal_list_query(acc_id=acc_id, start=start_date, end=end_date)
    if ret_h != ft.RET_OK or len(h_deals) == 0:
        print(f"\n⚠️ 未获取到近期成交记录: {h_deals}")
        trd_ctx.close()
        return

    h_deals['create_time_hkt'] = pd.to_datetime(h_deals['create_time'])
    # HKT (UTC+8) -> EDT (UTC-4), 慢 12 小时
    h_deals['create_time_edt'] = h_deals['create_time_hkt'] - pd.Timedelta(hours=12)
    h_deals['us_trade_date'] = h_deals['create_time_edt'].dt.date
    h_deals['amount'] = h_deals['qty'] * h_deals['price']
    h_deals = h_deals.sort_values(by='create_time_hkt', ascending=False)

    # 判定当前美股日与 T+1 交收状态
    # 以美东当前日期为准
    now_edt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=4)
    today_edt = now_edt.date()

    print(f"\n📅 美东当前交易日参考: {today_edt} EDT (系统本地时间: {datetime.datetime.now():%Y-%m-%d %H:%M:%S})")
    print(f"🔍 过去 {days} 天成交记录拆解 (按美股交易日与交收状态):")

    for dt, group in h_deals.groupby('us_trade_date', sort=False):
        # 计算 T+1 交收日 (周五跳到周一)
        trade_dt = dt
        settle_days = 1
        # 若是周五，T+1 为下周一
        if trade_dt.weekday() == 4:
            settle_dt = trade_dt + datetime.timedelta(days=3)
        elif trade_dt.weekday() == 5:
            settle_dt = trade_dt + datetime.timedelta(days=2)
        else:
            settle_dt = trade_dt + datetime.timedelta(days=1)

        is_settled = settle_dt <= today_edt
        status_label = "✅ 已完成交收 (Settled)" if is_settled else "⏳ 未交收 (Unsettled - T+1在途中)"

        buys = group[group['trd_side'] == 'BUY']
        sells = group[group['trd_side'] == 'SELL']
        buy_amt = buys['amount'].sum()
        sell_amt = sells['amount'].sum()
        net_amt = sell_amt - buy_amt

        print(f"\n--- 【美股交易日 {trade_dt}】 · 预计交收日: {settle_dt} · {status_label} ---")
        print(f"    买入: {len(buys)}笔 / -${buy_amt:,.2f} | 卖出: {len(sells)}笔 / +${sell_amt:,.2f} | 净额: ${net_amt:+,.2f}")
        
        for _, r in group.iterrows():
            hkt_str = r['create_time_hkt'].strftime('%m-%d %H:%M:%S')
            edt_str = r['create_time_edt'].strftime('%m-%d %H:%M:%S')
            side = r['trd_side']
            sign = "+" if side == "SELL" else "-"
            print(f"      • [{edt_str} EDT | {hkt_str} HKT] {side:4s} {r['code']:8s} {r.get('stock_name',''):10s} | {r['qty']:4.0f}股 @ ${r['price']:7.2f} = {sign}${r['amount']:8.2f} | 状态: {'已交收' if is_settled else '未交收'}")

    trd_ctx.close()
    print("\n" + "=" * 70)
    print("💡 现金账户风控备忘 (Good Faith Violation / GFV):")
    print("  1. 卖出股票回笼资金在 T+1 交收完成前，可立即用于买入新标的，但不可提现。")
    print("  2. 若使用未交收资金买入某只股票，必须等待原卖单完成 T+1 交收后方可卖出该股票，否则构成 GFV。")
    print("=" * 70)

if __name__ == '__main__':
    acc = 283445330641239202
    if len(sys.argv) > 1:
        acc = int(sys.argv[1])
    analyze_unsettled_funds(acc_id=acc)
