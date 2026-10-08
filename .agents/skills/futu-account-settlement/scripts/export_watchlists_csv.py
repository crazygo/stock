#!/usr/bin/env python3
"""
Futu OpenD 自选股全部分组与标的一键导出 CSV 工具
- 避坑：必须使用 ft.UserSecurityGroupType.ALL (勿用不存在的 SecurityGroupType)
- 内置频控防护：每批次请求增加 0.5s 停顿，防止触发 30秒内最多10次 的频率限制
- 输出格式：统一导出至 CSV 文件，并输出字段清单
"""

import sys
import os
import time
import argparse
import pandas as pd
import futu as ft

def export_watchlists(output_file: str = "futu_watchlists.csv", host: str = "127.0.0.1", port: int = 11111, target_groups: list = None, delay: float = 3.2):
    ft.SysConfig.enable_proto_encrypt(False)
    
    quote_ctx = ft.OpenQuoteContext(host=host, port=port)
    print("=" * 60)
    print("🔍 正在从 Futu OpenD 读取自选股分组...")
    print("=" * 60)

    # 1. 查询全部自选股分组列表 (必须使用 ft.UserSecurityGroupType.ALL)
    ret_g, group_df = quote_ctx.get_user_security_group(group_type=ft.UserSecurityGroupType.ALL)
    if ret_g != ft.RET_OK:
        print(f"❌ 获取自选分组失败: {group_df}")
        quote_ctx.close()
        return False

    print(f"✅ 成功获取到 {len(group_df)} 个自选分组:")
    for _, r in group_df.iterrows():
        print(f"  • 分组名称: [{r['group_name']}] | 类型: {r.get('group_type', '')}")

    if target_groups:
        group_df = group_df[group_df['group_name'].isin(target_groups)]
        print(f"🎯 过滤目标分组 ({len(group_df)} 个): {list(group_df['group_name'])}")

    # 2. 逐组获取标的列表 (内置频控防护与重试)
    all_records = []
    
    for idx, r in group_df.iterrows():
        g_name = r['group_name']
        print(f"\n📥 正在读取分组 [{g_name}] 的标的...")
        
        # 频率保护：富途限制 30 秒内最多 10 次请求。为留出安全裕度，每次请求前确保距离上一批请求有至少 3.2s 间隔，遇限流自动重试
        ret_s = None
        sec_df = None
        for retry_attempt in range(3):
            time.sleep(delay)
            ret_s, sec_df = quote_ctx.get_user_security(group_name=g_name)
            if ret_s == ft.RET_OK:
                break
            if "频率太高" in str(sec_df) or "30秒" in str(sec_df):
                print(f"  ⏳ 触发富途频控，等待 30 秒后进行第 {retry_attempt + 1}/3 次重试...")
                time.sleep(30)
            else:
                break

        if ret_s != ft.RET_OK:
            print(f"  ⚠️ 获取分组 [{g_name}] 标的失败或为空: {sec_df}")
            continue

        sec_count = len(sec_df)
        print(f"  -> 获取成功: {sec_count} 只标的")
        
        for _, s in sec_df.iterrows():
            rec = {
                'group_name': g_name,
                'code': s['code'],
                'name': s.get('name', ''),
                'market': s['code'].split('.')[0] if '.' in s['code'] else 'N/A',
                'symbol': s['code'].split('.')[1] if '.' in s['code'] else s['code']
            }
            # 如果有额外字段
            for extra_col in ['lot_size', 'stock_type', 'create_time']:
                if extra_col in s:
                    rec[extra_col] = s[extra_col]
            all_records.append(rec)

    quote_ctx.close()

    if not all_records:
        print("❌ 未获取到任何有效自选股票记录。")
        return False

    res_df = pd.DataFrame(all_records)
    
    # 确保目标目录存在
    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    res_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("\n" + "=" * 60)
    print(f"🎉 自选股导出完成！共 {len(res_df)} 条记录 (去重代码数: {res_df['code'].nunique()} 只)")
    print(f"📁 保存路径: {os.path.abspath(output_file)}")
    print("=" * 60)
    
    # 打印前 10 条样本预览
    print("\n📊 导出数据前 10 行预览:")
    print(res_df.head(10).to_string(index=False))
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Export Futu watchlists to CSV")
    parser.add_argument("--output", "-o", default="futu_watchlists.csv", help="Output CSV path")
    parser.add_argument("--host", default="127.0.0.1", help="OpenD host")
    parser.add_argument("--port", type=int, default=11111, help="OpenD port")
    parser.add_argument("--groups", "-g", nargs="+", default=None, help="Specific group names to export")
    parser.add_argument("--delay", type=float, default=3.2, help="Delay in seconds between requests (default: 3.2s for rate limit safe)")
    args = parser.parse_args()
    
    export_watchlists(output_file=args.output, host=args.host, port=args.port, target_groups=args.groups, delay=args.delay)
