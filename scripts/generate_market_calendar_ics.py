#!/usr/bin/env python3
"""
生成美股交易时段 (盘前 Pre-Market / 盘后 After-Hours) 及休市日历 ICS 文件
- 时区原生绑定: America/New_York (自动精准适配每年夏令时/冬令时切换，无缝显示北京时间)
- 盘前交易时段: 周一至周五 04:00 - 09:30 ET (内置开盘前 15 分钟闹钟提醒)
- 盘后交易时段: 周一至周五 16:00 - 20:00 ET (提示常规盘收盘)
- 自动排除法定休市日: 2026-2027 年 NYSE 全部闭市假日
"""

import os
from datetime import datetime, timezone

# 2026 - 2027 NYSE 官方全天休市日 (美东时间日期)
NYSE_HOLIDAYS = [
    # 2026
    ("20260101", "元旦 (New Year's Day)"),
    ("20260119", "马丁·路德·金纪念日 (MLK Day)"),
    ("20260216", "华盛顿诞辰 / 总统日 (Presidents' Day)"),
    ("20260403", "耶稣受难日 (Good Friday)"),
    ("20260525", "阵亡将士纪念日 (Memorial Day)"),
    ("20260619", "六月节国家独立日 (Juneteenth)"),
    ("20260703", "独立日休市 (Independence Day Observed)"),
    ("20260907", "劳动节 (Labor Day)"),
    ("20261126", "感恩节 (Thanksgiving Day)"),
    ("20261225", "圣诞节 (Christmas Day)"),
    # 2027
    ("20270101", "元旦 (New Year's Day)"),
    ("20270118", "马丁·路德·金纪念日 (MLK Day)"),
    ("20270215", "华盛顿诞辰 / 总统日 (Presidents' Day)"),
    ("20270326", "耶稣受难日 (Good Friday)"),
    ("20270531", "阵亡将士纪念日 (Memorial Day)"),
    ("20270618", "六月节休市 (Juneteenth Observed)"),
    ("20270705", "独立日休市 (Independence Day Observed)"),
    ("20270906", "劳动节 (Labor Day)"),
    ("20271125", "感恩节 (Thanksgiving Day)"),
    ("20271224", "圣诞节休市 (Christmas Day Observed)")
]

def build_ics_content() -> str:
    now_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # 排除日期的 EXDATE 字符串
    pre_exdates = ",".join([f"{h[0]}T040000" for h in NYSE_HOLIDAYS])
    post_exdates = ",".join([f"{h[0]}T160000" for h in NYSE_HOLIDAYS])

    ics = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Stock Assistant//US Market Trading Sessions//CN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "X-WR-CALNAME:美股交易时段 (盘前/盘后)",
        "X-WR-CALDESC:美股盘前交易与盘后交易时间表。自动适配夏令时/冬令时，包含开盘提醒与休市标记。",
        "X-WR-TIMEZONE:America/New_York",

        # 时区定义 (支持自动夏令时/冬令时调整)
        "BEGIN:VTIMEZONE",
        "TZID:America/New_York",
        "X-LIC-LOCATION:America/New_York",
        "BEGIN:DAYLIGHT",
        "TZOFFSETFROM:-0500",
        "TZOFFSETTO:-0400",
        "TZNAME:EDT",
        "DTSTART:19700308T020000",
        "RRULE:FREQ=YEARLY;BYMONTH=3;BYDAY=2SU",
        "END:DAYLIGHT",
        "BEGIN:STANDARD",
        "TZOFFSETFROM:-0400",
        "TZOFFSETTO:-0500",
        "TZNAME:EST",
        "DTSTART:19701101T020000",
        "RRULE:FREQ=YEARLY;BYMONTH=11;BYDAY=1SU",
        "END:STANDARD",
        "END:VTIMEZONE",

        # 1. 盘前交易时段 (周一至周五 04:00 - 09:30 ET)
        "BEGIN:VEVENT",
        "UID:us-market-pre-recurring@stock-quant",
        f"DTSTAMP:{now_stamp}",
        "SUMMARY:🔔 美股盘前交易 (04:00-09:30 ET)",
        "DESCRIPTION:【美股盘前交易时段】\\n• 美东时间: 04:00 - 09:30\\n• 北京时间 (夏令时): 16:00 - 21:30\\n• 北京时间 (冬令时): 17:00 - 22:30\\n\\n⚡ 提示: 盘前结束即为美股常规盘开盘时刻！",
        "DTSTART;TZID=America/New_York:20261005T040000",
        "DTEND;TZID=America/New_York:20261005T093000",
        "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
        f"EXDATE;TZID=America/New_York:{pre_exdates}",
        "BEGIN:VALARM",
        "ACTION:DISPLAY",
        "DESCRIPTION:美股常规盘即将在 15 分钟后开盘！(9:30 ET / 21:30 BJT)",
        "TRIGGER:-PT15M",
        "END:VALARM",
        "END:VEVENT",

        # 2. 盘后交易时段 (周一至周五 16:00 - 20:00 ET)
        "BEGIN:VEVENT",
        "UID:us-market-post-recurring@stock-quant",
        f"DTSTAMP:{now_stamp}",
        "SUMMARY:🌙 美股盘后交易 (16:00-20:00 ET)",
        "DESCRIPTION:【美股盘后交易时段】\\n• 美东时间: 16:00 - 20:00\\n• 北京时间 (夏令时): 次日凌晨 04:00 - 08:00\\n• 北京时间 (冬令时): 次日凌晨 05:00 - 09:00\\n\\n⚡ 提示: 常规盘已于 16:00 ET 收盘！",
        "DTSTART;TZID=America/New_York:20261005T160000",
        "DTEND;TZID=America/New_York:20261005T200000",
        "RRULE:FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
        f"EXDATE;TZID=America/New_York:{post_exdates}",
        "END:VEVENT"
    ]

    # 3. 休市日单独标注 (全天事件)
    for h_date, h_name in NYSE_HOLIDAYS:
        ics.extend([
            "BEGIN:VEVENT",
            f"UID:nyse-holiday-{h_date}@stock-quant",
            f"DTSTAMP:{now_stamp}",
            f"SUMMARY:🔴 美股休市: {h_name}",
            f"DESCRIPTION:美国证券交易所 (NYSE/NASDAQ) 全天休市。无盘前、常规盘及盘后交易。",
            f"DTSTART;VALUE=DATE:{h_date}",
            f"DTEND;VALUE=DATE:{h_date}",
            "STATUS:CONFIRMED",
            "TRANSP:TRANSPARENT",
            "END:VEVENT"
        ])

    ics.append("END:VCALENDAR")
    return "\r\n".join(ics) + "\r\n"

def main():
    content = build_ics_content()
    out_path = os.path.abspath("us_market_sessions.ics")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ 美股交易时段 ICS 文件生成成功: {out_path}")
    print(f"📊 文件大小: {len(content)} 字节")
    print(f"🌐 本地 Web 服务访问 URL: http://127.0.0.1:8768/us_market_sessions.ics")

if __name__ == '__main__':
    main()
