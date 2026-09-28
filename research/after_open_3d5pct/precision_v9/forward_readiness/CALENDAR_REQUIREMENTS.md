# 前向日历覆盖依赖

2026-09-27，根代理只读核对：共享 `market_data/calendars/nasdaq_sessions_2026_v1.json` 实际覆盖 2025-12-01 至 2026-10-15，共 220 个 session；R03 训练日历截止 2026-09-24。两者都不能直接承载从下一交易日开始的连续 60-session 前向 cohort 及其后最长标签窗口。

真实 cohort 启动前须另建不可变完整日历版本，覆盖输入回看、cohort 全期和最后一条信号的 1950 个常规交易分钟。固定期末评价与无信号日期分母均依赖这张表；禁止只按已抓到行情的日期补齐。

本次已从 [Nasdaq 官方交易日历](https://nasdaqtrader.com/Trader.aspx?id=Calendar) 和 [NYSE 官方交易时间表](https://www.nyse.com/trade/hours-calendars) 交叉核对：2026-11-26、2026-12-25 全日休市；2026-11-27、2026-12-24 常规盘在美东 13:00 结束。其余已公布年度节假日必须一并冻结，ET/UTC转换使用 America/New_York，包括11月夏令时切换。

当前账本单测可使用明确标注的 fixture 日历，但不由fixture创建真实cohort。本页仅记录来源和覆盖缺口，尚未生成新日历或开始真实前向采集。后续日历制品应保存来源URL、抓取时间、原始内容哈希及规范化session表哈希。
