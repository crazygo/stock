# calendars/ · 官方交易所交易日历说明

## 1. 数据来源与生成方法
- **官方依据**：Nasdaq 官方假期与交易时间排期（[Nasdaq Stock Market Holiday Schedule](https://www.nasdaq.com/market-activity/stock-market-holiday-schedule)）
- **生成脚本**：[`scripts/build_trading_calendar.py`](file:///Users/admin/Code/stock/scripts/build_trading_calendar.py)
- **主产物文件**：`market_data/calendars/nasdaq_sessions_2026_v1.json`
- **覆盖区间**：`2025-12-01` 至 `2026-10-15`（包含 2025 年 12 月特征回看预热、2026 全年以及用于未来 3 日标签成熟度计算的窗口余量）。

## 2. 口径与核心规则
- **显式交易区间**：每一个合法交易日提供美东时区与 UTC ISO-8601 格式的 `open_at` 与 `close_at`。
- **半日市与提前收盘**：
  - 2025-12-24（平安夜）：13:00 提前收盘（常规交易时长 210 分钟）
  - 2026-07-03（独立日前一日/调休）：13:00 提前收盘或全天休市（按官方规则严格标注）
  - 2026-11-27（黑色星期五）：13:00 提前收盘
  - 2026-12-24（平安夜）：13:00 提前收盘
- **夏令时 / 冬令时自动对齐**：使用 `zoneinfo.ZoneInfo("America/New_York")` 处理 2026-03-08（夏令时开始，UTC-4）与 2026-11-01（冬令时开始，UTC-5）。

## 3. 开发者铁律
**严禁从“某只股票是否有 K 线”倒推交易日历**！某只股票在某天没有数据，可能是停牌或抓取缺失，绝不等于交易所休市。任何模型、特征或标签的日历计算必须以本目录下的官方 `sessions` 为唯一真实标准。
