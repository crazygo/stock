# us_60m/ · 美股 60 分钟前复权全时段行情说明

## 1. 数据源与采集配置
- **提供方**：Futu OpenD Gateway (`ft.OpenQuoteContext.request_history_kline`)
- **K 线类型**：`ktype = ft.KLType.K_60M`（60 分钟）
- **复权口径**：`autype = ft.AuType.QFQ`（**前复权 Forward-Adjusted**）
- **时段范围**：`extended_time = True`（**全时段 ALL**，包含 24 小时各交易时段）
- **存放格式**：`market_data/us_60m/<SYMBOL>/2026.parquet`，压缩为 ZSTD level 7。

## 2. 字段口径
包含字段：`['time_key', 'open', 'close', 'high', 'low', 'pe_ratio', 'turnover_rate', 'volume', 'turnover', 'change_rate', 'last_close']`。
- `time_key` 为美东本地时间的结束时刻标签；
- `open`, `high`, `low`, `close` 均为根据最近一次除权除息因子调整后的前复权美元价格；
- `last_close` 为前一根 60m bar 的收盘价。

## 3. 使用场景与系统定位
- **定位**：本目录数据属于**既有生产分析资产**，用于支撑：
  1. [`analysis/clustering_methods/method_1_hourly_by_week/`](file:///Users/admin/Code/stock/analysis/clustering_methods/method_1_hourly_by_week/)（周度 × 小时交叉热力图）
  2. [`analysis/clustering_methods/method_2_hourly_by_day/`](file:///Users/admin/Code/stock/analysis/clustering_methods/method_2_hourly_by_day/)（日度微观小时达成率看板）
  3. [`analysis/today_intraday_watch/`](file:///Users/admin/Code/stock/analysis/today_intraday_watch/)（盘中动态监控看板）
- **注意禁令**：新预测模型（如开盘后 3 日 +5%）入场模拟**严禁使用本目录作为未复权成交价**，新训练请使用 `us_5m/` 或 `us_60m_raw/`。
