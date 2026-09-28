# us_5m/ · 美股 5 分钟未复权全时段行情说明

## 1. 数据源与采集配置
- **提供方**：Futu OpenD Gateway (`ft.OpenQuoteContext.request_history_kline`)
- **K 线类型**：`ktype = ft.KLType.K_5M`（5 分钟）
- **复权口径**：`autype = ft.AuType.NONE`（**未复权原始价格**）
- **时段范围**：`extended_time = True`（**全时段 ALL**，包含 20:00~04:00 夜盘、04:00~09:30 盘前、09:30~16:00 常规盘、16:00~20:00 盘后）
- **抓取调度**：首页间隔 $\ge 1.2\text{s}$，分页续页间隔 $\ge 0.3\text{s}$，单页 `max_count = 1000`，逐页传回 `page_req_key` 直至为空。

## 2. 字段与时间口径规范
每个 Parquet 文件路径为：`market_data/us_5m/<SYMBOL>/2026.parquet`，压缩方式严格为 **Apache Parquet (ZSTD level 7)**。

| 字段名 | 类型 | 说明与口径 |
| :--- | :--- | :--- |
| `symbol` | `string` | 美股个股代码（如 `AAPL`, `NVDA`, `QQQ`） |
| `time_key` | `string` | OpenD 原生时间戳字符串，为 **美东时间 (ET) 该根 K 线的结束时刻** (如 `2026-09-01 09:35:00`) |
| `start_at` | `string (ISO-8601)` | **UTC 标注的 bar 开始时刻**。对于 09:35 的 bar，start_at 为 09:30 ET 对应的 UTC |
| `end_at` | `string (ISO-8601)` | **UTC 标注的 bar 结束时刻**。与 time_key 对应 |
| `available_at` | `string (ISO-8601)` | **该 bar 在本系统最早可用时刻**。默认为 `end_at + 1秒`，严禁未来数据穿越 |
| `start_at_et` / `end_at_et` | `string` | 美东时区（`America/New_York`，自动自适应冬令时/夏令时）带时区偏移格式 |
| `session_date` | `string (YYYY-MM-DD)` | 对应美东交易日日期 |
| `session_type` | `string` | 时段分类：`regular` (09:30~16:00), `pre_market` (04:00~09:30), `post_market` (16:00~20:00), `overnight` |
| `open` / `high` / `low` / `close` | `float64` | **未复权原始美元价格** |
| `volume` | `float64` | 该 5 分钟内总成交股数 |
| `turnover` | `float64` | 该 5 分钟内总成交金额（美元） |
| `pe_ratio` / `turnover_rate` | `float64` | 分钟级该项多为 0（官方仅在日线级提供），不可视为真实零估值 |
| `last_close` | `float64` | 上一根 5m bar 的收盘价（非前一交易日收盘价！） |
| `price_basis` | `string` | 固定为 `"NONE"` |

## 3. 使用场景与风控禁令
- **核心用途**：新策略工程 [`research/after_open_3d5pct/`](file:///Users/admin/Code/stock/research/after_open_3d5pct/) 的唯一基准入场（11:35 首根 5m Open）与 1170 分钟标签窗口计算。
- **边界映射**：11:30 生成信号并模拟延迟后，首选入场价必须取 **`time_key = 11:40:00` 的 5m Open**（因为其 `start_at` 为 11:35:00）。误取 11:35 的 Open 会偷看 11:30 的开盘价！
- **严禁**：不得与 `us_60m` 中的前复权数据交叉混合计算收益率。
