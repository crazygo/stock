# us_60m_raw/ · 美股 60 分钟未复权全时段行情说明

## 1. 数据源与采集配置
- **提供方**：Futu OpenD Gateway (`ft.OpenQuoteContext.request_history_kline`)
- **K 线类型**：`ktype = ft.KLType.K_60M`（60 分钟）
- **复权口径**：`autype = ft.AuType.NONE`（**未复权原始价格**）
- **时段范围**：`extended_time = True`（**全时段 ALL**）
- **存放格式**：`market_data/us_60m_raw/<SYMBOL>/2026.parquet`，压缩为 ZSTD level 7。

## 2. 核心边界柱与时间口径修正（重大实证发现）
由于美股常规交易时段为 `09:30 ~ 16:00`，OpenD 的小时线并不是统一 60 分钟：
- **`16:00` 收盘柱**：其实际时间区间为 **`15:30:00 ~ 16:00:00`（时长为 30 分钟）**，已在规范化时精确设为 `start_at = 15:30`。
- **`09:30` 盘前收盘柱**：区间为 `09:00:00 ~ 09:30:00`（时长为 30 分钟）。
- **`10:30` 开盘首柱**：区间为 `09:30:00 ~ 10:30:00`（时长为 60 分钟）。
- **半日闭市日（如 13:00 闭市）**：收盘柱为 `12:30:00 ~ 13:00:00`（30 分钟）。

> **已核验证明**：在经此修正后，本目录的 60m OHLC 与 `us_5m/` 聚合出的每小时高低开收价在常规盘（RTH）达到了 **100.0% 的完全精确吻合**（详见 `market_data/manifests/quality_findings.md`）。

## 3. 使用场景
- 为新研究工程 [`research/after_open_3d5pct/`](file:///Users/admin/Code/stock/research/after_open_3d5pct/) 提供多尺度高层趋势特征、开盘前两小时动量特征；
- 原始价格与 `us_5m/` 及 `corporate_actions/` 同源，杜绝任何复权基准混合引发的因果泄漏。
