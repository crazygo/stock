# universe/ · 标的池元数据与基准映射说明

## 1. 数据来源与生成方法
- **来源基础**：Slickcharts Nasdaq-100 成分股清单与宽基/行业 ETF 基准配置。
- **生成脚本**：[`scripts/build_universe_metadata.py`](file:///Users/admin/Code/stock/scripts/build_universe_metadata.py)
- **主产物文件**：`market_data/universe/qqq_retrospective_v1.json`

## 2. 角色划分与口径规范
每个标的严格标注 `role`：
- **`candidate`（101 只候选股）**：QQQ 成分股中的个股（如 `AAPL`, `NVDA`, `MSFT` 等），作为策略特征计算、入场与标签判定的核心标的。
- **`benchmark`（7 只基准 ETF）**：
  - 宽基指数基准：`QQQ` (纳斯达克100), `SPY` (标普500), `DIA` (道琼斯)
  - 行业代理基准：`SOXX` / `SMH` (半导体行业), `IGV` (软件云服务), `XLU` (公用事业)
  - **规则**：基准标的用于计算超额收益与相对强弱特征，**不自动进入候选买入池**。

## 3. 幸存者偏差与协议模式声明
- **协议模式**：`universe_mode = "current_universe_retrospective"`
- **科学诚实性要求**：当前标的池为当前快照回溯。因尚未接驳历史每季度成分剔除生效公告，在研究报告中必须明确声明为“当前池回溯测试”，不得声称完全无幸存者偏差。
