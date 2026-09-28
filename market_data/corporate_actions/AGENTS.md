# corporate_actions/ · 美股公司行动与除权除息说明

## 1. 数据源与采集接口
- **提供方**：Futu OpenD Gateway (`ft.OpenQuoteContext.get_rehab(code)`)
- **存放格式**：`market_data/corporate_actions/<SYMBOL>.parquet`，压缩为 ZSTD level 7。
- **获取调度**：单次请求获取该标的上市以来的全量历史拆股与分红事件，无额外股票配额消耗。

## 2. 字段与口径规范
包含核心字段：
- `ex_div_date` (`string / date`): 除权除息日（生效日）
- `split_base` / `split_ratio` (`float`): 拆股拆合基数与比例（例如 1 拆 10 时，split_base=1, split_ratio=10）
- `cash_dividend` (`float`): 每股派发现金红利（美元）
- `forward_adj_factorA` / `forward_adj_factorB`: 前复权系数
- `backward_adj_factorA` / `backward_adj_factorB`: 后复权系数

## 3. 业务含义与模型使用规范
1. **未复权价格配对**：新策略工程 [`research/after_open_3d5pct/`](file:///Users/admin/Code/stock/research/after_open_3d5pct/) 采用未复权价格。若持有窗口内（如未来 3 个交易日）发生拆股，必须依据此表按比例调整目标价；严禁直接把未复权的价格跌幅（如 1 拆 2 导致的 −50%）计为真实亏损。
2. **因果防泄漏契约**：历史回测与训练时，必须严格以决策时点已知的信息为准，不得在除权公告日前提前偷看未来的拆股因子。
