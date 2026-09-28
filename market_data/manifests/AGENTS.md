# manifests/ · 行情数仓审计台账与数据血缘说明

## 1. 目录职责
本目录存放行情数仓与模型训练数据集的**元数据血缘（Lineage）、采购审计计划、请求流水日志与质量验证报告**，确保所有进入模型或策略的数据集具备 100% 的可追溯性与因果防篡改证据。

## 2. 核心产物文件清单
| 文件 | 格式 | 职责说明 |
| :--- | :--- | :--- |
| `acquisition_plan.json` | JSON | 数据采集实施方案：标的池、复权口径、时段、采样频率、限频配置与配额水位。 |
| `dataset_manifest.json` | JSON | 数据集快照清单：各数据源版本、SHA-256 哈希指纹、行数规模与环境信息。 |
| `quality_findings.md` | Markdown | 微观数据审计报告：5m-60m 跨周期对账结果、VWAP 越界比例与交易日历连续性验证。 |
| `coverage_report.md` | Markdown | 各标的覆盖度报告：K 线根数、完整度、停牌与缺失统计。 |
| `request_audit.jsonl` | JSON-Lines | 每一次对外部接口（OpenD / SEC）请求的逐页流水：起止时间、耗时、页数、重试与状态。 |

## 3. 纪律要求
- `request_audit.jsonl` 仅做追加写入，禁止篡改或覆盖过去的请求记录；
- 每次发布新模型训练集前，必须更新并固化 `dataset_manifest.json`，确保实验可复现。
