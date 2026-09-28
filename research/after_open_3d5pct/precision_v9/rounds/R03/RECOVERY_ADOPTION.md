# R03 独立恢复器采用登记与原失败复盘

登记日：2026-09-27；状态：方案采用登记完成，等待根代理核验后启动；本登记任务未运行采集、prepare或训练，未修改代码/旧run。目标仍是五路线至少三条各三群buy precision≥90%及供给要求；当前0路线通过。

## 实际状态与证据

- 根代理执行会话71777确认原采集在2026-09-27 11:12:08Z退出1，错误为 `normalize_sessions` 访问空表缺失 `start_at` 引发 AttributeError。本次未重新运行旧脚本；该进程退出/异常证据来自根代理执行结果。
- 只读 `runs/precision_v9_20260927/R03/progress.json`：`status=running`、305/545。running是异常退出留下的陈旧字段，不表示任务仍活跃；旧文件保持原样。
- 实际306个分片目录、305份result与bars。305份均声明分页完成、failure=null、rth_complete=true，均有audit；本次逐文件重算bars SHA全部匹配（合计70,672,613 bytes）。这项检查不替代恢复器对canonical schema/质量审计的重新核验。
- 最后完成的是HON/2025-12。HONA/2025-08只有 `audit.json` 与 `raw_001.parquet`，没有bars/result；不是只有audit，也不算第306个完成分片。
- HONA审计：R2键 `us_5m/HONA/2025.parquet` 未找到；OpenD第1页第1次请求于11:12:08.132120Z发出，11:12:08.682012Z收到，`ok=true, rows=0, has_more=false`。只能证明供应商成功返回零行，不能推断上市日、停牌、无成交或负标签。
- 冻结范围维持109证券×2025-08至12共545片、5m ALL/NONE、本地→R2→OpenD、no_upload=true。尚余240片需要处理，其中HONA/2025-08是旧未提交分片，其余239片尚无完成结果。
- `acquisition_recovery_review.md`报告独立恢复器4项离线测试；`RECOVERY_PREPARE_CODE_REVIEW.md`报告prepare适配与恢复联合16项测试通过。本次核对文档和源码指纹，没有重复执行测试；这些证据不等于真实恢复成功、数据可训练或模型有效。
- 指定新目录：`research/after_open_3d5pct/runs/precision_v9_20260927/R03_recovery_v1`。登记时该目录不存在；原 `R03` 只读保留。

## 短 backlog

| 顺序 | 可证伪假设/待办 | 验收 |
|---|---|---|
| 1 | 独立恢复器能识别并继承原305完成分片，避免重复请求和证据改写 | 按原登记/源码快照、result/audit/bars SHA与质量重新验证；新inherited记录及链接指向原同证券同月份 |
| 2 | canonical空成功处理能越过HONA零行边界，保留未知缺口并推进其他分片 | 显式retry旧未完成；空返回写完整canonical零行表、empty_success与unknown标记，rth_complete=false；失败/待处理不得记完成 |
| 3 | prepare新门禁能区分请求完成、行情覆盖和样本资格 | 545片全检查；coverage单列空月与未知缺口；旧2026配对标签/entry/terminal不变；实际manifest验收后另登记基线训练 |

## 方案矩阵与选择

| 方案 | 305片复用 | 空返回/证据保留 | 失败与成本 | 选择 |
|---|---|---|---|---|
| 原目录原脚本直接重跑 | 可能复用 | 未修空表路径会再失败；原目录已有未完成分片 | 不能消除已确认原因，易混淆新旧代码证据 | 不采用 |
| 独立恢复器+新输出+显式retry | 校验后继承原305片 | 新canonical空表标未知，原零行页/audit不覆盖 | 保留分页失败、有限重试和时间预算；只补剩余工作 | **采用** |
| 丢弃旧结果，全545片重新采集 | 不复用 | 可得到新快照，但丢失复用价值 | 请求/耗时更高，不能恢复历史实时receipt，也无额外模型独立证据 | 不采用 |

本次只改变恢复路径和成功空表的工程表示，不改变股票池、时间范围、标签、模型路线或训练选择。预计继承305片是经当前核对的候选结果；正式运行仍须逐片验真，不能硬编码305当完成。

## 根代理核验后的单次启动参数（本任务未执行）

```bash
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.precision_v9.acquire_recovery \
  --source research/after_open_3d5pct/runs/precision_v9_20260927/R03 \
  --output research/after_open_3d5pct/runs/precision_v9_20260927/R03_recovery_v1 \
  --max-minutes 180 --retry-source-incomplete
```

只允许本次显式重试旧未完成分片；新输出的失败片不能隐式无限重试。保持单进程锁、每月最多30页、每页最多3次；不同时启动旧采集或其他冲击同一额度的任务，不推送R2、不接交易。

## 验收与回退条件

1. 运行前核对本页及下列源码/原登记指纹；运行身份记录新代码、导入模块、原登记和原代码快照。恢复器初始化只登记哈希，不自动保存新源码字节：由根代理在新目录初始化后保存 `source_snapshot/`，包含acquire_recovery及identity列出的全部imports、采用文档，并逐项核对SHA；无需改恢复器或另起开发。指纹不符先停止并重新审查/登记，不修改登记去迁就已运行代码。
2. 原R03的登记、progress、HONA审计和所有被继承分片保持不变；继承错链/源哈希变动/质量不一致必须拒绝，保留错误。不能将可疑片静默重新下载并冒充原证据。
3. 新summary必须分列完整非空、成功空、unresolved_failure、pending；只有全545片请求结束且无失败才可叫acquisition_complete。成功空片仍有未知覆盖缺口，不宣称全部RTH行情完整。
4. 超时、额度/网络错误、页数上限或新异常时保留新审计与checkpoint，状态为未完成/失败；不删除失败、不改旧run、不自动扩股票范围或重试预算。若新输出存在未提交片，按恢复器要求另建后继输出并重新登记。
5. 采集完成后才运行prepare；拒绝缺片、未完成分页、失败、bars/audit/result不一致、错误继承链接和未知恢复版本。完整5个月为空但2026有数据可保留缺口拼接；两年均空显式失败，不伪造bar。
6. prepare须通过既有2026配对行的时点/entry/terminal/九标签不变检查，并报告新增历史对输入/群/样本数的影响。通过后才签actual_manifest并另行冻结基线；恢复本身不代表训练通过，更不代表90%目标完成。

## 原失败复盘

根因是“接口RET_OK、分页结束”被当作非空表结构成立：上游空原始DataFrame未生成canonical列，下游直接访问start_at。原先静态审查记录为可能KeyError；本次实际链路报AttributeError，以真实异常为准。数据来源成功、标准化完成、分片提交是三个状态，不能合并。

已完成305片和零行原始页/接收时刻均有价值，应保留。独立恢复器把空成功转换为带schema的0行输出与未知覆盖状态；prepare门禁补齐来源核验。16项离线测试只降低此路径的工程风险，仍需真实运行结果与数据验收；不得用HONA名称或空表推测上市状态。

## 登记时指纹

下表路径中，R03指 `runs/precision_v9_20260927/R03`，其余文件相对precision_v9。

| 文件 | SHA-256 |
|---|---|
| R03/registration.json | `5eabd78e764f31dadf87d36d76eb5ee131a597379a1781deaee122a083b78789` |
| R03/progress.json | `a69cca599e83e2eb9c78420fd3ae2105d73f1ceebcbc911c05baa9f210bdc537` |
| R03/parts/HONA/2025-08/audit.json | `3a24469b1150d62faaa713cc42bc5728d012adbf82009d2044223b98bb5906f7` |
| acquire_recovery.py | `6b4e056492dc05dad7a6a56b358dc6dbcb463ee91b7c6e621bd32a831f735627` |
| prepare.py | `0dffa738a804bafce87b962352878d7901b5600f88869384840cf7476969cca1` |
| acquisition_recovery_review.md | `21a0345995b2f0724688a55572589af7d59c5f0765f60bc38c050be83d27a9cd` |
| RECOVERY_PREPARE_CODE_REVIEW.md | `714d8fba281363b1ef9b7d53ec66174173bb9d209de5edee1f0403b106ad8aae` |
| test_data.py | `2dbb756af5bb59c3745a0664b01810b42b1cb7c82f17516f85ce766701f1deb5` |
| test_acquire_recovery.py | `2df17dd64600063abdf3c084a4108ea33688c80d9857651ff39017e065ccbca4` |
