# 三群买入信号 precision：五路线持续迭代

目标来自用户：五个模型中至少三个模型，各自在计算与通信芯片、光通信产业链、数据与存储三个操作群，发出的买入信号至少 90% 实际达成；每群平均每周至少一次机会。定义、去重、样本门槛和历史/前向证据边界见 [PROTOCOL.md](PROTOCOL.md)。

当前没有模型通过完整验收。R00 暴露了校准段高命中向后续时段迁移失败；R01 五个候选改动全部回退。R02 验证历史行情能取得，并修复新适配层的午夜 session 分类。R03 原采集在305/545片后因HONA空返回标准化异常退出；新本地复用恢复版已通过独立离线审查并冻结。当前唯一等待会话为74618，仍等另一任务释放共享锁；原22976已在确认未采集后退出。详见[接管记录](rounds/R03/local_reuse/ROOT_HANDOVER.md)；数据验收后才登记、执行五路线原配方匹配基线。

已采用[本地原始行情复用执行版本](rounds/R03/local_reuse/RUN_ADOPTION.md)：原305片优先，另一采集任务新增月片必须经过原始分页、来源与质量验证；上一盘点剩240片尚未本地就绪，执行时重新核查。545月份要求有可审计状态，真实空月和缺口继续单列unknown。28项套件与3项独立检查通过仅证明工程门禁，新run尚未采集。

## 分工与上下文边界

用户于 2026-09-27 指定方案/复盘使用 GPT-6 Astra xhigh，纯代码使用 GPT-6 Sol xhigh。各模型由独立研究代理维护，根代理负责共同定义、数据版本、调度和验收。当前最多四个并发代理，五路线分批执行；某路线等待不意味着取消。

| 路线 | 独立研究工作区 | 匹配基线之后的待检验机制 |
|---|---|---|
| B 有群 | [B_group](routes/B_group/STATE.md) | 固定同伴集合的 10:30→11:30 盘中变化，保留原树模型作为对照 |
| B 无群 | [B_no_group](routes/B_no_group/STATE.md) | 本股波动状态条件成熟先验与残差树，对照无条件本股先验 |
| C 有群 | [C_group](routes/C_group/STATE.md) | 有界群残差，对照原模型和匹配的零群残差模型 |
| C 无群 | [C_no_group](routes/C_no_group/STATE.md) | 仅 5m 分支的日内跨 patch 残差，对照近等参数的逐点模型 |
| C 无日级 | [C_no_daily](routes/C_no_daily/STATE.md) | 小型同伴盘中增量残差头，对照原模型和同容量终点头 |

五份独立方案均已完成，尚未执行上述新机制。共同前置是 R03：先补齐旧样本的历史输入、保持旧拟合行，再在同一快照增加早期拟合日期。每一步都配对相同 2026 评价行；结果改变瓶颈时先复盘、重写下一登记，不机械执行本表。

B无群已根据原305片的实际覆盖，另登记[部分日级历史实验](routes/B_no_group/parallel_readiness/REGISTRATION_DRAFT.md)：108股与14,424旧键全部保留，仅为来源完整的60候选补长日级输入；106列可变，其余563列不变。这是独立的部分来源实验，不等于完整R03。根[采用](routes/B_no_group/parallel_readiness/ADOPTION.md)当前只授权离线构建、工程门禁及旧模型控制回放；实际manifest经独立审查后再决定两开发折拟合。

每个工作区使用四个短入口：`STATE.md` 记录已证实状态，`BACKLOG.md` 记录下一可证伪假设与依赖，`MATRIX.md` 比较可行方案并定本轮选择，`HANDOFF.md` 给代码代理明确输入/输出/验收。进入实验后，另存不覆盖的轮次登记与 `REVIEW.md`。代码代理不自行更改目标、群成员、拆分、阈值选择或保留标准；研究代理不把未执行方案写成已验证效果。

代理向根代理仅汇报：完成/等待项、每群信号数与命中数、供给、保留或回退、下一动作、证据路径。完整表格和训练日志保留在各自制品中。轮换代理时先读工作区，不重新复制全部对话。

## 轮次证据

| 轮次 | 先登记 | 复盘 | 状态 |
|---|---|---|---|
| R00 既有预测诊断 | [backlog](rounds/R00/BACKLOG.md)、[矩阵](rounds/R00/MATRIX.md) | [复盘](rounds/R00/REVIEW.md) | 无路线通过 |
| R01 每路线一个结构/目标改动 | [backlog](rounds/R01/BACKLOG.md)、[矩阵](rounds/R01/MATRIX.md) | [复盘](rounds/R01/REVIEW.md) | 15 次拟合，五路线全部回退 |
| R02 跨年行情试采 | [backlog](rounds/R02/BACKLOG.md)、[矩阵](rounds/R02/MATRIX.md) | [复盘](rounds/R02/REVIEW.md)、[适配层修正登记](rounds/R02/NORMALIZATION_AMENDMENT.md) | 五证券试采完成 |
| R03 历史扩量与匹配基线 | [backlog](rounds/R03/BACKLOG.md)、[矩阵](rounds/R03/MATRIX.md) | [原失败与恢复采用](rounds/R03/RECOVERY_ADOPTION.md)、[本地复用独立审查](rounds/R03/local_reuse/CODE_INDEPENDENT_REVIEW.md)、[接管](rounds/R03/local_reuse/ROOT_HANDOVER.md) | 原305片保留；新本地复用恢复版等待共享锁；尚未开始新基线训练 |
| R03M 冻结2026群元数据兼容验证 | [状态](rounds/R03M/STATE.md)、[backlog](rounds/R03M/BACKLOG.md)、[矩阵](rounds/R03M/MATRIX.md)、[登记](rounds/R03M/REGISTRATION.md) | [首失败与修复登记](rounds/R03M/FAILURE_REVIEW_AND_RETRY_REGISTRATION.md)、[重建审查](rounds/R03M/GROUP_ONLY_REVIEW.md) | 223源验真，14,424键全部群/本股/标签/先验/监督全等；不重训。模型九输出与证书另验，不改变R03扩量计划 |

R03M数值首run因1套历史eval批次复放超差保留失败；[独立复盘与有限重试登记](rounds/R03M/NUMERIC_FAILURE_REVIEW_AND_RETRY_REGISTRATION.md)已采用。retry1的15套×14,424键旧/新九raw/p、15套原eval按历史批次复放、六固定时点的90组feature-only同单样本配对均零差；Astra独立核验来源与数值通过。[数值证据索引](forward/NUMERIC_EQUIVALENCE_CODE_REVIEW.md)保留首失败及跨批次微小差异，不能称任意批次恒等。正式预测接口G1尚未通过。

[数值证据采用与优先级](rounds/R03M/NUMERIC_EVIDENCE_ADOPTION_AND_PRIORITY.md)决定暂缓六旧样本专用loader例外；优先实现[R03原生工件合同](rounds/R03/native_artifact/ADOPTION.md)，让后续真实训练保留三臂身份并通过正式预测接口验真。当前仅授权工程与已有模型的事后封装对照，不表示30套新模型已经训练。

完整可复现数据/模型制品位于 `../runs/precision_v9_20260927/`，行情和训练缓存不入 Git。原R03 `progress.json`保留305片但running字段已陈旧，须结合进程退出和检查点判断；恢复目录 `R03_recovery_v1`目前只有registration/calendar/source_snapshot。未完成时不把缺失分片写成成功。模型训练需另外登记实际 manifest、样本覆盖和比较集合。

补充共享证据：[R00 模型信号与误报重叠](rounds/R00/CROSS_MODEL_AUDIT.md)、[R03 公司行动来源核对](rounds/R03/ACTION_PROVENANCE_AUDIT.md)、[跨年构建器代码审查](pipeline_code_review.md)。

前向工程已交付五路线静态checkpoint复放、两无群路线feature-only复放及账本fixture；三个带群路线全链路、全候选持久成熟先验与本地适配见 [INTEGRATION_PLAN](forward_readiness/INTEGRATION_PLAN.md)。[独立2026官方日历](forward/calendar_2026_official_v1/CALENDAR_CODE_REVIEW.md)已生成并验真；具体cohort仍需检查完整1950RTH分钟覆盖。G2/G3与30秒全池性能尚未证明，不能计为最终达标。

群输入已修复 QQQ 混入 candidate peers、无效分类/版本、群全候选集合绑定与 PriorStore 封存失败传播问题。最新58项forward测试通过，[Astra独立复盘](forward_readiness/INTEGRATION_REVIEW.md)采用这些限定修复；旧带群真实 G1 仍未通过。[本地适配器](forward/LOCAL_ADAPTER_CODE_REVIEW.md)保存逐候选拒绝和全依赖哈希；真实历史AMD无群路径被账本按缺receipt拒绝，合成N5只证明分母处理。

## 连续迭代规则

1. 先读真实状态和上一轮复盘，更新各自 backlog 与方案矩阵。
2. 只选择一个主要机制，明确固定项、代码需求、执行成本、保留/回退标准。
3. Sol 实现并验证后，冻结代码与数据，再执行登记实验；不覆盖旧制品。
4. Astra 根据结果复盘，区分假设被推翻、证据不足和工程失败，生成下一轮 backlog。
5. 根代理按五模型×三群验收，报告信号/误报重叠；共享行情不等于五份统计独立证据。

R03 同时改变历史长度和数据构建支持。旧 2026 行必须按 `(symbol, session_date)` 配对，分别报告标签、原有短窗口、长历史和群输入的差异；若构建语义发生变化，不把所有效果归因于增加训练日期。当前池回溯、历史接收时间假设和 High 触及代理的限制继续保留。
