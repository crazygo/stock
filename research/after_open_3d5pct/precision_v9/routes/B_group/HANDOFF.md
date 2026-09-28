# B有群 · root / Sol 交接

2026-09-27。本轮只做方案/复盘，无实现或训练。纯代码请root转交 **GPT-6 Sol / xhigh**；方案和复盘由 **GPT-6 Astra / xhigh**。以下任务在root分配范围内执行；B0验收和复盘先于B1。

## 1. 只读诊断接口

`diagnose_route(artifact_paths, dataset_manifest, frozen_selection, calendar) -> new_diagnostic_artifacts`。固定读取R00/B有群及R01/B_group的cal/eval预测与selection，沿用共同selector；不回写旧制品，不在eval选阈值。

输出：cal逐固定阈值的n/TP/FP/日期/供给及四项约束失败原因；raw唯一值/分位数/阈值边距；raw与p分别的pooled/同股AUC、同股配对数及单类股票数；固定信号的sample_id、时点、冷却终点、trigger_groups、raw/p/y、terminal_3d及阈值/无阈值/冷却拒绝原因。FP按股票/日期/因果群版本/覆盖拆分并保留TP；路线间交集交root汇总。必须说明阈值变化后冷却集合可能非嵌套。

## 2. B0匹配基线接口

显式接收新dataset/calendar，不依赖v8写死路径或2026-only日历。使用`focus_v8/run.py::fit_b`旧保留配方；不能经`precision_v9/train.py::CHANGES['B_group']`自动带回已回退类别屏蔽。

登记并保存新数据/源/日历/公司行动/成员/schema哈希、原配方/种子/时间折、各折完整126日比例与群支持、旧新2026行标签/差集/特征差异、四段九目标raw/p/y、校准参数、完整cal曲线、固定eval信号、成本与checkpoint复放。builder语义修复和增加fit日期须分清；可选短fit桥接对照需root另登记。R03采集与builder未验收时不训练。

## 3. B1特征接口草案，B0复盘后冻结

`peer_intraday_change(sample_ids, immutable_source_views, causal_week_memberships, feature_cutoff, decision_at) -> values[n,33], schema, provenance, missing_reasons`。无标签参数、训练内不联网。现有group_seq只有六周11:30截面，不能准确还原10:30，必须有新的as-of sidecar。

沿用当周因果trend15/trend63/trend126/volatility/liquidity五种市场群，排除本发行人、同发行人去重。每槽使用两段都完整/可用的**同一个同伴集合**。每同伴：`r1=log(C10:30/O09:30)`、`r2=log(C11:30/C10:30)`；同时满足bar_end≤cutoff、available_at≤decision，按规范化5m边界和官方日历计算。

每个同伴槽6列，共30列：

1. `median(r2)-median(r1)`。
2. `mean(r2>0)-mean(r1>0)`。
3. `own_r2-median(r2)`。
4. `log1p(paired_observed_issuers)/5`。
5. `paired_observed_issuers/eligible_peer_issuers`。
6. `valid`；至少一名两段合格同伴，缺数值保留NaN/原因，不补零收益。

QQQ另3列：`qqq_r2-qqq_r1`、`own_r2-qqq_r2`、valid。若基础schema仍863列，则候选896列；用命名schema，不靠硬编码位置。原特征、九头、容量、校准/投影和raw动作规则保持。缺sidecar行保留基础资格、候选缺失值可由树处理，使评价行配对；单报覆盖，不私加行动门禁。行业操作群不回填成PIT行业特征，不引入夜盘或群成熟标签。

必要检查：人工可算同伴例、发行人排重和自身排除、两段共同分母、未来/迟到/下一周成员不污染过去、标签不变、模型复放。先保存两个开发折的保留/回退决定，再看已暴露诊断；规则见[BACKLOG](BACKLOG.md)。返回逐群TP/FP/n、precision/供给/日期、raw/p排序、Brier、FP迁移与缺失成本；失败返回具体样本及恢复点，不伪造结果。
