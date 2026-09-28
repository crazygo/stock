# R03M backlog（实施前）

| 优先级 | 实际缺口 / 明确产物 | 验收或拒绝 |
|---|---|---|
| P0 | 独立轻入口 `prepare_group_only`，新目录冻结本文四件、源码字节、223 来源 SHA、旧 dataset/训练 identity | 不调用硬依赖未完成采集的 R03 prepare；旧源前后哈希不变 |
| P0 | 按 v8 实际两条分类生成路径从真实 bars 重算，保存逐成员 window/最后使用日/可用时间/动作过滤依据 | 不给旧 JSON 批量补日期；缺历史/完整性/动作证据明确不分类，不伪造 receipt |
| P0 | 108 candidate peers + QQQ 独立 benchmark，重建每个旧 key 的 group/group_seq | GOOG/GOOGL issuer 去重、六周代表日与因果检查保留；全池无缩减 |
| P0 | 全键强审计：14,424 个 sample_id/顺序/监督 keys 完全相同，x5/x60/xday/y 与受保护行字段逐行全等 | 不能只比较共同交集；任何 own/标签/entry/terminal 改变立即拒绝 |
| P0 | 分组覆盖、分类标签、事实值、每槽/每通道差异及原因报告 | 分清真实 source 重算差异、数值/动作口径、实现错误；不调分类阈值追求全等 |
| P1-A | 若群张量也全等：版本化 metadata 与可核查的模型 schema 兼容声明 | 全键模型有效输入/九目标复放通过；不重训、不重选阈值、不称效果提升 |
| P1-B | 若群张量不同：先交三个模型 Astra 各自复盘，再另登记固定 recipe/seed/keys 的匹配 refit | 不在本登记下自动拟合；无群旧模型作为不变对照 |
| P1 | 将新 metadata 接真实 feature-only 路径，验证 cutoff/接收限制/六周依赖，逐路线静态输出配对 | 历史 fixture 不能 G2；新 schema/旧 checkpoint 不得未经证书或训练来源握手直接接通 |

轻入口复用点：`prepare._compare_array/audit_2026_pairs`、v6/v9 `_group_map/_group_state` 及既有分类器。现 `audit_2026_pairs` 不拒绝 own 改变或 old_only/new_only，必须加本轮强门禁；`baseline.validate_common_keys` 的交集检查也不等于全旧 keys 检查。

禁止直接运行 `vd.build` 后再挑交集：它会重建 own、标签、样本过滤，混入扩量 builder 的其他差异。旧 own/y NPY 字节应直接保留；只有 group/group_seq 可重算。若 group_valid_count 改变，可更新该派生行列并单独审计，其他行列保持相同；训练监督、权重/先验历史 IDs 不得随之变化。

首轮复盘只作三选一：A 全等→兼容证明；B 可解释群差异→另登记 refit；C 非群变化/来源不足/时序不合法→停止、修实现或回 R03 数据依赖。计算超时先定位缓存/内存，不删证券、不截样本、不关闭检查。
