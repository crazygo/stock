# 前向证据架构矩阵

2026-09-27。下表是基于实际代码的工程比较，尚未实现/压测；没有任何一项能保证90%效果或增加真实机会供给。开发历史再漂亮也不能替代新时期记录。

| 方案 | 正确分母/五路线 | 时点与择日证据 | 跨run/跨期 | 数据风险 | 成本 | 本轮选择 |
|---|---|---|---|---|---|---|
| A 直接适配v3 run_once和evaluator | 需重写模型加载、九输出、三群trigger、buy集合和成熟分母；目前评全部预测 | 已有快照/过期机制可参考；水位只有最新bar真实receipt证明不够 | v3没有v9持久cooldown/cohort官方日账，需追加 | 容易把HAB旧制品、六时点、研究candidate及成熟touch_rate误作v9证据 | 初看低，迁移耦合高 | 不选为主；只复用经测试的小函数/设计 |
| B 独立v9 feature-only + ledger | 明确route/schema/阈值/三群；全部buy入账后再补标签 | 严格收到版本、实际发布时间、缺失/过期/拒绝全保留 | 事务幂等、按route持久cooldown、官方session全分母、冻结cohort | 新接口需完整E2E，日历/版本/先验必须冻结；缺receipt就拒绝 | 中等且边界清楚；模型训练不重写 | **选择** |
| C 仅补写历史预测/调用data.build | 模型输出可重放；先筛全未来标签天然丢未成熟/缺失buy | 无当时发布与收到证据；不能恢复真正前向 | 每次冷启动易重复计数；补写越多不等于独立期越长 | 2026全部exposed，回填2025仍无法制造当时系统记录 | 最低，但达不到目标 | 只允许development诊断，永远不能计最终通过 |

## B 的最小组成

1. `freeze.json`固定模型/变换/校准、九任务顺序、raw或processed决策列、每群阈值与成员、动态群规则与版本可用性、先验更新、日历、数据口径、起止sessions、发布deadline及全部SHA-256。
2. `features_asof`先按实际receipt挑版本，再构造与训练相同的tensor/表格；仅预测函数不读取outcomes。成熟历史先验由独立as-of库按冻结规则提供。
3. `forward.sqlite`仅追加事件（单写者事务、unique id、禁止UPDATE/DELETE），按route分事件序列与cooldown；报表/状态是可重建投影。每次运行保存不可变快照、预测与报告文件；重复请求返回原记录。
4. `mature`只给已存在signal_id追加九目标outcome版本；评估器从全部issued buy左连接标签，绝不先inner join成熟数据再算precision。
5. `sessions`预登记官方日历全集；停机由对账时追加missing_run事件并写实际发现时间，不伪造当日运行。供给的总session数独立于数据文件和有信号日期。

SQLite是本地研究账本建议，不是部署要求。每route一套独立模型/动作命名空间；可用同一物理数据库保证事务，但不得把五模型投票当三个模型达标。交叉路线信号/误报另报重叠，不称统计独立。

## 复用边界

- 可参考 `hourly_v3_snapshot.visible_versions` 的“先时间过滤后选择版本”次序，须加全部依赖非空receipt和覆盖审计，不能直接复用宽松缺失逻辑或其联网刷新入口。
- 可复用 `timeaxis.add_regular_minutes`，前提是带版本官方日历覆盖到最晚标签/cooldown终点；日历未来不够只记pending_calendar，不能从有bar日期推断休市。
- 复用 `precision_v9/evaluate.select_signals` 的阈值trigger/排序/union语义，但把busy从临时字典改为事务内读取已发事件，保留原时间边界。
- 复用实际冻结路线的推理/校准/单调投影数值函数，不能从 `baseline.run` 或 `focus_v8.load_data` 的训练带标签入口直接推理。
- 独立保存v9日志、cohort及报告；v3记录只可列为旧工程证据，不迁移为v9 buy。

## 决策的失效条件

若不能从制品精确恢复训练输入/变换，则G1失败，保留研究预测并补制品接口；若不能取得全部输入的真实截止前receipt，G2失败；若账本无法证明全部issued buy与完整官方session分母，G3失败。三者都不能靠降低阈值或删失败行修复。任何语义变更另版本与新cohort，保留旧期结论。
