# v9 前向账本、标签与评价代码复核（2026-09-27）

状态：**仅本地工程 fixture 验证**。没有真实前向预测、真实接收/发布、正式 cohort、行情下载、服务、订单或上传。本实现仅新增 `forward/ledger.py`、`labels.py`、`evaluate.py` 及三个定向测试；未改活动采集、既有 run、模型训练或共享 feature/inference 文件。

## 本次可核验的工程契约

- `register_cohort` 在首个官方 session 前对**完整 CohortSpec** 做哈希，冻结 ≥60 个连续官方日期、完整日历、期末及五条路线各自的模型版本、行动列、阈值和模型/schema/calibration/源码 SHA；显式拒绝另带 `registration` 内层副本的包装对象，`issue_once` 与 `evaluate_cohort` 均执行同一约束。同 cohort 不能中途换配方；新 cohort 仍承接同路线历史禁重。live 模式另要求日历已核验、日历/协议哈希及 live 标记，并拒绝非 live lineage。现有 R03 日历只到 2026-09-24，不能据它开启未来连续 60-session cohort。fixture 注入时钟的 DB 与 live DB 模式互斥；live 拒绝注入时钟或回填时刻，发布与账本记录取实际系统时钟。
- `issue_once` 逐候选检查九目标、预测 JSON 哈希、完整 lineage/receipt、冻结阈值、模型/schema/calibration/源码/输入快照实际文件哈希和 11:30:00–11:30:30 ET 实际发布时间窗；所有输入 `available_at/received_at` 还必须不晚于**实际发布时间**。单个事务先写预测，再写 buy 或拒绝；提交后才返回结果。`route/cohort/cutoff/symbol/type` 唯一，重试需预测完全相同。禁重状态由历史已发 buy 决定，跨 run、重启、cohort、模型版本保持 1170 RTH 分钟；提前触及不解锁。交叉成员只计实际触发阈值的群，联合 buy 一条。晚到/过早/过期/缺证据/坏 hash 保留拒绝；落盘失败整笔回滚。`run_empty` 与闭市后 `reconcile_session` 的 `missing_run` 分开。
- `evaluate_candidate` 是可用于**任意候选**的纯九目标 helper；从 11:35 开始的 5m bar Open 入场，按官方日历取完整 78/234/390 根 RTH。完整窗口及来源可用时间未到前不输出 TP/FP、entry 或 first-touch。公司行动/停牌覆盖必须有显式验证区间与来源 SHA；缺证明为 `evidence_unknown`，不能默认空事件。缺 bar、停牌、split 价单位、坏 bar 分别保留未解决状态；完整后记录 entry bar、阈值价、first-touch 区间、标签结束/可用时刻、grid/source hash。High 首触只是价格代理，不是成交证明。`mature_signals` 仅将已有 BUY 的结果追加为 outcome 版本，修订以 `supersedes` 指向上一事件，不删原 buy。
- `evaluate_cohort` 要求 ledger cut 已真实存在，从**全部已发 buy**左连接最新 outcome；每群每目标保留 `N=TP+FP+U`、已知 precision 与 `[TP/N,(TP+U)/N]`，零信号 precision=null。供给分母始终是冻结的全部官方 session（含无信号、停机、半日）；展示前后半段、15 格共用日期抽样的 5/10-session block 区间，零信号抽样保持 null。主目标只有期末、无 U/违规、完整日账、≥30 成熟信号、≥20 日期、每 5 session ≥1 条、点 precision≥90%、live 证据均满足才能通过；90%点值不是置信下界。后发现的公开违规追加 `protocol_violation`，原 buy 继续占分母。报告可 `seal_report` 入账并核对字节。

## 离线验收

运行 `research/after_open_3d5pct/.venv/bin/python -m unittest -q research.after_open_3d5pct.precision_v9.forward.test_ledger research.after_open_3d5pct.precision_v9.forward.test_labels research.after_open_3d5pct.precision_v9.forward.test_evaluate`：**23 项通过**。fixture 覆盖并发/重启/跨 cohort 与版本、同 cohort 换版本/阈值/SHA 及包装对象外层篡改拒绝、半日/假日/跨年禁重、成功与失败落盘、11:30前过早发布/发布时尚未收到输入/截止后发布、晚 receipt、坏预测哈希/未来标签、九目标完整成熟与 first-touch、未知公司行动覆盖/缺失/停牌/split/坏来源、结果修订、`TP=2/FP=1/pending=1/missing=1` 的 N=5/已知 2/3/区间[2/5,4/5]、60 天分母、期中与单条全胜门禁、未来 ledger cut 拒绝、模型与报告字节篡改。

## 尚未满足的门槛

G1 只验证账本消费接口的 fixture，五路线真实 checkpoint 推理重放由共享 inference 工程另验；G2 没有真实逐依赖 receipt、真实发布和已核准未来日历；G3 没有真实冻结 cohort 或独立新时期效果。模型所需成熟先验必须来自**全部合格历史候选决策**，本模块的 BUY-only outcome 不可当作先验库；纯 `evaluate_candidate` 已可复用，但全候选 store 更新与特征侧集成尚未交付。真实运行前还需把冻结 spec/route/cohort 实例、日历来源与五路线制品接通，并完成端到端实际接收验收。上述 fixture 结果不能称为三路线 90% 达标。
