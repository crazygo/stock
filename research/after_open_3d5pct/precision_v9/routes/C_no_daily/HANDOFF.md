# C 无日级 · 给 root / Sol 的执行交接

研究方案/复盘由Astra负责，纯代码由root调度Sol。本子任务只写四个入口；以下P1为P0结果之后的待登记方案，不是已启动实验。禁止修改运行中的`acquire_batch.py/acquire_pilot.py`或覆盖旧WIP/制品。

## P0共享runner接口

1. 显式接收 `dataset_manifest, calendar_path, output, route=C_no_daily, arm, split_key_map, seed=3566`；模型从v8 final_selection取原配方，仅 `balance/regularization/calibration/support=true`，不能载入R01 capacity。
2. 数据层可供共享审计读取xday；**该路线forward和任何新增特征都不得读取本股xday数值**。5m/60m、六周群分支保留；group_support仅fit正权重行决定，按类型至少10个有效fit日期，不按eval激活。
3. P0a限制旧fit监督keys；P0b仅扩大早期fit，两臂旧行张量逐值一致。tune/cal/eval旧新共同合法keys固定；必要时在旧快照相同交集重拟合桥接。记录所有差集及原因，拒绝隐式按结果删行。
4. 固定旧三折边界及19/20/18官方评价sessions；同日同侧，九任务最长窗口end/available purge。原32epoch/patience5/batch256/thread1、seed3566、九头BCE、AdamW lr=.001/weight_decay=.05、head dropout=.25、balance配方不变。
5. 保留类别、权重算法、Platt及单调投影；各自cal拟合参数、raw主分数按旧14阈值网格选择，eval只评价。保存每臂实际参数与阈值，不能沿用新概率却假装旧阈值已验收。
6. 输入、标签及1170分钟冷却使用同一官方日历；11:35 Open对应原始结束标记11:40的5m bar。群成员as-of、最大146日依赖、半日及公司行动变化须在拟合前审计。

P0完成后研究代理先复盘，登记P1实际数据/schema/code哈希、行数、比较臂、成本上限，再执行下面的单一机制。

## P1数据接口：peer_intraday_delta_v1

建议独立只读函数：`build_peer_intraday(snapshot, rows, weekly_memberships, calendar) -> endpoint[N,27], delta[N,27], audit`。不得调用行情API；不能从group_seq的周状态伪造10:30数据。原group_seq张量不变。

- 固定 `t0=10:30,t1=11:30 ET`。使用当周决策前已知的五种市场群成员，排除本股同issuer、GOOG/GOOGL等重复发行人；不新加操作行业成员输入。
- 对每个群槽，eligible集合来自同一个冻结成员版本；paired集合是两时点均有完整09:30→对应时点前缀、有效正价且截至决策可取得数据的共同成员。缺口不前填；记录eligible/paired数量、排除原因、成员哈希和最晚source available。
- `C_p(t)`取结束于t的5m Close，`O_p`取09:30开盘bar Open。`r_p(a,b)=log(C_p(b)/C_p(a))`，`R_p(t)=log(C_p(t)/O_p)`。所有输入bar_end≤11:30、available_at≤11:30:30；t0切片不混入后续bar。
- D每个槽5字段：`100*median_p r_p(t0,t1)`、`mean_p[1(R_p(t1)>0)-1(R_p(t0)>0)]`、`100*(r_own(t0,t1)-median_p r_p(t0,t1))`、`paired_count/eligible_count`、`valid`。
- E每个槽5字段：`100*median_p R_p(t1)`、`mean_p 1(R_p(t1)>0)`、`100*(R_own(t1)-median_p R_p(t1))`、相同coverage、相同valid。两臂全部p来自相同paired集合，避免把成员变动归因给盘中变化。
- 五槽共25维；QQQ另2维：D为`100*(r_own(t0,t1)-r_QQQ(t0,t1)), valid`；E为`100*(R_own(t1)-R_QQQ(t1)), valid`，QQQ两臂也用相同双时点完整性条件。
- 无eligible/无paired或本股所需前缀缺失时，槽全零且valid=0，原因另存；无效值不得冒充真实平稳。有效槽即使覆盖低也保留coverage，不新增效果驱动的覆盖筛选。再乘原fit冻结group_support；QQQ遵守其独立valid。
- 新增特征只使用5m短前缀和既有群成员；长期群分组仍可存在，不能将此方案说成完全无长期信息。无未来标签、入场价、最终cohort结果或本股xday输入。

**现有schema细节**：前五槽ch4是peer return IQR，QQQ槽ch4实际是QQQ rvol；`prepare.compare_datasets`通用标签不能替代逐槽源码语义。新增E/D直接读行情，不把QQQ ch4当离散度或做同类群差分；审计标签需区分槽类型。

## 模型接口与匹配对照

- `NoDailyPeerResidual(incumbent, mode=endpoint|delta)`：`logits=base_logits+Linear(27,9,bias=False)(u)`，新增243参数，权重全零初始化；原5m/60m/群GRU/门控及九头结构、support、loss不变。九输出一起训练，投影/校准顺序不变。
- I/E/D共享相同初始base权重及训练顺序；新层构造不得推进其他层随机初始化。零残差时E/D raw应与相同权重I一致；全无效u时残差严格零。E/D都重新独立拟合，不把incumbent训练结果偷偷用作某臂额外warm start。
- E/D各2开发折共4拟合；先按 [BACKLOG](BACKLOG.md) 双对照门槛写决定，再各1次diagnostic。I可复用同快照同seed P0，不复用不同交集分数。固定单窗口、无参数搜索。

## 必要测试与交付

1. 两组同伴11:30终点相同、10:30不同的合成路径：E相同，D不同；证明新增能力不是周差分或重新缩放终点。日期/成员顺序置换不得改变聚合。
2. 共同paired集合：只在t1出现/迟到/缺失一段的同伴不能进入任一臂；一个issuer多ticker不重复计数，排除本股，未来生效成员不可入群。QQQ单独公式对账。
3. 固定快照/模型后追加或改写cutoff之后数据、决策后到达bar、未来成员及未成熟标签，不改变过去E/D/raw；改写本股xday也不改变路线预测。
4. 所有mask、support被屏蔽时不产生伪观测；E/D coverage/valid逐值相等，零初始化与I一致，新增头有非零可计算梯度；九输出有限、checkpoint重放误差≤1e-7。
5. P0a/b同旧行输入一致；旧新entry/label/terminal配对、跨年/DST/半日1170分钟、基于时间而非提前成功的冷却及交叉群联合计数回归。

每臂保存 `registration, source/schema/split hashes, model, trace, calibration, fit/tune/cal/eval九目标raw/p, selection完整曲线, precision, signals, peer_audit, paired_metrics, decision_before_diagnostic`。报告每群TP/FP/n、供给、日期、拒绝/未成熟及缺失；记录参数量、耗时、内存、早停。全部2026为development，工程测试通过不代表90%能力或真实成交。
