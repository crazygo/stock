# 05 · 实施路线与验收

## M0 · 本次已建立

- 目标、默认决策、规范、数据契约、特征清单、验证和实施文档。
- 标准库可运行的时间/标签/特征/split/B0骨架；合成样本到实验产物的端到端 smoke。
- 边界测试：开盘前信号拒绝、时间可用性、未来追加不变、延迟入场、跨周末/半日/DST分钟、未成熟与缺失标签、purge。
- 根 README 和根 AGENTS 的新工程路由，澄清历史达成率不可直接实时使用。

M0 完成只代表工程基础，不等于模型训练完成。

本次补充 `market_dual_track_v2` 文档方案：仅行情、HAB 主方案、LightGBM/小型 TCN 两路线。实现范围按 [07 · 行情双路线训练方案](07_model_training_plan.md)执行，模型与完整新特征仍待 M2–M4 交付。

## M1 · 下一位 coder 应先做：数据就绪审计

输入：显式选定本地行情文件和候选名单，阅读数据获取 skill 后按需补数据。此阶段不训练、不自动选出“最佳策略”。

交付：

1. `dataset_manifest.json`：源文件哈希、范围、提供方、粒度、复权/公司行动版本、时区和到达时间质量。
2. `universe_membership`：带历史生效/可用时间的候选与benchmark角色；获取不到则报告缺口，探索另用明确标注的静态池协议。
3. 官方日历 `sessions`：交易日、常规/半日开收盘、覆盖到最晚标签终点；不要用股票bar来生成日历。
4. 独立的只读数据适配器：规范化到 `Bar`，不在 import 或训练中联网。60m仅用于适合的历史特征，5m用于entry和labels，覆盖不足不能继续假装完整。
5. `coverage_report.md/json`：按symbol/date/granularity统计完整率、缺失、停牌、重复、晚到、混复权、无历史资格、新上市回看不足、标签未成熟；展示被排除分母。
6. `exposure_registry.md`：过去查看/调参过的历史范围；提出真实可用的development和未来独立验证方案。

验收：抽查普通日、周末/假日、半日、夏冬时、公司行动、迟到数据；给出供应商bar起止语义证据。QQQ/行业相对特征必须有同步基准行情。**数据不够就是这一步的明确结果。**

采集范围以 [06 · 数据交接单](06_data_acquisition_feasibility.md)的本轮行情要求为准：保留盘后/夜盘/盘前，分别审计覆盖及资格。事件数据为后续预留，不阻塞本轮。报告数据支持哪些 H/HA/HB/HAB 组合及粒度，不能把只有小时线标为细粒度 TCN 或 5m 执行数据已就绪。

## M2 · 可复算样本与标签

实现 `build_dataset(manifest, config)`：只读快照 → PIT资格与日历时点 → 特征前缀 → 入场代理 → 标签。落盘分离 features/outcomes/exclusions；sample_id稳定且唯一，包含策略版本/股票/decision。传递 `price_basis`、来源时间、输入快照 lineage。

实现 H 历史背景、A 盘后/夜盘/盘前、B 开盘后前缀/前两小时/最近窗口，共用因果前缀构造 LightGBM 摘要和 TCN 序列视图。保留市场/行业同窗口路径、时段身份、间隔与 mask；无成交、缺失、未覆盖与真实低波动分开。新增 v2 特征 schema，不覆盖旧 v1。各族独立开关并登记模块依赖，cohort/Heat/事件不进入本轮。

输出无模型分小时基准率、覆盖和 MFE/MAE 分布，以及八格比较所需的共同样本 ID/排除清单。抽样人工核验正、负、未成熟、缺条和边界例。追加/篡改未来bar不能改变历史特征；HA 不可使用常规开盘后信息，H 不可使用当日变化。

## M3 · 基准与两条真实训练路线

建立隔离环境和依赖锁；原合成脚手架无需第三方依赖，新训练路线需单独声明并验证 LightGBM/TCN 所需依赖和计算设备。训练器接口建议：

```text
build_features(snapshot, decisions, feature_schema) -> feature_rows + exclusions
build_sequences(snapshot, decisions, feature_schema) -> sequence_rows + masks + exclusions
build_labels(snapshot, entries, calendar, evaluation_as_of) -> outcome_rows
make_folds(rows, split_plan) -> train/calibration/validation IDs + purge reasons
fit(train_X, train_y, train_config) -> model_artifact
predict(model_artifact, feature_rows) -> timestamped predictions
evaluate(predictions, mature_outcomes, preregistration) -> metrics + audit + report
```

训练函数只读取本训练折已经成熟的 outcomes，不调用行情API；predict接口没有标签参数。保留 B0/B1，实现独立 LightGBM 与小型多尺度 TCN 训练器，共用样本、标签、时间折和评分接口。训练内选择窗口、填补/缩放、有限超参、early stopping 与时间校准，保存实际 model/schema/摘要列顺序或序列通道顺序。固定模型概率的前缀不变测试此时补齐。

执行 H/HA/HB/HAB × 两模型八格计划，以 HAB 为主方案；自动搜索前冻结各路线候选、trial/算时上限、种子和停止规则，保留全部实验记录。TCN 第一版为分支编码后拼接＋小型 MLP，不同时扩展多种新架构。

## M4 · 冻结与独立评估

填写实验登记中所有日期和数值门槛；交付逐条样本外预测、校准、误报/漏报/覆盖/排序与风险报告、时间块区间和负结果。既有数据若已暴露，启动新时期只读纸面记录；到未来标签成熟后再评估，不能用已有历史冒充新测试。

八格报告同时解释 A、B 的增量、组合价值及两种表示的差别，报告全池推理时延、训练成本和跨时期稳定性。按预登记规则选型；表现接近优先 LightGBM。集成仅在误差互补且新的独立评估证明改善时进入后续计划。

## M5 · 只读产品接入（后续工作）

11:30主结果与后续更新，显示股票、参考价格时间、+5%概率、另行验证后的风险预测、信号年龄和数据质量。保留历史版本，陈旧或缺数据明确 unavailable。用户自己判断并下单。无自动交易接口。

## 每次交付核对

- [ ] 已读本工程规范；变更仍匹配用户的可执行时点和目标。
- [ ] 语义变更升版本，实验登记先于结果，曝光历史已记录。
- [ ] 真实数据/合成数据、已实现/待实现、预测/事后标签标识准确。
- [ ] 因果性、数据覆盖、标签成熟和时间切分检查通过。
- [ ] H/A/B 命名及边界统一；双路线信息范围与时间折可比，八格与失败 trial 均有记录。
- [ ] 报告包含分母、排除原因、未成熟数、失败和不确定性。
- [ ] 可复算命令、产物哈希、逐条预测和验证证据随交付提供。
- [ ] 保留无关WIP，未改旧研究结果，未自动下单或上传。
