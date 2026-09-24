# 05 · 实施路线与验收

## M0 · 本次已建立

- 目标、默认决策、规范、数据契约、特征清单、验证和实施文档。
- 标准库可运行的时间/标签/特征/split/B0骨架；合成样本到实验产物的端到端 smoke。
- 边界测试：开盘前信号拒绝、时间可用性、未来追加不变、延迟入场、跨周末/半日/DST分钟、未成熟与缺失标签、purge。
- 根 README 和根 AGENTS 的新工程路由，澄清历史达成率不可直接实时使用。

M0 完成只代表工程基础，不等于模型训练完成。

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

## M2 · 可复算样本与标签

实现 `build_dataset(manifest, config)`：只读快照 → PIT资格与日历时点 → 特征前缀 → 入场代理 → 标签。落盘分离 features/outcomes/exclusions；sample_id稳定且唯一，包含策略版本/股票/decision。传递 `price_basis`、来源时间、输入快照 lineage。

先实现 Gap/前两小时/滞后1/3/5/10/20日，再添加 breadth/relative/cohort；每个特征族独立开关，缺失用显式原因。输出无模型分小时基准率、覆盖和 MFE/MAE 分布。抽样人工核验正、负、未成熟、缺条和边界例。追加/篡改未来bar不能改变历史特征。

## M3 · 基准与首个真实训练器

建立隔离环境和依赖锁；脚手架当前无需第三方依赖，并未安装 sklearn。训练器接口建议：

```text
build_features(snapshot, decisions, feature_schema) -> feature_rows + exclusions
build_labels(snapshot, entries, calendar, evaluation_as_of) -> outcome_rows
make_folds(rows, split_plan) -> train/calibration/validation IDs + purge reasons
fit(train_X, train_y, train_config) -> model_artifact
predict(model_artifact, feature_rows) -> timestamped predictions
evaluate(predictions, mature_outcomes, preregistration) -> metrics + audit + report
```

训练函数不访问未来 outcomes，也不调用行情API；predict接口没有标签参数。先做B0/B1与正则逻辑回归，管线封装训练内填补/缩放和时间校准；保存实际model/schema/feature顺序。只有稳定增量后增加树模型和cohort消融。固定模型概率的前缀不变测试此时补齐。

## M4 · 冻结与独立评估

填写实验登记中所有日期和数值门槛；交付逐条样本外预测、校准、误报/漏报/覆盖/排序与风险报告、时间块区间和负结果。既有数据若已暴露，启动新时期只读纸面记录；到未来标签成熟后再评估，不能用已有历史冒充新测试。

## M5 · 只读产品接入（后续工作）

11:30主结果与后续更新，显示股票、参考价格时间、+5%概率、另行验证后的风险预测、信号年龄和数据质量。保留历史版本，陈旧或缺数据明确 unavailable。用户自己判断并下单。无自动交易接口。

## 每次交付核对

- [ ] 已读本工程规范；变更仍匹配用户的可执行时点和目标。
- [ ] 语义变更升版本，实验登记先于结果，曝光历史已记录。
- [ ] 真实数据/合成数据、已实现/待实现、预测/事后标签标识准确。
- [ ] 因果性、数据覆盖、标签成熟和时间切分检查通过。
- [ ] 报告包含分母、排除原因、未成熟数、失败和不确定性。
- [ ] 可复算命令、产物哈希、逐条预测和验证证据随交付提供。
- [ ] 保留无关WIP，未改旧研究结果，未自动下单或上传。
