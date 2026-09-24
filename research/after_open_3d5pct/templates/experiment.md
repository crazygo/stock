# 实验登记：填写后再运行

状态：DRAFT（未填齐不得标记 FROZEN / PASSED）

## 身份与假设

- experiment_id / strategy_version / feature_schema_version：
- 登记时间、负责人、Git SHA/dirty patch hash、运行时/依赖锁、随机种子：
- 可证伪假设（相对什么基准、为什么可能有增量）：
- 这次只改变什么；之前失败结果链接：

## 数据与时点

- 数据/日历/股票池/公司行动 manifest 路径与 SHA-256：
- PIT资格与 source available_at 证据；是否只是 assumed：
- 暴露历史范围与原因；未接触 holdout 或前向纸面方案：
- 决策时点、特征截止、人工延迟、entry proxy、1170分钟边界：
- 特征allowlist、缺失规则、候选总数/消融列表：
- 标签成熟/停牌/退市/缺失处理、排除分母：

## 冻结评估计划（全部数值须预填）

- 按ET日期的 train / 内层校准 / walk-forward validation / final holdout 区间：
- purge条件、折数、训练窗口、训练/评估的 as_of：
- B0 / B1 定义；模型和有限超参列表；多重比较处理：
- 概率校准方法、threshold或Top-K规则与选择数据：
- 主指标 Brier；相对B0/B1改善的95%区间下界 >0：
- 最小交易日/时间块/样本数，最大校准误差，最低覆盖，最大误报：
- bootstrap块长、次数、seed；按小时/月份/股票/状态拆分：
- stop/go规则（失败怎么记录，禁止测试集回调哪些参数）：
- 执行延迟/滑点敏感性（与主统计分离）：

## 运行后附录（不得改上面的冻结内容）

- manifest、配置、模型、schema、split IDs、逐条预测/排除记录路径：
- 指标、置信区间、校准图、false positives/misses/coverage：
- 11:30主时点与更新时点的差别；MFE/MAE风险：
- 失败/异常、数据限制、结论层级与不允许宣称的能力：
- 下一步；如果已看holdout，登记为已暴露：
