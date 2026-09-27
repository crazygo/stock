# 实验登记：填写后再运行

状态：DRAFT（未填齐不得标记 FROZEN / PASSED）

## 身份与假设

- experiment_id / strategy_version / feature_schema_version：
- model_plan_version（本轮 `market_dual_track_v2`）/ label_contract_version（沿用 v1）：
- model_family（`lgbm` / `tcn`，基准另标）/ input_set（`H` / `HA` / `HB` / `HAB`）：
- 登记时间、负责人、Git SHA/dirty patch hash、运行时/依赖锁、随机种子：
- 可证伪假设（相对什么基准、为什么可能有增量）：
- 这次只改变什么；之前失败结果链接：

## 数据与时点

- 数据/日历/股票池/公司行动 manifest 路径与 SHA-256：
- PIT资格与 source available_at 证据；是否只是 assumed：
- 暴露历史范围与原因；未接触 holdout 或前向纸面方案：
- 决策时点、特征截止、人工延迟、entry proxy、1170分钟边界：
- 特征allowlist、缺失规则、候选总数/消融列表：
- H/A/B 每个字段的模块依赖和截止时间；相同信息范围的摘要/序列映射：
- 波动/成交实际值与自身历史相对值配对；历史分母、同类时段对齐、零分母/历史不足规则：
- 候选历史长度、分钟粒度、固定前两小时/最近窗口；重采样与价格/量归一化规则：
- 时段身份、间隔、observed/no_trade/missing/not_covered/padding 定义与 mask/池化规则：
- 各格完整资格数/排除分母、八格共同样本 IDs、不同窗口候选的样本掩码：
- 标签成熟/停牌/退市/缺失处理、排除分母：

## 冻结评估计划（全部数值须预填）

- 按ET日期的 train / 内层搜索与early stopping / 内层校准 / 外层walk-forward validation / final holdout 区间：
- purge条件、折数、训练窗口、训练/评估的 as_of：
- 预测输入更新时点；独立登记训练历史长度、重训周期、可选时间衰减候选及内层选择规则：
- 外层重训回放计划、每次训练标签成熟/可用截止、模型启用时间；禁止回写过去预测：
- B0 / B1 定义；模型和有限超参列表；多重比较处理：
- 两路线各自 trial/算时上限、种子/重复数、失败与提前停止记录规则：
- TCN 各分支感受野、参数量上限、融合结构；LightGBM 叶子/深度/迭代限制：
- 概率校准方法、threshold或Top-K规则与选择数据：
- 主指标 Brier；相对B0/B1改善的95%区间下界 >0：
- 最小交易日/时间块/样本数，最大校准误差，最低覆盖，最大误报：
- bootstrap块长、次数、seed；按小时/月份/股票/状态拆分：
- stop/go规则（失败怎么记录，禁止测试集回调哪些参数）：
- HAB 主方案、HA−H/HB−H/HAB−HB/HAB−HA/HAB−H 配对比较；跨模型实用差异容限与成本选择规则：
- 首轮分别评估、不默认集成；若另做集成，互补性/权重选择数据与新的独立验证期：
- 执行延迟/滑点敏感性（与主统计分离）：

## 运行后附录（不得改上面的冻结内容）

- manifest、配置、模型、schema、split IDs、逐条预测/排除记录路径：
- 指标、置信区间、校准图、false positives/misses/coverage：
- 11:30主时点与更新时点的差别；MFE/MAE风险：
- 八格完成/失败/数据不足状态及全部 trial 台账；未校准/校准和匹配窗口/各自最优窗口结果：
- 参数量、训练时间、峰值内存、制品大小、全池推理及数据到齐至输出总时延：
- 选型结论：A/B 增量、跨期稳定性、LightGBM/TCN 差异与不确定性：
- 失败/异常、数据限制、结论层级与不允许宣称的能力：
- 下一步；如果已看holdout，登记为已暴露：
