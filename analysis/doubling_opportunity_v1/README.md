# 30 / 60 日翻倍机会研究

**当前没有有证据支持的“当下60日翻倍概率>80%”股票。** 默认为广池证券分类、精确CIK、原生日线、财务与同业量价的v4。原v1–v3、全部失败及未知保留；[旧v3入口说明](README_V3.md)和[旧v3交付审计](DELIVERY_AUDIT_V3.md)为历史快照。

```bash
# 单次广池扫描：完整来源相同则复用模型，更新有预算的报价，生成报告与调试页
python3 analysis/doubling_opportunity_v1/scan.py --top 30 --debug

# 读取实际来源、训练与最新报告状态，不启动重复作业
python3 analysis/doubling_opportunity_v1/scan.py --status

# 新来源cohort；重新获取目录和SEC，保留原始来源与旧模型
python3 analysis/doubling_opportunity_v1/scan.py --refresh --top 30 --debug

# 已准备模型的一次参考报告；日线过期时明确保留过期状态
python3 analysis/doubling_opportunity_v1/coverage_v4/run.py --top 30

# 明确复跑历史版本，各版本的最新指针独立
python3 analysis/doubling_opportunity_v1/scan.py --version raw-v3 --top 30 --debug
python3 analysis/doubling_opportunity_v1/scan.py --version joint-v2 --top 30 --debug
python3 analysis/doubling_opportunity_v1/scan.py --version independent-v1 --top 30
```

本机OpenD行情连接须可用，SEC从公开接口读取。完整历史/财报准备与训练属于注册批处理阶段；未完成全部来源时拒绝拟合下载顺序子集。单次报告的报价补充在可终止子进程中运行，默认30秒，超时保留已完整落盘的文件并输出日线参考及缺失；不能将该预算解释为首次全市场准备和训练在30秒内完成。原始行情、报价载荷、模型和生成的行情调试页留在忽略目录，不自动上传R2，不调用交易接口。

[获取协议](coverage_v4/PROTOCOL.md)、[首次拟合前登记](coverage_v4/TRAINING_PROTOCOL.md)、[v4说明与证据](coverage_v4/README.md)、[交付审计](DELIVERY_AUDIT.md)。研究在main继续，只提交本研究清单，保留其他WIP。此版未提供云端可移植模型；不能把旧月模型、旧行情或本机OpenD当作云端当前覆盖。

算法使用183日历天价格/量能、当时已披露财务和当前SIC同业量价：multinomial LogisticRegression / multiclass HistGradientBoostingClassifier预测首次翻倍时间三分类，联合温度校准；P30=q2，P60=q1+q2。月前隔离训练、校准、选择，固定评价2026年4–9月，门槛严格>80%，同股60日冷却。特征定义和数值方案沿用冻结协议，没有为获得高概率修改门槛。“价值表现”包含股价、量能与经营必要条件，内在价值估值和业务催化联合预测尚未完成。

当前Nasdaq traded目录13,295行全部保留去向：5,516明确股票、98类型待核验观察证券、7,681排除；全观察池5,614。精确唯一CIK覆盖5,603个证券，登记5,555个发行人；两种SEC来源全部终止，56家财务源不可用。当前成员与SIC仍是回溯信息，不是历史PIT；OTC和退市历史尚未完整覆盖。财务IFRS/币种未支持者保持未知，不能解释为经营差。

v4有4,632,199个真实股票日，2,756,540个价格可评分历史行；当前日线参考截止2026-10-02，可评分3,886/5,614，1,728个未评分仍在全表，11个标的缺少最新完整交易日。全池最高P60探索估计30.88%；25个通过经营必要条件与个股热度的候选中，最高P60为3.02%，同业热度没有通过。没有合格股票。

固定4–9月主策略六组均为零>80%信号，precision不可评分，机会供给和效果门禁均未通过。另有[首次拟合前登记的冻结概率诊断](coverage_v4/PROBABILITY_DIAGNOSTIC_PROTOCOL.md)：去掉质量/热点条件但保持80%和60日冷却，月前选择60日有4次诊断机会、1TP/1FP/2未知；成熟precision=50%，小样本和未知不能证明80%概率。这些不是主策略有效信号，不跨算法累计。

[latest_run.json](coverage_v4/latest_run.json)指向单次Markdown/JSON报告、全池CSV、排除原因和本次报价接收记录。报告含上一确认高峰时间/价格、日线与快照last_price的两个比例、阶段、P30/P60、经营状态、热度、来源支持与回测。[CURRENT_EVIDENCE.json](coverage_v4/CURRENT_EVIDENCE.json)记录27,199项来源/产物SHA核查及3,886个当前估计的实际复算。概率属于完成日线Close×1.002参考，当前报价变化不会更新该概率。

[默认入口实际复跑](coverage_v4/DEFAULT_REPEATABILITY_VERIFICATION.json)验证相同源下复用模型，5,614逐股行的冻结概率与资格不变。本次30.01秒超时后保留1,200个快照，4,414个报价缺失明确展示；没有把缺报价证券删去或把日线参考称为当下入场概率。

调试页为[coverage_v4/debug.html](coverage_v4/debug.html)，在已有127.0.0.1:8768服务打开。5,613股的模型范围日线可按需加载，SVA在该范围没有日线。现有NONE五分钟源122股中109股通过核查，19,023个完整RTH股票日可与日线比较；这只是已有缓存的诊断，不重标训练标签，不是全池RTH或可成交验证。日期、两年mini真实OHLC、至少6px间距、显式平移及缺失路径保留。

MXL已暴露案例：2026-09-02 $60.10至2026-10-02 $105.93，涨幅76.256%，不是翻倍。v4在10月2日日线参考下P30=0.30%、P60=7.02%，上一确认局部高峰8月17日$89.00，比例119.02%，阶段为回升趋势；这些是开发估计，不能当作当前报价入场保证。旧案例核查保留在[原始记录](raw_daily_v3/MXL_CASE_REVIEW.md)。

完整公司行动/历史身份、全池常规盘标签、PIT与退市、业务催化、当前入场参考及新的独立效果观察仍不足。只能称历史前向开发回测，工程完成没有达成80%目标。
