# 全市场 30 / 60 日翻倍机会

目标：找好公司、处于热点、60 日翻倍概率超过 80% 的股票。现版本还没有达到该目标。当前默认入口为完整原生日线、拆股与财务行业的 v3.1；原 v1/v2、全部失败与未知保留。最新报价、历史模型锚点与财务核验分开显示。

算法、价格参考、六个月特征、未来标签、时间隔离、去重、公司质量、准入条件见 [PROTOCOL.md](PROTOCOL.md)。这里的“价值表现”首先是股价 / 成交量表现；内在价值估值尚未完成，不能将它们混同。

```bash
# 扫描整个证券目录，输出一次报告，并生成独立离线调试页
python3 analysis/doubling_opportunity_v1/scan.py --top 30 --debug

# 刷新目录、报价与完整同源NONE日线快照：Local -> R2 -> OpenD
# 本机OpenD行情连接须可用；财报使用已冻结缓存，不调用交易接口
python3 analysis/doubling_opportunity_v1/scan.py --refresh --top 30 --debug

# 明确复跑原独立期限模型；不覆盖 v2 的最新指针
python3 analysis/doubling_opportunity_v1/scan.py --version independent-v1 --top 30

# 原始固定评价月份全部重新训练 / 校准 / 回测，所有失败与未知保留
python3 analysis/doubling_opportunity_v1/train.py

# 三分类首次触及时间的价格对照模型
python3 analysis/doubling_opportunity_v1/joint_v2/train.py

# 全注册财报及身份资料终止后，冻结完整覆盖并运行财务×行业量价实验
# 未完成时拒绝用下载顺序子集回测；可选择有时限等待
python3 analysis/doubling_opportunity_v1/joint_v2/train_financial.py --wait-for-acquisition

# 运行因果 / 标签 / 概率区间检查
python3 -m unittest discover -s analysis/doubling_opportunity_v1 -p 'test_*.py' -v
python3 -m unittest discover -s analysis/doubling_opportunity_v1/joint_v2 -p 'test_*.py' -v
```

原始两个期限模型出现 p30>p60 时，两列可用概率均撤回，原始值保留于全表；不以抬高 60 日数值制造一致性。

v2 使用同一分布的 q0/q1/q2，P30=q2，P60=q1+q2；联合温度校准保持期限嵌套。价格对照当前最大 P60 约79.47%，高估计股票不自动满足经营质量；5–8月月前选择的60日策略有2 TP / 14 FP / 20未知，成熟 precision=12.5%，未通过。旧v2财务联合版已经完成；显式使用--version joint-v2时，输入/代码指纹一致的报告使用该版，价格对照概率与回测另列保留。联合版在5–8月有660个同时通过当时经营健康与个股/行业热度的股票日（183股），两算法均无>80%有效信号；precision不可评分，机会供给未通过。当前联合模型最高估计60.85%，MXL为16.46%，两者均锚定09-29，不能当作最新概率。

Python 3.14、pandas 3.0.6、numpy、scikit-learn 1.9.1、scipy、joblib、pyarrow。全部缓存、模型和原始行情留在忽略目录；不自动上传 R2，不调用交易接口。默认v3使用OpenD同源NONE日线；源失败必须显式报错或保留缺失，不能冒充最新。旧v1/v2的Massive路径需要相应凭证。

默认产物：`raw_daily_v3/runs/<UTC>/report.md` / JSON / `all_stocks.csv` / `universe_exclusions.csv`，`raw_daily_v3/latest_run.json` 指向本次报告；另保存本次不可变报价快照。旧v2产物在独立的 `joint_v2/runs/`。`joint_v2/backtests/` 是价格对照；完成后 `joint_v2/financial_backtests/` 是独立命名的财务×行业实验。原 `runs/` 和 `backtest_v1/` 保留。

旧v2调试页在已有8768静态服务打开：`joint_v2/financial_debug.html` 展示财务与行业实验，`joint_v2/debug.html` 展示价格对照。`scan.py --version joint-v2 --debug` 分别生成两页；单文件离线，灰阶原生表单。价格页有189只日线详情、254条跨策略信号记录；财务页有39只日线详情、零已发信号；两页均有4只股票的60分钟路径。全池表、日期和两年mini导航、显式平移、失败当时特征与未知状态保留。原始OHLC嵌入的生成页已加入忽略规则，不入Git；生成器和模板可版本化。其余候选详情及全部5分钟路径仍缺失，不能称完整分钟验收通过。

每日全量前向预测在 `.cache/doubling_opportunity_v1/forward_predictions.parquet`；用于复算，不能把每日日预测当独立买入机会。

目录来自 Nasdaq 全市场名单。当前目录回溯是有幸存者偏差的开发回测。SEC ticker 文件若不可访问，尝试从 SEC 财务 frame 中唯一、精确归一化发行人名取得 CIK；歧义或失败保持未核验。并不由模糊公司名猜 CIK。最新财务只决定当前核验状态；联合版历史特征另按当时披露信息构造，不以今天的财务状态回填历史。非美会计标签、金融业指标及缺失财务需要单独处理。

当前逐项审计见 [DELIVERY_AUDIT.md](DELIVERY_AUDIT.md)。全池财报获取：`python3 analysis/doubling_opportunity_v1/acquire_fundamentals.py`，支持本地断点续取，最多 3 工作者、总请求发出间隔至少 0.4 秒。

SEC当前ticker/SIC元数据：`python3 analysis/doubling_opportunity_v1/joint_v2/acquire_submissions.py`。快照：`python3 analysis/doubling_opportunity_v1/acquire_quotes.py --refresh`。两项获取均仅查询公开数据，不接交易接口。财务事件、插补、温度校准、信号必要条件见 [v2预登记](joint_v2/PROTOCOL.md)。

**行情口径限制**：日聚合来自Massive拆股调整，不是Futu QFQ；批量日线尚未确认仅含常规盘。因此旧High触及结果仅为提供商日聚合的探索标签。最新报价不能补齐9月30日至10月2日历史；跨源高峰比值也待公司行动核对。详见 [SOURCE_AUDIT.md](SOURCE_AUDIT.md)。

尚待完成：全市场最新日线与更长历史；历史 PIT 普通股目录和退市样本；跨归档批次的公司行动统一；历史业务催化特征及联合回测；新的独立观察；达到有样本和置信区间要求的 >80% 证据；全池分钟失败路径核对。现版本不能声称完整交付目标。

原生日线、拆股与财务行业 [v3.1](raw_daily_v3/TRAINING_PROTOCOL.md) 已完成完整注册源和固定2026年4–9月回测，当前模型行情截至10月2日。源获取与工程核验已经完成，>80%机会目标未通过；旧v1/v2产物独立保留。

```bash
# 默认完整扫描：未变化的输入/模型复用；重新读取广池报价并产生一次报告
python3 analysis/doubling_opportunity_v1/scan.py --top 30 --debug

# 检查实际获取、审计、训练状态，不启动重复作业
python3 analysis/doubling_opportunity_v1/scan.py --status

# 明确v3版本；与默认入口相同
python3 analysis/doubling_opportunity_v1/scan.py --version raw-v3 --top 30 --debug

# 新的目录和完整日线源快照；完成恢复/动作审计后重新训练
python3 analysis/doubling_opportunity_v1/scan.py --refresh --top 30 --debug

# 原主策略之外的已冻结高估计诊断，不改变模型或信号资格
python3 analysis/doubling_opportunity_v1/raw_daily_v3/diagnose_probabilities.py

# 离线重放已有报价必须明确指定，报告保留缓存年龄
python3 analysis/doubling_opportunity_v1/raw_daily_v3/run.py --reuse-quotes --top 30
```

v3完整注册5,261候选＋SPY；原始与恢复源均终止，5,262标的有非空原始结果，5,252含最新交易日（包括SPY），9只候选停留旧日，SVA停留2019年。原始空/旧结果与33只恢复记录全部保留。派生原生日线4,328,415个股票日，价格可评分2,582,650个；当前可评分3,663/5,261，未评分1,598个仍在全池CSV。

主策略将当时已披露经营质量、个股量价与SIC同业量价门禁一起应用。六个月730个当时条件合格股票日，两个算法与月前选择的30/60日策略均为零>80%信号，precision不可评分，供给门禁未通过。当前20股通过经营必要条件和个股热度，但其同业热度没有通过，完整联合资格为零。当前全池仅SDEV的P60估计92.50%；公司质量不通过、同业热度不通过，不能列为合格股票。MXL估计P30=0.13%、P60=0.65%，这是日线锚点下的开发估计，不能承诺当下翻倍概率。

另外的[冻结概率事后诊断](raw_daily_v3/PROBABILITY_DIAGNOSTIC_PROTOCOL.md)读取同一已有预测、不施加质量/热点门禁，沿用>80%与60日冷却；月前选择的60日诊断有4次机会，0TP/2FP/2未知。它不是主策略信号，也不是独立验证。不同算法、每日行和重叠窗口不能合计为独立市场证据。

v3首次拟合前核查完整2,838条Nasdaq公告；发现11项原文生效日期不一致、15项组合动作，保留未知，不猜日期或给父公司套子公司拆股。原生get_rehab、QFQ与HFQ共同遗漏的明确DUKR/DVLT拆股按独立主源核对，重复公告只应用一次。规则与原来源记录见[ACTION_RULES_ADDENDUM.md](raw_daily_v3/ACTION_RULES_ADDENDUM.md)。公司身份冲突、不支持动作、非法OHLC、缺失或未成熟未来前缀均保留未知；合股自身不算翻倍。现金股息不计入。

本轮原生日线采集没有调用历史K线接口，代码SHA一致；全局历史额度实际373→374，归因未确认，不能将整个采集称为额度完全不变。本连接K_DAY订阅已清理（own_used=0）。数据走Local→R2→OpenD，不与旧Massive/QFQ拼接，不上传R2。

最新单次报告由[raw_daily_v3/latest_run.json](raw_daily_v3/latest_run.json)指向，包含股票列表、上一确认高峰时间/价格、日线参考与快照last_price的两个比例、阶段、P30/P60、经营状态、热度与六个月回测；[CURRENT_EVIDENCE.json](raw_daily_v3/CURRENT_EVIDENCE.json)记录完整源SHA和3,663个当前预测的实际复算。财务原始/派生资料、全部SIC元数据、官方日历、来源审计、模型与评价产物均纳入冻结指纹。改动这些输入须产生新版本，不能沿用旧验收。

调试页：[raw_daily_v3/debug.html](raw_daily_v3/debug.html)，在已有8768服务打开。5,260只股票的已有日线可按需加载，SVA在模型范围没有日线，保持空白。已有登记NONE五分钟缓存116股，来源核验104股，完整RTH与日线同时覆盖18,098个股票日；其中3日的日High高出RTH至少0.5%。104股原价路径独立加载，12股未核验来源不画；该子集不改标签，不证明全市场RTH或可成交。日线和分钟原价分开，目标按股份单位换算；未核实动作不画未经确认的分钟目标。--debug复用未变化的分钟审计，不重做模型选择。

完整公司行动/历史身份、原生日High时段、PIT与退市、业务催化历史、当下真实入场参考和新的独立效果观察仍不足。只能称历史前向开发回测；工程产物和模型大于80%的数值均不能提升为真实80%机会。

已暴露案例分别见[MXL_CASE_REVIEW.md](raw_daily_v3/MXL_CASE_REVIEW.md)和[SDEV_CURRENT_REVIEW.md](raw_daily_v3/SDEV_CURRENT_REVIEW.md)；调试页面的实际桌面/320px记录见[UI_VERIFICATION.json](raw_daily_v3/UI_VERIFICATION.json)。


2026-10-05新增覆盖审计发现：旧股票目录按名称误排普通ADS、注册/投票股，SEC frame精确名称关联也未覆盖NVDA、AMAT、VRT、RKLB等已有行情股票。因此v3的5,261是旧注册池，不能称无遗漏全市场股票覆盖；财务缺失不能解释为经营不健康。独立[v4覆盖修复](coverage_v4/README.md)按完整13,295行目录重新注册：5,516明确股票、98待核验证券、7,681排除；完整观察池5,614证券。v4来源获取和新评估尚未结束，旧v3结果继续保留，没有新>80%机会证明。
