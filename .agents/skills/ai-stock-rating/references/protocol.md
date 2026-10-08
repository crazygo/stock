# ai_quality_v1：AI 业务质量与研究优先级

本协议是可审核的分析规则，不是收益模型，也不是统一行业的信用评级。按产业群及发展阶段解释财务，保留每项证据；不生成未经验证的百分制总分。

## 业务判断

每条人工/模型辅助业务核查记录包含 code、stage、materiality、commercial、moat、risk_flags、risk_reviewed、risk_evidence、reasons、sources、observed_at、review_status。分项是有序类别，不是上涨概率；资料不足取 null。

| 字段 | 值与含义 |
|---|---|
| stage | mature：已有稳定商业业务；scaling：付费业务扩张但盈利未成熟；precommercial：研发、试点或临床，商业规模尚待兑现 |
| materiality | 1：AI 为小功能、内部效率或较远间接路径；2：具体业务承接 AI 需求且已有重要客户/产品验证，AI 占比可能未单独披露；3：官方经营披露证明 AI 是重要增长动力、关键终端或可量化收益来源 |
| commercial | 1：产品、试点、研发/临床路径明确但规模未兑现；2：实际付费客户或交付，规模/重复性仍待验证；3：经营披露证明规模化收入/订单/持续采用，不能把纯意向订单算收入 |
| moat | 0：已查明替代性强/优势不足；1：具备专用技术、工作流/渠道/客户绑定，但持续优势尚待观察；2：有设计采用、产品壁垒、认证、集成生态、转换成本或市场位置证据支持持续优势 |

来源须是实际取得的 SEC 财报/正式经营披露、公司官方投资者或产品资料。ETF 行业字段和 SEC SIC 仅能支持行业。主营是否 AI 与公司自身是否优秀分开；供应配套可以 A+，公司内部用 AI 不自动成为核心研究对象。

风险记录须绑定已查阅来源，不能只写 `risk_reviewed: true`。`risk_evidence` 是数组，各项格式为
`{domain: "customer" | "competition" | "funding", risk: "具体风险", evidence: "披露依据及判断", source_url: "sources 中的原始URL"}`。
A/A+ 须覆盖客户、竞争、资金三类；A− 至少有一项具体引证风险并说明发展路径及资金承受证据。
重大风险如已解释，另用 `risk_resolutions: [{risk: "与risk_flags原文一致", resolution: "有依据的解释或处理", source_url: "原始URL"}]` 记录，不能把重复风险描述当作缓释。

## 财务判断

固定截止日，仅用 filed≤截止日且 end≤截止日的真实事实。优先最近可复算 TTM；年度结果及期间明确显示。TTM=上完整年度+本年累计−上年同期累计，必须同币种、同会计概念，期间长度/财政年对齐。不把三个月、九个月、全年和瞬时余额混用；若无 capex，FCF 未知，不能当作 CFO。

FCF=CFO−现金购买固定资产支出，仅作现金代理，注明租赁、并购、资本化成本等未覆盖项。增长率比较同口径前一期；毛利率/净利率的分子分母必须同期间。现金跑道仅在现金及最近经营现金消耗可比时计算，现金÷月均经营现金消耗；不冒充可用授信或融资保证。该指标仅含现金及现金等价物，不含可流动投资，是流动资产跑道的下界，不能单独证明公司融资能力不足。

财务支持类别：
- strong：最近完整期间营收、净利润、CFO、capex可得且匹配；净利润及FCF均正；有增长和现金兑现证据。默认净利率及FCF率各≥5%，营收同比为正；这些是筛选下界，仍按产业解释。债务/现金、客户集中和稀释须在业务核查中检查。
- sound：最近收入、净利润及CFO可得且匹配，利润及CFO均正；或扩张阶段有正CFO、明确资本开支及融资承受证据。缺关键风险说明时不能升 A+。
- developing：商业路径存在，盈利/FCF未成熟，最近现金及现金消耗显示至少12个月跑道；或业务核查有明确资金保障来源。高投资行业负FCF不自动等于差公司。
- strained：有真实亏损/现金消耗及不足12个月资金承受证据；没有可核查补足资金来源。
- unknown：关键字段、期间可比性、币种或资金承受证据不足。

不对银行/保险用普通工业FCF公式判优劣；这类公司需适当行业财务核查，缺少时待评级。原始取得时间不明的文件标“取得时间未保留”，复制/哈希校验时间不写成下载时间。

以资金弱点评级 B 时，资金风险引证还须明确 `liquid_assets_reviewed: true`、`latest_financing_reviewed: true`、`funding_capacity: "insufficient"`，表示已核查可流动证券和最新融资后仍不足。仅现金口径不足而其他资金未知，保留待评级。
`funding_protection` 保留融资类型、金额/币种、用途、取得时间、可用期限及官方来源。已在期末现金中包含的融资不能重复计算；建设项目融资不能自动视为公司运营资金。

## 评级门槛

| 等级 | 必需条件 |
|---|---|
| A+ | materiality=3、commercial=3、moat=2，财务strong；业务核查含具体客户/竞争/资金风险与对应证据，最近财务期末不超过180日，无未解释严重风险 |
| A | materiality≥2、commercial≥2、moat≥1，财务strong或sound；已核查具体客户/竞争/资金风险及对应证据，无未解释严重风险 |
| A− | materiality≥2、commercial≥1，已有发展阶段及风险核查；财务developing，或财务sound/strong但商业<2或竞争优势<1。具备事实依据，不能用它填充A/A+缺少的风险资料 |
| B | materiality=1且有业务核查依据；或已证实重大弱点/strained。B表示低研究优先级，不断言公司完全不使用AI |
| 待评级 | 任一所需判断缺失、身份不明、商业证据只有营销口号、财务unknown或资料过期。新股不因名字含AI获得评级 |

业务核查默认90日内，财务核心数值期末默认210日内（A+180日内），Company Facts 来源快照取得默认30日内；正式半年度/年度披露制度可用明确记录的例外及理由。每次运行有效期最多30日，并受业务、财务及来源快照的剩余时效约束；资料发生重要变化需提前重评。超过有效期默认不进入当前核心名单，但保留旧等级供回看。

## 保存与变化

`quality/universe.json`为本次研究输入；`reviews_*.json`是逐股判断原料，不能直接决定最终等级。`financial_snapshot.json`保存事实及来源，`rate.py`执行上述门槛。生成 current.json 与 runs/<run_id>/snapshot.json、评级CSV、变化摘要；快照不可覆盖。每只股票写 quality.grade、decision、financial_support、financials、reasons、missing、risks、sources、as_of、expires_at、rule_version、review_trigger、previous_grade/change。

默认入围名单仅有效 A+/A，A−作为潜力观察。数量由证据决定，不凑配额、不强制每群都有A+。资料未评级不计为B或失败投资；旧等级转待评级保留转变原因。新成员没有旧等级，离开成员池单列退出。变更规则版本时不将旧版本等级变化解释为基本面改善。

本次执行仅更新评级、证据、名单及 HTML；不改变预测研究的模型、标签、阈值、冻结准入、模拟账本或券商自选。价格/估值另列；无当前报价及可比口径时显示未评估。

权威接口与阅读入口：[SEC XBRL API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)、[10-K/10-Q 阅读指南](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins/how-read)。
