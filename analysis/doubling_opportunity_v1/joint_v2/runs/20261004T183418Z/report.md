# 首次翻倍时间模型 v2 · 单次运行报告

运行 2026-10-04T18:34:18.986373+00:00；全市场日线截止 **2026-09-29**，最近应有完整交易日 2026-10-02，状态 **stale_not_current**。
完整证券目录 13295；普通股 / ADR 候选 5261；价格可评分 3763。30日≤60日概率矛盾：**0**。
最新报价覆盖 5261 个候选，接收时间 2026-10-04T18:05:36.199529+00:00。报价与历史概率分别显示，报价不会补齐缺少的中间日线。
当前 SEC 财报已覆盖 2281 个候选；其中增长且经营健康必要条件通过 188 个。获取状态：仍在运行，未覆盖保留待证。
**没有通过完整“60 日翻倍概率 >80%”证据门禁的股票。**

**日线时段待核验：Massive 批量日线未确认仅含常规盘。已算 High 触及保留作探索标签，不能视为已验证的常规盘翻倍。公司行动与证券历史类型也尚未统一。**

算法：同一个 multinomial LogisticRegression / multiclass HistGradientBoostingClassifier 预测首次触及时间，联合 temperature scaling 校准，P30=q(前30日)，P60=q(前30日)+q(31–60日)。
这是价格对照模型加当前财务核验报告。财务尚未作为训练输入，当前经营健康条件没有事后回填到价格对照的历史分母；公司质量 × 热点的联合模型回测仍未完成。
概率为模型估计，高置信度尾部尚未通过可靠性检验。当前目录、跨批次公司行动和行情过期问题仍限制解释，不能将极高输出当成已证实的80%机会。

## 增长且经营健康、量价热度候选（探索）

|股票|最新报价 / ET时间|概率参考价 / 日期|上次确认高峰 / 日期|最新报价占旧高峰*|阶段截止 / 阶段|30日模型估计|60日模型估计|最近营收同比|经营证据|量价热度 / 业务催化|
|---|---|---|---|---|---|---|---|---|---|---|
|MXL|$105.93 / 2026-10-02 20:02:44.169|$92.72 / 2026-09-29|$89.00 / 2026-08-17|119.02%|2026-09-29 / 回升趋势|1.06%|2.73%|55.17%|经营利润率 -2.48%；TTM现金流率 2.89%|个股量价热度通过 / issuer_evidence_verified|
|FSLY|$25.81 / 2026-10-02 20:02:41.937|$25.48 / 2026-09-29|$25.36 / 2026-08-27|101.77%|2026-09-29 / 回升趋势|0.39%|2.13%|23.27%|经营利润率 -7.87%；TTM现金流率 17.40%|个股量价热度通过 / 待核验|
|CEVA|$37.10 / 2026-10-02 20:02:32.346|$34.48 / 2026-09-29|$39.47 / 2026-08-07|94.00%|2026-09-29 / 震荡过渡|0.16%|0.86%|13.07%|经营利润率 -7.15%；TTM现金流率 3.29%|个股量价热度通过 / 待核验|

*最新报价是未复权价，旧高峰为归档拆股调整价；比值为待公司行动核对的参考。阶段和模型均截至概率参考日期。财务核验截至 2026-10-04，不会回填历史模型。

## 全池最高价格模型估计（未证明为好公司）

|股票|30日模型估计|60日模型估计|财务质量|量价热度|证券类别核查|
|---|---|---|---|---|---|
|FCUV|9.06%|79.47%|not_passed_or_incomplete|量价热度未通过|SEC当前ticker已确认|
|GMEX|6.22%|60.40%|financial_cache_missing|量价热度未通过|Nasdaq名称规则，尚待精确类型核验|
|NCT|38.46%|46.56%|financial_cache_missing|量价热度未通过|Nasdaq名称规则，尚待精确类型核验|
|CD|9.52%|43.45%|growing_but_operating_health_unproven|量价热度未通过|SEC当前ticker已确认|
|TANH|35.12%|42.19%|financial_cache_missing|量价热度未通过|Nasdaq名称规则，尚待精确类型核验|
|CDT|21.05%|42.03%|financial_statement_missing_or_unsupported|量价热度未通过|SEC当前ticker已确认|
|CTNT|12.23%|40.43%|not_passed_or_incomplete|量价热度未通过|SEC当前ticker已确认|
|IPST|3.19%|39.17%|financial_cache_missing|量价热度未通过|Nasdaq名称规则，尚待精确类型核验|
|LGHL|31.13%|38.91%|financial_cache_missing|量价热度未通过|Nasdaq名称规则，尚待精确类型核验|
|GTBP|15.70%|38.29%|financial_cache_missing|量价热度未通过|Nasdaq名称规则，尚待精确类型核验|
|IMCC|32.55%|37.34%|financial_cache_missing|量价热度未通过|Nasdaq名称规则，尚待精确类型核验|
|VERI|4.24%|36.99%|financial_cache_missing|量价热度未通过|Nasdaq名称规则，尚待精确类型核验|
|KITT|19.59%|36.13%|not_passed_or_incomplete|个股量价热度通过|SEC当前ticker已确认|
|CMPX|2.41%|34.36%|not_passed_or_incomplete|量价热度未通过|SEC当前ticker已确认|
|JAGX|27.22%|33.56%|not_passed_or_incomplete|个股量价热度通过|SEC当前ticker已确认|
|DCX|30.87%|33.17%|financial_cache_missing|个股量价热度通过|Nasdaq名称规则，尚待精确类型核验|
|APUS|30.88%|33.06%|financial_statement_missing_or_unsupported|个股量价热度通过|SEC当前ticker已确认|
|CPOP|10.47%|32.41%|financial_cache_missing|量价热度未通过|Nasdaq名称规则，尚待精确类型核验|
|RENT|11.75%|31.01%|financial_cache_missing|量价热度未通过|Nasdaq名称规则，尚待精确类型核验|
|INDP|10.53%|30.91%|financial_statement_missing_or_unsupported|量价热度未通过|SEC当前ticker已确认|
|GLND|20.11%|30.23%|financial_statement_missing_or_unsupported|个股量价热度通过|SEC当前ticker已确认|
|VEEA|20.22%|30.13%|financial_cache_missing|量价热度未通过|Nasdaq名称规则，尚待精确类型核验|
|ISPC|2.46%|29.52%|not_passed_or_incomplete|量价热度未通过|SEC当前ticker已确认|
|LONA|18.73%|29.47%|financial_statement_missing_or_unsupported|量价热度未通过|SEC当前ticker已确认|
|ANPA|5.11%|28.05%|financial_cache_missing|量价热度未通过|Nasdaq名称规则，尚待精确类型核验|
|INO|4.26%|27.21%|not_passed_or_incomplete|量价热度未通过|SEC当前ticker已确认|
|AIFU|14.65%|27.12%|financial_cache_missing|量价热度未通过|Nasdaq名称规则，尚待精确类型核验|
|NCPL|22.35%|26.91%|financial_cache_missing|量价热度未通过|Nasdaq名称规则，尚待精确类型核验|
|ARTL|19.27%|26.23%|financial_statement_missing_or_unsupported|量价热度未通过|SEC当前ticker已确认|
|TNMG|19.25%|26.23%|financial_cache_missing|量价热度未通过|Nasdaq名称规则，尚待精确类型核验|

## 保留的历史前向开发回测

|目标|算法 / 策略|发出>80%信号|TP / FP|未知|precision|TP/全部信号|门禁|
|---|---|---|---|---|---|---|---|
|30日|multinomial LogisticRegression|27|4 / 15|8|21.05%|14.81%|未通过|
|30日|multiclass HistGradientBoostingClassifier|2|0 / 2|0|0.00%|0.00%|未通过|
|30日|monthly joint-model selection|24|3 / 14|7|17.65%|12.50%|未通过|
|60日|multinomial LogisticRegression|157|23 / 104|30|18.11%|14.65%|未通过|
|60日|multiclass HistGradientBoostingClassifier|8|1 / 7|0|12.50%|12.50%|未通过|
|60日|monthly joint-model selection|36|2 / 14|20|12.50%|5.56%|未通过|

## MXL 已暴露案例

MXL：最新报价 $105.93，ET 2026-10-02 20:02:44.169；模型截至 2026-09-29，参考收盘 $92.72，P30=1.06%，P60=2.73%；经营质量层为 growth_and_operating_health_checks_pass。这不是独立验证样本。

财报按决定日前 filed 的记录读取；季度累计差分、TTM 连续季度与来源 accession 全部保留。业务催化出处与行情、目录、模型切分、财报覆盖见 report.json。全部候选及未覆盖 / 未评分行在 all_stocks.csv；没有事后删除失败。
