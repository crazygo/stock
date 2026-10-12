持仓小时曲线历史相似窗口 v1 · 2026-10-08 独立探索批次

打开 index.html；无外部前端依赖，可直接打开，也可通过既有8768静态服务访问。
当前页面为 sidebar_result_view_v1：左栏集中算法版本、阈值、股票、历史日期、范围和
统计时段；右侧只呈现图表、日期、匹配状态与数值结果。窄屏设置折叠到顶部。
页面语言约定见 UI_COPY_GUIDE.md；公式、原理、研究边界和来源见 METHOD_NOTES.md。
该次改版沿用2026-10-08的固定批次及既有匹配方法，未刷新行情或修改筛选计算。
sidebar_browser_verification.json与sidebar_verification.json保存界面及源数据对照验收；
sidebar-results-preview.jpg为当前预览。改版前页面保存在runs/tail_gate_before_sidebar/。
主结果为09:00 ET小时截止，最新盘前快照获取于09:13:30 ET。
0086持仓快照获取于08:49 ET；19美股逐股分析，1港股单列不适用。
窗口从上一交易日04:00到本日09:00，为29个交易时段小时、31个价格点。
原v1：N=整体Pearson曲线相关系数0.90，历史/当前曲线极差比0.5–2。
当前默认探索v2：在原v1候选上额外要求末半段Pearson相关≥同一N，以及末半段
OLS拟合斜率同向且非平坦。N为相关系数门槛，不能解释成百分比相似度。
阈值是本次探索定义，不是验证通过的预测门禁。当前默认v2只有NVTS的2026-08-28达标，其余股票没有达标样本；不据此制造统计。

文件：
- protocol.json：在09:00批次结果计算前登记的口径和时点；沿用用户确认的N。
- results.json：逐股全体可评分日期、匹配、聚类、24小时/常规盘结果及基准。
- acquisition.json：Local→R2→OpenD尝试、原始数据路径与SHA256。
- positions.json / snapshot.json：实际只读持仓与报价；报价更新时间和获取时间分别保留。
- verification.json / independent_verification.json：数据检查及独立重算。
- report.template.html / render.py：自包含报告模板与渲染。
- extend_paths.py / curve_extension.json：主图的历史真实3交易日延伸与时间轴定义。
- verify_extended_paths.py / extended_path_verification.json：逐点核查历史延伸、缺失及原统计不变。
- runs/20261008T124943Z_0800/：保留首轮08:00结果，不与09:00混用。
- runs/20261008_0900_before_3session_overlay/：保留主图延伸前的09:00批次。

原始Parquet在仓库.cache/premarket_self_analogs_v1/，被Git忽略。
为避免不同时间点QFQ基准混用，先取一份360天全序列，再补09:00新柱；补取前逐项
比较全部重叠Open/High/Low/Close，确认复权基准未变化。没有上传R2，没有下单。

当前09:00批次复算（不取新数据）：
  python3 analysis/premarket_self_analogs_v1/analyze.py
  python3 analysis/premarket_self_analogs_v1/verify.py
  python3 analysis/premarket_self_analogs_v1/verify_extended_paths.py
  python3 analysis/premarket_self_analogs_v1/build_shape_match_v2.py
  python3 analysis/premarket_self_analogs_v1/verify_shape_match_v2.py
  python3 analysis/premarket_self_analogs_v1/render.py

acquire.py是本次固定证券、固定日期的采集脚本，支持分页、限制OpenD并发与可终止
子进程。它不是自动更新/调度器，也不自动刷新持仓或改变分析时点。新日期研究应新建
批次、重新读取持仓并更新口径，不能把此报告标为后续时点的实时结果。

所有涨跌、均值、中位、High/Low均以该历史窗口截止价为分母。历史美元价不直接
跨年平均，而以收益率映射到本次09:00价格。历史高低价中位不是承诺价格区间。
周末/假日24小时内闭市保留真实最后交易点；完整延伸到次日盘前的子集单列。
短历史、缺失、零量柱、报价陈旧、单例及零匹配均显式保留。

主图延伸：09:00之后的当天为第1个交易日，画到第3个交易日盘后结束（半日市17:00，
完整交易日20:00 ET）。按交易所日历接续交易日，周末/假日不制造价格，真实周日晚
夜盘保留。本次映射日期为10月8日、9日、12日；所选历史案例的实际日期另列。
黑色本次曲线只有已发生的31点，结束于09:00；彩色历史路径用同一个输入起点的
价格归一化，延伸后的累计对数涨跌不改成相对09:00收益。缺失或未成熟处断线。
原v1数据及其24小时/当天常规盘结果保持不变。探索v2仅使用已发生的29小时改变
入选日期，不使用之后结果筛选；后续结果的价格定义保持一致，按v2入选日期重新汇总。
默认高亮当前筛选范围内日期最新的达标匹配，检查日期列表按日期倒序排列；切换股票、
门槛或日期范围后重新选择最新匹配，手动选中的其他日期在重绘时保留。没有v2达标时，
主图仍保留全部原v1候选供核查，高亮最新候选并标为未通过。没有原v1候选时，主图
只显示本次，下方个案为单独标记的未达标参考。

全部候选图例：每个整体相关候选日期一条，日期倒序，无三条截断；显示对应色标、匹配起止、
当前口径相关系数、匹配状态、后续三交易日的时间范围及终值/最高/最低幅度。整体相关、
尾段相关和幅度比保留在历史日期表中。历史只用两种颜色：
6个自然月以内统一蓝色#3178b9，更早统一灰色#aeb4bc。蓝色按日期倒序依次使用实线、
虚线、点线、点划线等线型；图例的SVG与曲线使用相同dash数组。选中线加粗，保持
所属颜色及线型。列表标注六个月分组及线型，点击日期同步高亮。

近远期对照按6个自然月，以固定批次2026-10-08为基准，2026-04-08起为蓝色，
2026-04-08以前为灰色。月末日期向目标月末截断，不以180日历天近似。
两组都保留全部候选，按当前口径达标日期的完整三日结果逐日期等权报告样本数、上涨数、均值、中位及
高低价中位；没有按时间衰减计算预测概率。时间分组没有控制市场、行业或新闻环境。
三日结果相对各自匹配09:00真实Close起算，High/Low取之后三日全部所需小时OHLC；
图形仍相对输入起点归一化。未完成三日的案例显示已知行情截止及已知涨跌，完整三日
结果保留未知，不纳入三日均值/中位。recency_comparison.json保存原v1的N=0.90分组重算；
shape_match_v2_verification.json与shape_match_v2_browser.json保存当前v2与原v1对照验收。
recency_verification.json与recency_browser_observations.json保存此前原v1的逐股蓝灰图例验收；
当前v2与v1对照的页面验收见shape_match_v2_browser.json。
recency_display_protocol.json保存时间分组、颜色、三日结果与未知口径。

末半段诊断与探索v2（2026-10-09修订）：
- shape_match_v2_protocol.json / shape_match_v2.json：独立口径及全部4232个输入窗诊断，
  results.json及protocol.json保留原v1，不覆盖原始冻结批次。
- build_shape_match_v2.py：读取已有同一时点价格点，不联网、不刷新报价；
  末半段按压缩交易小时x≥14.5，实际使用x=15至29的15点，缺失不插值。
  分段相关系数=min(整体r,末半段r)，不是0–100的相似百分比。
- 曲线斜率用实际交易小时OLS；绝对斜率≤1e-8视为平坦，平坦/未定义末半段不入选。
- NRG 2026-09-08：整体r=0.922110，末半段r=-0.758284；同一末半段本次-1.5103%、
  历史+0.9279%。原整体相关候选11条，v2达标0条，全部候选仍显示且可检查后续。
- 全窗口30段上/下/平三类方向一致率12/30=40%，是独立诊断，未用作新筛选门槛；
  逐点变化相关和累计对数曲线RMSE亦仅诊断。没有凭视觉指定50%或调阈值制造匹配。
- 本次在已看到历史个案与后续结果后修订，只称探索描述，无独立预测验证。未来数据
  不参与代码筛选；不能把更严格筛选或单个保留案例的结果称作模型效果通过。
- runs/whole_curve_v1_before_tail_gate/保存此前蓝灰图例与报告源码/验收。
