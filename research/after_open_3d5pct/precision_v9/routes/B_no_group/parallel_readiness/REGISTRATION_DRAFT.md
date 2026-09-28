# B 无群 · 部分日级历史输入干预登记草案

2026-09-27；`draft_pending_root_adoption`。本文件可供根采用后交Sol实现，**不是已执行登记/训练结果**。实验身份建议`B_no_group_partial_daily_history_v1`，新run建议`runs/precision_v9_20260927/B_no_group_partial_daily_history_v1`，不存在才可创建。不得改原R03/P0/P1登记、冻结prepare/采集代码、旧制品或在本轮执行。

## 单一问题与固定对照

在原108股、14424条旧样本及其原序完全不变的情况下，为已取得完整2025-08至12月来源的60候选补日级回看，是否改善B无群的三群买入precision/供给及同股排序？选股集合由本次来源盘点确定，运行后不会因成败更换。它不是全545片扩量，也不增加早期fit日期。日级补长包含历史形状、126日有效支持及日级内部20日归一化的共同变化；必须分别报告改动谱，不能把概率差异归因于单独mask变化。

- 对照：v8 R9 `round_09/B_no_group/{dev1,dev2}` 的真实冻结checkpoint、cal参数、原cal/eval概率，原recipe为`representation,regularization,calibration`三项true，**prior/curves/capacity/balance/specialize/recency/bagging均off**。
- 处理：全108股保留，60股只替换`xday`；另48股全部有效tensor不变。全108的`x5/x60/group/group_seq/y`原字节保留。原group/group_seq仅留作不可变旧数据元信息，不进入B模型；不重建群、不读取QQQ作为输入。
- 所有样本ID/顺序、symbol/date/cutoff/decision、entry_at/entry_price、九标签、label_end/available、terminal1/3/5、group_week/group_valid_count保持不变。仅处理股的`full_126_prior_days`与`available_daily_days`可按实际新xday支持重算；诊断元信息不作为新特征。
- dev1 fit/tune/cal/eval固定3064/1045/936/1975行，dev2固定6090/934/930/2056行；逐键及原权重数组一一绑定。fit所有原行权重1，tune/cal群等权原值不动。先两开发折；**diagnostic不构建评分/不训练，本草案不授权它**。

## 精确来源与价格/时间路径

1. 旧根`runs/focus_v8_20260927`的dataset、config、protocol、checkpoint/校准、冻结inputs及source_provenance为控制真值。`inventory.json`列当前身份和60/48名单。旧features身份`6bce8471011f1c2c54ce2551ba7584f18196fb5b056580e528d95532d5c0e744`、rows身份`b6220c3645ff6477ff59763d75deaff26b6f60c7b1ad34f6d0b2b913a4fed2fe`，不可替换为共享warehouse新文件。
2. 处理来源仅原`R03/parts/<60候选>/<2025-08..12>/bars.parquet`共300片，及同60股v8冻结`inputs/market_data/us_5m/<symbol>/2026.parquet`与公司行动表。QQQ的5片不使用。实施前将300组result/bars实际SHA逐项写入完整manifest并核盘点集合指纹；原始页/审计可追溯，不复用未完成恢复目录。
3. 官方日历使用原R03/calendar的2025段，与v8冻结2026段逐session开/收盘/duration核对；有差异即停止。只从同一股票官方RTH完整5m构成日OHLCV/turnover，2025-11-28、12-24按实际210分钟，其他正常日390分钟；夜盘/盘前/盘后不进入日线。NONE价格不作新复权。
4. **所有108股的本股数值路径原本都是v6 float32**：原5m值cast float32、RTH聚合与`_encode/_prior_median`的float32数组和既有运算次序保留，最终xday float32。原101股float64/新增7股float32分流仅属于群分类，不能移植到本股。2026 daily重建须先在2026-only负对照中逐元素/NaN-mask匹配旧xday，再加2025；不允许全表转float64后再cast冒充同一路径。
5. 只在决策前完整且可用的历史日编码，bar_end≤cutoff且available≤decision。R03历史available为bar_end+1秒假设，没有received_at，不冒称实时证据。日级相对成交额仍用之前最多20官方session逐列中位数，保持旧partial-history语义和valid mask，不能用未来/全样本分母。首旧日2026-03-02之前只有145已归档官方session，不能称146日依赖全部完整；实际分母计数逐行记录。
6. 按冻结公司行动表检查全部新日级依赖（最多126日+20日分母及边界前收）。本次盘点60股8099旧行的过去146session至当日已知split冲突为0；实现仍逐行检查。任何新增冲突、来源迟到或不能保持原键合格即整run停止、另登记修复，不能删键或忽略split。供应商事件表完整性仍未外部认证，冻结事件表不是PIT公司行动公告证据。

## 允许改动谱与禁止变动

原树有效schema为585本股摘要+84路径=669列。每个尺度摘要按`valid_fraction, mean(ch0,3,4,5,9,11,12,13), max(ch1), min(ch2), first_half_mean(ch0), last_half_mean(ch0)`共13项。

| 有效输入范围 | 行为与必须证据 |
|---|---|
| `f_000..f_103`，8个5m日片段摘要 | 全14424行字节/NaN位置相同 |
| `f_104..f_506`，31个60m日片段摘要 | 全14424行字节/NaN位置相同；即使完整补长会改变早期20日成交分母，本实验明确保持原值 |
| `f_507..f_584`，6个21日日级片段摘要 | 仅60股允许变化，共78列；每列/日/股给差异计数和来源窗口 |
| `path_000..path_027`（5m）、`path_028..path_055`（60m） | 全14424行相同 |
| `path_056..path_083`（10/21/63/126日×7路径摘要） | 仅60股允许变化，共28列；沿用原path_summary公式/NaN处理，不加新曲线 |

因此只有106/669列可变，剩563列及48股全669列不变。处理组xday新2025槽可由旧padding变真实14通道；既有2026槽的OHLC/量/时长/类型/位置/valid数值保持，只有依赖新增历史的cross_bar_return、relative_dollar、gap_log_hours、relative_dollar_valid可以变化，且须逐项给边界来源。不得使旧invalid的2026日变valid或改写实际价格。`2026-08-04`起连126日+20日分母都只依赖2026，60股全部日级有效输入也必须与旧值相同；`2026-07-07`起无2025可见日线，但此前分母依赖仍可能影响早期2026槽。此日历边界需实现时独立复算，不硬套行号。

## 实施入口与开跑门禁

新增薄入口（建议独立`partial_daily_history.py`/purpose，实际路径由根排期）读取旧NPZ及固定键，仅生成xday替换版、支持sidecar、完整改动谱和实际partial manifest；**不调用带写入的原prepare，也不伪造R03完整actual_manifest**。源获取完全离线。冻结依赖源码字节、Python/LightGBM版本、运行前后source哈希；新artifact使用真实本轮训练来源，不能事后将旧模型重命名为新模型。

拟合前全部必须通过：

- 旧键14424/108股/原序完全相同；所有固定字段/五个固定tensor、split IDs、正权重监督IDs、权重数组、prior-disabled状态全等；旧制品前后哈希不变。
- 60股2026-only日级构建与原xday完全一致；处理版改动只落在上述allowlist。48股input全等；563不变有效列全等；所有key的未来/非本股/QQQ/群扰动不得改变本股有效输入。
- 在旧checkpoint固定时，以控制输入复放两开发折全部旧raw/p九目标及原校准结果，最大误差≤1e-7；这是有效输入/执行器负对照，不要求新训练后未处理48股预测不变（共享树参数本来可能改变）。
- 原始值/完整RTH/时间/known-split检查通过，源哈希与原训练身份明确，实际manifest声明`partial_daily_history_patch`、60/48来源不均、exposed_development，绝不声明545完成或历史PIT。
- 只有全部工程门禁后才允许根启动两开发折拟合；如无有效输入差异则记录无干预，不重复训练。

## 拟合、选择、成本与复盘

每折重训原九头一次，共18个LightGBM头；单线程、seed3566、120树上限/7叶/深4/min_child150/L2=30/lr=.035/早停20，tune/cal及收缩Platt、九目标单调投影均不改。对照优先复用已通过控制复放的原checkpoint，不增加控制重训；若执行器/历史环境不能复放，停在工程层并另登记，不临时切换控制定义。

动作raw主目标仍只在各自cal的14阈值(.30至.95，步长.05)选择：precision≥.90、n≥5、≥3日期、供给≥1/5session，优先n/precision/阈值；无解拒绝。共同三群、1170分钟同股冷却、全官方eval分母19/20、交叉群联合触发、零信号null、High触及代理均保持。不以eval挑阈值，不把处理概率≥.9当precision。

开发保留：先比较两折×三群通过单元，增加且不丢失原通过单元可留为候选；通过单元相同则要求平均同股AUC≥+.02且平均Brier恶化≤.005。其余回退；AUC未定义不补值。完整TP/FP/n、供给、信号日期/个股、拒绝、九目标概率质量与终值估值照报，5/10官方session块区间只作暴露开发诊断。任何保留都不代表90%目标完成；诊断折及P1待独立复盘另议。

主要成本为60股日级来源重建、旧大NPZ复制/解压及改动谱，不新增网络；旧两折九头实际约6.1/8.0秒仅作拟合参考，不是新任务时长承诺。实现设单worker、20分钟观察检查点，预算到只停并留失败/进度，不缩股/缩键。预计数据门禁比拟合更贵；无需等待余240片或群重建才能完成本实验的数学输入，但需等待根采用与Sol资源。
