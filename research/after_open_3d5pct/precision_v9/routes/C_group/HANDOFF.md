# C 有群 · root / Sol 交接

2026-09-27。方案/复盘由GPT-6 Astra xhigh；纯代码由root交GPT-6 Sol xhigh。**本次未实现、未训练。** 只在root后续分配范围开发；不改`acquire_batch.py`、`acquire_pilot.py`或共享旧制品。P0先于P1，参考 [STATE](STATE.md) 的E1–E6证据。

## P0：共享数据接口与原配方

输入必须显式接收`dataset_manifest/calendar/schema/split_row_keys`，训练器不联网。保留04:00–20:00和7日5m/30日60m/126日形状；新夜盘留档，不混入本轮。共享Sol负责146日源回看、公司行动、半日聚合、群来源与旧新2026行配对；本路线消费审计，不能因文件存在即标数据通过。

原模型使用`focus_v8.support.build_c('C_group', recipe)`和旧fit流程；recipe严格只有`balance/representation/calibration/support=true`。不能直接使用`precision_v9.train.CHANGES['C_group']`带回失败dropout。AdamW lr=.001/weight_decay=.01，九头平均BCE、batch256、32epoch/patience5、seed3566；权重、校准及动作算法不顺便优化。

P0a：新回看但旧fit keys；P0b：同新快照、相同tune/cal/eval keys，fit增加已登记2025合法日期。若新builder排除旧行，先旧数据在相同交集重拟合桥接；旧全域预测另存legacy reference。每个比较保存`symbol/session_date/decision_at`映射、标签/entry/terminal与短窗/长窗/群输入差异、拒绝原因。数据语义修复与增日期分开解释。

三折边界仍引用`focus_v8/protocol.json`；同日全股同侧，最长九目标完整label_end/available严格在下一段前。每段保留官方全部session和实际合格日期，cal未成熟行不转0，不为了增加cal改成熟窗。

“保持校准/阈值”指保持旧方法、权重、日期及选择规则；新模型在自己的cal拟合新参数/选新阈值并保存，旧模型参数及已选阈值不可覆盖。raw主头仍在.30–.95/步长.05网格，cal≥90%、n≥5、≥3信号日、供给≥1，按n/precision/阈值排序。eval不选阈值；去重须用传入新官方日历，不能继续隐式调用2026-only日历。cal支持审计可引用旧9/9/10日期作对照。

P0交付：registration、bridge/paired报告、全fold支持表、九头fit/tune/cal/eval raw/p/y、模型/校准、完整cal曲线、eval signals、成本、哈希与复放。研究代理复盘后才冻结P1实际输入和路径。

## P1：bounded_group_residual_v1规格草案

唯一主变更是群融合方式，输入/标签/权重/训练及选择流程固定；无新行情或成熟群标签。

1. 使用现有representation后三个PatchBranch各16维，拼成本股`u∈R48`。本股头`h(u)=Linear(48,32)→GELU→Dropout(.1)→Linear(32,9)`。
2. 沿用类型特定`member_weight[6,10,12]`/bias、六周有效槽池化、`GRU(12,12)`，得到`q∈R12`；不改变slot、字段、QQQ语义。先用原fit-only support屏蔽，buffer随checkpoint保存。
3. 群修正`r=.5*tanh(Wq)`，`W∈R(9×12)`无bias、零初始化；仅有108个输出层权重，无群→本股乘性门，无relation直接拼入h。
4. `R_logits=h(u)+r`；`Z_logits=h(u)`。当整段群序列在support后全无有效槽时令r=0，防止GRU偏置制造无数据群信号；这只是缺失处理，不是择日门禁。
5. 保留九头sigmoid raw、原cal方法和九目标单调投影p；动作只读raw_3d_5pct。保存h、r与最终raw便于逐行归因；r界限是logit幅度，不能称校准概率或90%保证。
6. R、Z本股分支/头用同一初始化副本、相同样本顺序及本股dropout随机流；群分支单独种子流，避免创建模块改变本股初始化。Z无需群参数训练，不能借用另一路线的prior/specialize checkpoint。
7. 每臂1固定配置/折，先两开发折4次新拟合；保存开发决定后才做已暴露diagnostic两次。I为P0相同数据/keys的原C_group。若对照I需重拟合，单独列计算预算；不更换种子挑结果。

保留规则完整见 [BACKLOG](BACKLOG.md)：必须同时通过R对I总体改善门槛和R对Z群增量门槛；六群折均报告，无群增量不算C_group机制成功。规则/实际manifest/代码哈希在拟合前落不可变registration，任何后续变更另升版本。

## 必要验证与逐行报告

- checkpoint复放最大误差≤1e-7；复制同一份本股分支/头权重、置eval模式后，W=0的R与Z raw逐行相等（不要求各自训练后头权重相同）；任意输入r均在[-.5,.5]，无有效群r严格0；group参数与主头有有限有效梯度。
- Z对任意群/QQQ输入修改不变；R可能受当时合法群信息影响；改未来/迟到bar、下一周成员、当前/未成熟标签均不得改变历史特征或固定模型预测。支持mask只能来自fit，不随eval开放。
- 日期块而非行重采样；报告raw/p pooled及同股AUC、同股正负配对数、三群TP/FP/n/precision/供给/信号日期、拒绝/冷却原因、全部九头Brier/校准及terminal风险。动作信号按原时间冷却，不能按是否提前成功解禁。
- 高分误报诊断同时保留TP，按日期/股票/群support/数据完整度列明；r正负/饱和比例及R-Z逐行差异只用于解释，不在eval上再设阈值。重现E2的57/34/23与两日21FP作为只读聚合核对。
- 交叉群一条信号联合只计一次；与其他路线信号/误报交集交root统一报告。研究预测、成熟价格触及、实际成交和资金收益始终分开。

失败输出具体层级：数据无法配对、工程复放失败、机制门槛未过、前向样本未积累。不得以已有2026开发结果填独立通过；不得为了维持任务完成状态修改90%或供给目标。
