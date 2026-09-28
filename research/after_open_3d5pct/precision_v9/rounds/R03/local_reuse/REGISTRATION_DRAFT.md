# R03 本地复用入口登记草案

2026-09-27；**DRAFT，未采用、未实现、未启动**。本页为根与后续Sol coder提供可审查的有界方案，不修改已冻结RECOVERY_ADOPTION，不批准更改其他任务。方案/复盘由Astra承担，纯实现由Sol承担。目标、五路线、108 candidate＋QQQ、Aug–Dec 2025、原2026、标签与三臂比较保持原登记。

## 建议采用范围

建议新入口/身份 `r03_acquisition_local_reuse_v1`，新输出 `runs/precision_v9_20260927/R03_recovery_local_v1`（启动时必须不存在；若占用则另登记新名）。原R03、已初始化R03_recovery_v1及其冻结字节保持不变。不得让新逻辑复用旧identity冒充已登记的recovery_v1。

最小实现为独立离线来源验证/重放助手＋新恢复入口；prepare只新增明确版本的来源验证分发，原R03/recovery_v1分支保持原要求。不要只往旧恢复器Local路径列表加目录，也不要以tier=futu伪装本地重放。此处仅规定设计，不现在修改prepare或任何代码。

## 根接管顺序

1. 根先审查并采用本页或修订后的具体登记。采用前不操作现等待器；另一任务PID96498、其文件、OpenD及共享锁均不动。
2. 只有根确认会话22976仍是等待状态、R03_recovery_v1仍无parts/网络工作，才可结束本任务自己的等待器并记录时间及状态；保留初始化证据。若它已开始采集，不按“尚未请求”假设中断，先保全实际checkpoint，另审查承接范围。
3. 根安排Sol实现新入口与门禁/离线fixture，Astra独立复盘通过后冻结源码、登记、日历、normalizer、来源策略及运行参数。新入口仅在单一共享锁下运行；必须由另一采集任务自然释放锁后继续，不另开竞争采集器。
4. 获锁后重新盘点实际545范围，不能沿用本次0/240或未来230的预估。先冻结所有可用Local候选；原305逐片验真继承，再处理原缺240。使用新输出保存开始/结束状态、采集与导入分母。

## 来源与规范化规则

优先顺序固定为：原R03已完成片→已验证的范围内Local（含独立月库与原白名单）→R2→OpenD。Local之间若出现矛盾先记录并停止该片，不能按模型效果或便利顺序选值。原305始终使用原R03；外部同键任何修订另列，不覆盖。只拼接原冻结2026文件和公司行动表，不导入外部2026替代它们。

每个外部provider请求先建立独立source bundle：复制registration、相关源码/日历/metadata（若引用）、complete、imports、请求result/audit、全部raw页及normalized、涉及的revision resolution/differences。复制前后核原文件SHA并核副本SHA，提交source manifest后只用新run副本重放。避免symlink到仍可变化的外部月库；请求组只复制/重放一次，由各月引用同一冻结bundle。代码/registration不匹配或源在复制期间变化则不提交。

必须核验：

- symbol、冻结月份包含于请求范围、5m/NONE/ALL来源配置；原始页hash/行数与audit、成功页号从1连续、非末页has_more=true、末页false、result已提交且非failed、全部关键文件存在、请求时间≤实际接收时间。raw中证券与时间范围一致，不允许重复time_key或选择性漏页。
- 外部complete先写、请求result后写；complete已存在但result缺失为pending，不能当完成。原始audit没有保存请求cursor或全部参数字段，manifest如实记录“registration＋匹配源码支撑”，不虚构逐页参数或provider承诺。
- 用冻结R03 normalizer从已核验raw重新生成canonical列，再按ET session_date切月；遵守月末午夜、DST、半日。不要直接信外部已标准化bars。三例逐值相同只是先验可行性，全部导入月仍需重放比较。
- 重算R03未作上市折扣的月级quality，并报告extended session实际观察量。分页完成、月RTH完整、非空、历史资格、模型可用是独立字段。坏OHLC、重复、来源矛盾记unresolved quality/provenance，保留证据，不能自动重拉后抹掉失败。
- 对月bars与raw重放差异列出行增删/字段差。若外部bars合并过多个来源，不直接继承合并表；仅在所有参与源/修订链可冻结且选择规则明确时承接。优先完整新provider原始范围重放为新来源，旧R03同键仍优先；不能静默keep-last。当前无真实revision样本，必须fixture覆盖且未来实样单独核验。
- 完整、有效provider响应但月覆盖有缺口，可保存`observed_partial_coverage`及逐日unknown缺口，不宣称完整。准入训练仍由prepare/样本覆盖规则决定；缺口不是负例，不填bar。不能为了达到质量统计而删证券或裁短月历。

真实received_at按来源audit保留；新复制/导入另记imported_at。历史available_at仍按原bar_end＋1秒假设，不能用实际回填receipt倒填成2025实时到达证明。

## 空月与十个疑似上市前分片

**验收目标是545分片的来源/请求状态齐备，不是545非空RTH完整。** HONA/SPCX的原冻结2026均非空；五个2025空月可沿用`join_2025_parts`已有合法短历史语义，不改变108候选作用域，也不要求虚构126日历史支持。

空月仅接受以下两类真实provider证据：一个完整零行请求；或完整覆盖该月的多月请求全部raw页验真、该月确实零行。后一类必须用新version的范围证明，不能伪造一页“月度零返回”去骗旧门禁。结果保存canonical空表、provider-complete/empty及unknown覆盖、真实audit/raw引用，RTH完整=false。

外部before_current_regular_trading仅metadata声明、无请求raw，**不能转成上述空成功**。HONA/SPCX预期10片先尝试合格本地raw证据，再按R2→OpenD补缺。原HONA/2025-08原始零页可以成为候选证据，但未提交分片的承接必须明确登记、核原audit/raw SHA，不追认原run完成数。若暂不实现这一特例，可保留原失败证据并按已登记有限重试获得新响应。

若拟以后仅凭官方证券身份/上市史免发这类请求，须单独验证原文/身份/场所与时间证据、登记新的metadata-backed gap状态及prepare接受条件；不在本草案中用现有metadata替代。SPCX metadata与冻结行情最早日期不同须列冲突，禁止用它过滤原2026。空成功只证明供应商未给该月bars，仍不能证明上市、停牌或无成交。

## 后续网络与失败规则

Local来源缺失或无适用数据才进入R2/OpenD；存在source hash错误/修订矛盾属于审计失败，不能隐瞒后直接走网络。对仍需网络的片保留既有单worker、共享锁、首页/分页间隔、30页/月、3次/页、180分钟预算与no_upload。新入口增加的本地阶段耗时单列，网络预算不得因来源难验隐式扩充；总预算不足则checkpoint。

成功非空完整、成功非空部分覆盖、成功空、metadata-only、quality/provenance失败、请求失败、pending分别计数。metadata-only不计provider完成；全部545身份齐备且无unresolved/pending才可叫acquisition_complete，仍需单列unknown覆盖。失败/未提交新片保留原样，另登记后继run，不自动无穷重试、删除失败或覆盖旧文件。

## 离线验收与交付门禁

至少覆盖：完整多月raw切月；原305优先和外部冲突；缺页/错hash/重复时间/错证券/错basis/session；complete已写但请求result未写；源复制中变化；合并修订/无修订证明；午夜、DST、半日；整请求空与范围中零月；metadata-only拒绝；HONA/SPCX五空＋冻结2026非空；2025与2026皆空拒绝；未知version拒绝。测试不得调用R2/OpenD。

真实运行后先做545来源与覆盖验收，再prepare与paired_2026_audit。原2026 entry/terminal/九标签变化须停止并审查；新历史改变输入/群/样本支持如实报告。通过数据验收后另签actual_manifest和五路线baseline登记，不直接使用外部整库训练、不改变90%及供给目标、不称独立验证。

当前基线源SHA见 [SUMMARY.json](SUMMARY.json)。这些是本次审计指纹，实施/启动前必须重核；源变化须留下新登记，不能改本页去迁就已运行结果。
