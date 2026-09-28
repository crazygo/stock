# 全候选成熟先验与本地端到端适配：下一轮登记方案

2026-09-27；仅方案，不开发、不采集、不启调度。依据现有 `forward/{features,inference,ledger,labels,evaluate}.py` 和 [特征审查](../forward/FEATURE_CODE_REVIEW.md)、[账本审查](../forward/LEDGER_CODE_REVIEW.md)。不改变模型路线、九目标、主目标、群成员或最终90%门槛；当前0路线通过。

## 实际状态

| 层级/接口 | 已有证据 | 尚缺 |
|---|---|---|
| G1静态模型 | 五路线真实checkpoint九raw/processed最大误差≤1e-7 | 使用冻结batch tensor；不是五条feature-only全链路 |
| G1特征到输出 | B_no_group、C_no_group真实feature-only配对；9项定向测试通过 | B_group、C_group、C_no_daily三条带群路线真实同伴全链路待测 |
| 账本/标签/评价 | 23项fixture通过；真实AMD feature-only预测送issue_once按缺receipt拒绝，时点契约已通 | 单股链路检查已具备，仍缺全候选prior及完整本地E2E适配器 |
| `_prior` | 接收内存list，按本股决策及九标签完成/可用时间筛最近63行，Beta(1,1) | 没有全候选持久登记/成熟/修订/as-of快照；metadata的all_candidates及hash声明不等于已验源 |
| `evaluate_candidate` | 任意候选的纯九目标helper可直接复用 | 不落先验库；`mature_signals`只更新已有BUY，不能当全候选库 |
| G2/G3 | 尚未证明 | 真receipt、群/元数据真实发布时间、冻结动作制品、30秒性能、完整未来日历、真实cohort与成熟效果 |

日历缺口见 [CALENDAR_REQUIREMENTS.md](CALENDAR_REQUIREMENTS.md)：共享表至2026-10-15、R03表至09-24，均不足新60session cohort及最后1950 RTH分钟标签。现有v8诊断manifest动作未冻结，不可直接转正式启用。

根代理最终验收：feature9+ledger23=32项联合测试通过；cohort.registration wrapper冻结漏洞已修复，register/issue/evaluate三入口均拒绝包装对象并冻结完整CohortSpec。现有feature审查中的31是更早联合计数；本方案采用最新32项状态，不代表新增集成已通过。

下一步可独立产出v9日历artifact延至2026-12-31并保存官方Nasdaq/NYSE来源bytes、取得时间与SHA；根已实时核对2026闭市和11-27/12-24的13:00收市，但本任务未生成日历，缺口仍在。`scripts/model_history_calendar.py`是他人WIP，可只读参考，不能修改或无版本直接依赖。若cohort最后1950 RTH分钟超出新表，必须拒绝并另登记更长版本，不能用workday猜测。

## 本轮可证伪假设与 backlog

假设：在不改变冻结先验公式与模型数值的前提下，用全候选追加库生成可追溯as-of先验，能够使既有feature→inference→ledger→labels→evaluation在本地串联；任意未来追加/修订不改变旧输出，且非BUY成熟行确实进入prior。

| 顺序 | Sol交付 | 必过证据 |
|---|---|---|
| P0 冻结三份清单/时钟/schema | prior_universe、forecast_candidates、required_market_dependencies；记录采样资格来源与版本 | prior包含全部冻结candidate，QQQ等benchmark不冒充候选；不能缩成21操作股或BUY集合；同伴依赖不按操作池裁剪 |
| P1 all-candidate store | 候选登记、九结果追加/修订、未知原因、as-of读取、不可变snapshot | 每候选日一次，不因route/group/阈值重复；全九成熟才进prior，去重/幂等/重启不变 |
| P2 严格本地loader/adapter | 真实文件→snapshot/metadata/prior_view→现有接口；失败也入审计 | 验实际字节与内容绑定，不设置无证据的verified布尔值；不调用data.build或读取当前未来标签构造X |
| P3 五路线数值/E2E | 三带群真实配对补齐；五静态与无群既有回归；fixture账本全流程 | group_seq及raw/processed配对，有错误即该路线G1未过；fixture永不宣称G2/G3 |
| P4 运行成本登记 | 冷/热缓存、载入/解码/特征/推理/哈希/事务/报告分段耗时 | 固定全依赖规模、线程/内存、硬件、源hash；未测/超30秒继续标未知/不满足，不放宽deadline |

## 方案矩阵

| 方案 | 先验语义/因果性 | 持久/复现 | 工程成本 | 选择 |
|---|---|---|---|---|
| A 直接用BUY outcome喂 `_prior` | 只剩策略选中样本，改变本股基率与输入分布 | 账本现成但来源范围错误 | 低；错误不可接受 | 拒绝 |
| B 每次从当前完整batch数据重建内存list | 依赖未来全九标签，实时漏当前；修订/回填边界难证明 | 历史诊断方便，不能替代前向持久库 | 低至中 | 只保留冻结fixture对照 |
| C 独立全候选追加库+不可变as-of快照+薄适配器 | 保留所有预定候选/非BUY；先成熟再进入；版本按实际可知时刻筛选 | 可跨run恢复、行级sourcehash、旧snapshot不变 | 中；复用现有纯函数 | **采用** |

## 给Sol的接口与字段

建议仅新增 `forward/prior_store.py`、`forward/local_adapter.py`、定向tests/只读运行手册；必要的既有接口绑定修补明确列差异，不重写训练器/采集器。代码代理与另一个模块owner先锁定dict契约，不能各自发明第二套预测schema。

适配器只传plain完整CohortSpec：route_contracts、spec/calendar、session全集、groups/routes及门槛全部属于同一冻结对象并参与hash；存在 `registration`包装字段直接拒绝，注册/issue/evaluate共同使用同一完整对象。必须验证篡改任意外层字段会失败，不能靠旧wrapper哈希相同放行。

```text
register_candidates(universe_version, official_session, eligibility_manifest, store) -> candidate_ids + coverage
update_candidate_outcomes(candidate_ids, local_outcome_snapshot, calendar, asof, store) -> appended_versions
read_prior_asof(symbols, information_deadline, store_cut, store) -> immutable PriorView
load_local_bundle(input_manifest, prior_view, mode) -> verified snapshot + metadata + decisions
run_local_e2e(plan, new_output, fixture_ledger, prior_store) -> artifacts + stages + gates
```

候选主键 `(universe_version,symbol,session_date,cutoff,label_contract)`，保存sample_id、11:30/11:30:30/11:35时刻、当时资格/缺失原因、成员与输入证据；登记不看未来命中、分数、threshold或BUY。不同route共享同一行情标签事实，不重复累计prior行。缺历史/缺前缀等候选仍留覆盖登记，成熟资格按冻结共同样本规则处理，不能回填后静默改原资格。

追加结果保存九目标各自status/y、label_end/label_available、evaluation_asof、source/receipt/action/calendar哈希、recorded_at、版本及supersedes。原始标签可用时间与本系统实际知道时间分别记录；读取先按截止前已记录/收到版本选择，再检查全九完整、二元0/1及过去时点；未成熟/缺失/未知不填0，正负统一等满窗口。修订不得改旧snapshot。

`PriorView.rows`适配当前 `_prior`所需 `{sample_id,symbol,decision_at,label_end_at,label_available_at,y[9]}`，输出前拒绝重复IDs/错误维度/非二元；live视图的有效label_available不早于源实际receipt与该版本recorded_at。按原公式由 `_prior`筛本股最近63行，不能另设成功优先或只取BUY。提供store_cut、全候选分母/各状态计数、选中IDs与版本/来源hash、view文件SHA及max_known_at。

冻结历史bootstrap可从已登记历史全候选数据导入，但其实际导入/收到时间就是现在，不能伪造过去receipt；历史数值对照明确historical_fixture。缺来源/资格的bootstrap标未验证，不用一句all_candidates就放行G2。后续补记候选须保留真实补记时刻与原因，不声称当日已预测。

适配器从实际路径加载与验证 bars、群版本/成员、公司行动、日历和PriorView，再构造已支持的dict；snapshot manifest绑定这些字节、解析内容及prior所选版本。现有 `mature_prior_scope/all_candidates`、`mature_prior_snapshot_sha256`只能由验源结果产生。lineage应带prior_view/IDs/hash，并把prior、群和公司行动的最晚实际可知时间纳入发布检查，不能仅使用bars的latest_*冒充全部输入可知。

时钟按现有接口：`decision_at=cutoff_at=11:30`，`information_deadline_at=prior_decision_at=11:30:30`，planned_entry=11:35；账本检查实际发布时间与latest_available/received。禁止适配器把decision改成11:30:30以迁就旧命名，禁止把截止后结果回填发布。九raw/processed与 `quality.artifacts.files`直接沿用predict_route，不重新计算或抹去其G2拒绝。

feature拒绝/模型失败必须映射逐候选拒绝审计，不能填假0/.5概率或整批删行；仅真实无候选时记run_empty。输出同一固定candidate全集的成功/失败计数，非BUY结果更新prior store，BUY结果另调用 `mature_signals`，二者signal_id/sample_id对应可审计但分母不混用。

## 本地验收与失败门禁

1. 真实历史两无群及三带群路线依次 feature-only→真实checkpoint，X/先验IDs/九输出与冻结原制品配对；带群覆盖同伴、QQQ、六周group_seq与公司行动边界，误差>登记容差（现有参考1e-7）即失败，不靠换样本过关。
2. 全候选fixture含非BUY成功/失败、BUY成功/失败及未成熟/缺失，证明最近63与Beta(1,1)一致；重复/重启/多route不增加prior计数，pending提前触及不进入，晚到/未来修订不能改旧prior/X/预测。
3. 本地真实历史路径→账本应保持historical_fixture的不可发布标记，得到明确拒绝；另用完全合成且fixture命名空间的receipt场景走模拟buy→成熟→全N/期末评估。不得改真实历史预测quality使其假装live。
4. 全链路包含已有N=5、TP=2、FP=1、pending=1、missing=1案例与跨run cooldown/trigger union/60session无信号和停机分母；新prior store的增删修订绝不能改变BUY分母。
5. 篡改PriorView、母snapshot、群/公司行动字节、遗漏候选或receipt/日历不足时失败或明确未验证；不能只匹配字符串SHA。真实动作未冻结、日历不足、群三路未配对、30秒未知、G2/G3未证实时分别列门禁，禁止总状态“可上线”。
6. 保存计划、源码字节与before/after SHA、模型/数据/先验快照、stage结果、全部失败和耗时；本轮只交离线适配，不创建真实cohort，不部署、采集、调度、上传或交易。源已被他任务改动时停止并重取快照，不能沿用本页旧SHA声称通过。

## 本轮审查源码指纹

| 文件（forward/） | SHA-256 |
|---|---|
| features.py | `65fedae4ddcc2cda359846e99f019b175c01a73ac8eaa0251d2e4b1208002de0` |
| inference.py | `e8ef043748b7542bc6cb38a0f891dc96d7456792d20f68856daee55bfe79eae8` |
| ledger.py | `7217789b5deab63e8d54dbbc2e090276dc65a9860db6afec75adf2c2b784af0f` |
| labels.py | `3eb08f3634c9a0299bd1a5bd2bf44cec2a842aa44f4994aba806b9571137f2e0` |
| evaluate.py | `899acea925cf447d252273ea34c2c9ac29c084afd4dff3d9abd779f56c8bcea3` |
| FEATURE_CODE_REVIEW.md | `069249e8ee89f0c7996b05428e225e9e932efb88833a4749889f25e819415673` |
| LEDGER_CODE_REVIEW.md | `cd50aa917e03f5ffb6ace0f120b587106f72206185a3c0e75009997843bab228` |
