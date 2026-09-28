# Forward 群门禁 / prior / local adapter 独立复盘

2026-09-27。**本轮限定合约修复已独立复验并采用**：prior seal 与新增分类 ID、版本引用、完整候选集合门禁均通过定向反例。三带群正向 G1、G2/G3 未通过。本审查未改代码/数据、未训练、未启调度；冻结 [INTEGRATION_PLAN](INTEGRATION_PLAN.md) 与 R03M 四件保持原字节。

## 实际状态与已核对结论

- 已读 [GROUP_GATE_REVIEW](../forward/GROUP_GATE_REVIEW.md)、[GROUP_GATE_REGISTRATION](../forward/GROUP_GATE_REGISTRATION.md)、[LOCAL_ADAPTER_CODE_REVIEW](../forward/LOCAL_ADAPTER_CODE_REVIEW.md)、[PRIOR_STORE_CODE_REVIEW](../forward/PRIOR_STORE_CODE_REVIEW.md) 及对应源码/测试。
- candidate 与 QQQ 边界修复符合旧 builder：features 的 peer map 只取 candidate 并显式排除 QQQ，行情 views 仍保留 QQQ 第六槽。AMD 六周趋势差异已由此前独立数值核验关闭；不外推为全部股票/日期配对通过。
- 有效 classified own/peer 的 history_end 缺失、畸形、当日/未来会返回 X=None 与稳定 metadata 拒绝；合法未分类空群仍 mask。拒绝 lineage 有成员、周、代表日、字段/原因及内容指纹。该内容 SHA 不是源文件 bytes SHA。
- `legacy_static_only` 群 schema 无法走 feature-only；旧静态 replay 的 quality.g1_ready=false、G2=false。causal schema/feature lineage 须握手，加载需训练 provenance；旧 `focus_v8_` 版本不能冒充 causal 模型。入口约束本身不证明任意 provenance 声明的真实训练来源。
- PriorStore 为完整 candidate universe 登记，不依赖是否 BUY；实际 capture time 与 deadline 取较早者，结果 known time 包含记录/源/动作 receipt，九目标完整成熟后才进入先验。历史导入保留实际导入时间，不能用当下回填当过去可知。
- adapter 仅接受 fixture ledger；三列表分别表示 prior 全池、forecast 子集、实际行情依赖。成功 feature 的 used_market_symbols 与声明依赖比较；缺模型/输入或 feature 拒绝保留逐候选 audit。它主动保持 G2/G3=false，不把本地 hash 等同收到证据。

## Prior seal bug：修复核验

修前 `_checked_input_hashes` 捕获 `verify_prior_view` 异常后，只比较未变的 view 文件 SHA，可能放行已损坏 seal/hash chain。修后保留 `seal_error` 并无条件加入 mismatch；strict pre/post 检查抛错，fatal 保留相同文件 SHA 与 seal 失败，已记 fixture BUY 按原实现追加 protocol_violation、不删除分母。

已亲读回归：ledger 返回后仅损坏 `view_created.payload_sha`，view 文件字节不变，要求 fatal 与 seal_error。本次独立执行 group_gate、inference、prior_store、local_adapter **21 项通过，5.993 秒**。这是定向验证，不是本审查重跑完整50项；开发者的50项绿测也不代表 G1/G2/G3。

## 独立发现与补修条件

| 实际反例 / 缺口 | 观察与层级 | 当前处置 |
|---|---|---|
| `trend_15:unregistered_category` | 修前直接 features 接受，X 非空且 mask=1，只验证前缀 | 已修；独立反例得 X=None / group_metadata_incompatible |
| classified own/peer 指向 `missing_version` | 修前直接 features 无拒绝，静默 mask=0；adapter 另有未知引用检查 | 已修；own/peer 独立反例均显式 missing_or_mismatched_version |
| group universe 未绑定完整 prior universe | 修前直接删 peer 可得空群；adapter 未与已冻结全池比集合 | 已修；按 role=candidate 绑定完整集合，缺/多/重/混 QQQ 全拒绝 |

修前直接 features 反例已独立实跑，adapter 集合缺口先由源码确认；精确输入/输出见修前 [GATE_FOLLOWUP_REGISTRATION](GATE_FOLLOWUP_REGISTRATION.md)，其字节保持。Root 要求立即修复，由 adapter owner 执行；R03M owner 不改 features。

修后亲读 preflight 与集合绑定实现，独立再次运行三种直接反例，全部 X=None 且原因正确；合法群和 QQQ 独立槽 mask 均为1。再独立构造完整 AMD/COHR prior 全池、仅 AMD forecast，逐份重新计算 group 文件/manifest SHA：完整集合成功，漏 COHR、增 XYZ、重复 COHR、混入 QQQ 均触发集合拒绝，未借文件 hash mismatch 代测。另独立运行 group_gate/features/local_adapter **16项通过，6.776秒**，覆盖合法未来/非依赖版本与空群不误拒；既有 seal 同SHA回归仍在该 adapter 测试中通过。

## 证据层级与未来兼容证书

- 五路线旧 checkpoint 静态九目标复放仍是既有工程证据；两无群 feature-only 配对保留。三群首次数值 FAIL 保留，新 test_group_parity 绿测只证明正确拒绝旧缺证据 metadata。
- adapter 真实 AMD→B_no_group→ledger 拒绝证明本地链路；其历史导入 PriorView 没有 June 当时可知记录，不能代替原58行先验的数值配对。`g1_local_path` 不等于 `g1_numerical_parity`。
- 五个合成 BUY→九标签→60 sessions 分母测试在独立 ledger/labels/evaluate 组合路径，不是五真实路线经过 adapter 的全流程；TP=2、FP=1、pending=1、missing=1、N=5 只验证未解决买入不消失。
- R03M 当前已登记并开始轻入口工作，尚无全键结果。本轮不改其 [REGISTRATION](../rounds/R03M/REGISTRATION.md)：全键输入等价才形成兼容证书、不重训；群有可解释差异才另登记 refit。
- loader 目前无等价证书路径。未来扩展须绑定旧模型/原 schema、重建 metadata、全旧 keys 输入 parity 报告与目标 schema SHA；保留原模型身份、默认拒绝和 G2=false。不能改 model_version、伪造训练 provenance 或把局部匹配扩成全池证明。
- 独立日历已产出 [calendar_2026_official_v1](../forward/calendar_2026_official_v1/CALENDAR_CODE_REVIEW.md)：251 sessions、两官网原 bytes/SHA；root 已离线 verify 退出0。本审查只读取交付报告，未重跑其4项测试。**cohort coverage 仍按具体 start 另验**，须覆盖全sessions及最后1950 RTH分钟，不延伸声称2027可用。真实 receipts、全池30秒性能、真实 frozen cohort/成熟结果仍待证明。

## 本轮决定与后续

采用 seal 与三个新增门禁的限定合约修复，继续保持 local_fixture_only。下一步是 R03M 全键审计→必要时兼容证书或另登记匹配训练→真实接收/性能/日历门槛；详细队列见 [BACKLOG](BACKLOG.md)。本轮测试绿关闭的是具体错误接受路径，不能跳成三群 G1 或90%通过；用户目标仍0模型。

## 本次审查字节

| 文件 | SHA-256 |
|---|---|
| features.py（新补修前） | `810c7fd9f7c7d996534e02ed7c42d549a1f18c5754a14e8a78c4f599b418a5c1` |
| inference.py | `e2a0667c5d44cc62c5760a10fa6d5571f7c236e78488d4b2f5d5cf349bb9ded3` |
| local_adapter.py（seal已修、集合补修前） | `403ea3246e2b346e0c83eb2d7216726b87de1916ec4e4dddf451b0988696fa4c` |
| test_local_adapter.py | `1c93b19b0d1a82d9f0c2bfcd5285e34f789daaa68bd8573847d529b3cb188d5d` |
| prior_store.py | `8e45732f14204d68d309846188bd7e47e6cd6802d795079a158a7c435c6f9786` |
| 原 INTEGRATION_PLAN.md（保持） | `2e8431c5fe63a259a4c55f9357ae508c7000018f9739c912d2b1b456277d85c3` |

修后采用字节：features `528344d95bab167fb520584e854c0223e2ccba2b592b3fcaa5a2a9818229e646`；local_adapter `0987c1d1425addf81fffcc18cb35d725b46352f142e44a55c2669e41eccdbf98`；test_group_gate `5606f4abe9ea9bfdb0e135e39db4fce26caade987c2fc89af1f444f4992709fc`；test_local_adapter `e93e08a080282ccb8dd8d4e8e8208353aa04eb66f909b1ae816dc9bba51114fd`。inference/prior_store 未变；不覆盖原失败 JSON/诊断字节。
