# R03 本地新来源复用：实际状态

2026-09-27；仅离线审计与方案草案，未采用、未实现、未采集、未训练。唯一写区为本目录。未操作根等待会话22976、占锁PID96498或其他任务文件。目标与108 candidate＋QQQ范围不变，90%目标仍未达成。

**结论：来源结构支持有条件复用，但当前剩余240片中可立即复用为0。建议在另一采集任务自然释放锁后重新盘点、冻结并验证本地来源，再仅对未解决分片走R2/OpenD。** 原305片继续直接继承原R03，不替换成另一库副本。

## 本次实际盘点

主证据捕获时间：12:56:46–12:56:50 UTC。精简结果在 [SUMMARY.json](SUMMARY.json)，全清单、逐文件指纹和代表分页证据在 [evidence_20260927T125646Z.json](evidence_20260927T125646Z.json)，SHA-256 `00b39299a4c5feb8ab40726c25793cddef82decf3b21c0c9841c12281acb736d`。

| 范围 | 实测 | 能说明什么 |
|---|---:|---|
| 冻结R03 | 109证券×5月＝545片 | 范围未变 |
| 原完成片 | 305片、1,851,996行 | 全部bars哈希与原result一致；重算quality一致、RTH完整 |
| 另一库对应原305副本 | 305片逐列逐值全等 | 没有发现这305片的内容冲突；仍优先原来源 |
| 原缺240片 | 0 complete、0 bars、0共享2025年文件 | 当前没有可立即接管的新增片，不能用全库数量替代交集 |
| 另一库全部complete标记 | 1,089个 | 557 source_returned、305继承R03、189 existing_local_or_r2、38 before_current_regular_trading；仅库存，不是质量认证 |
| 完成标记但无bars | 38个 | 全为上市/恢复交易前声明，不是供应商空响应 |
| source_revisions记录 | 0个 | 代码支持修订，当前未观察到实样；不能声称实测通过修订路径 |
| 本次读取并核前后哈希 | 1,694文件，变化0 | 只证明本次读取窗口稳定；源任务仍在运行，尚非冻结导入快照 |

另一任务先按证券补2025更早月份、2024、2023，再处理2026；当时日志到STX，尚未到原缺片的HONA及后续48证券。原恢复器的Local白名单只有共享`market_data/us_5m`与R02，不含`market_data/model_training_history_v1/parts`。因此“等待其完成后原恢复器自然复用这些独立月片”目前不成立。

## 代表抽验与边界

以下例子用于检验来源结构，均不是已补好的剩余240片。

| 样本 | 实测 | 判定 |
|---|---|---|
| AAOI 2025-01 | 5,681行；RTH缺0、坏OHLC 0 | 完整非空月示例 |
| SNDK 2025-02 | complete=source_returned；3,105行；按完整官方月历缺624 RTH、117坏OHLC | 请求完成不等于行情完整；不能直接接纳为合格完整月 |
| ALAB 2024-03 | 2,016行；严格月历缺1,014 RTH；上游按上市后所需日期评估为完整 | 上市折扣后的请求quality不可代替R03月级quality |
| SNDK 2025-01 | before_current_regular_trading、无bars/raw | 只能保留元数据声明，不能制造provider-empty |
| CTAS 2025-11 | 5,389行；RTH缺0，含半日市；与原R03全等 | 可保持原305的冻结引用 |
| AAOI 2026-01 | existing_local_or_r2＋imports源哈希匹配 | 标记本身没给raw/receipt，须沿imports继续追源；2026不属于本次替换范围 |

AAOI、SNDK、ALAB三个完整请求分别42、34、57页，共133页：原始页哈希、审计行数、连续页号、末页has_more=false及前页has_more=true均通过；无重复time_key。所抽三个月按现有R03 normalizer重放，与另一库对应月全部列逐值相同。2025-08至12两份日历的日期/开收盘/分钟投影全同。这些是可复用性证据，不把三例外推成全部源已验收。

实际received_at保存在每个请求audit；它是2026-09-27回填接收时间。bars.available_at仍为历史bar_end＋1秒假设。外部audit记录symbol/range/page/attempt，但没有逐事件ktype/autype/session/max_count或分页cursor；NONE/ALL/5m参数由registration及匹配脚本字节支撑，不能伪称供应商逐页回显。当前script SHA与registration一致。

## HONA/SPCX：545状态齐备不等于545非空

外部security_metadata声明HONA上市日2026-06-15、SPCX上市日2026-06-12；按该脚本，二者2025五个月预计产生10个before_current_regular_trading标记。这只是本地元数据与代码的推断；本次没有联网验证上市史，也没有观察到这10个未来标记。其他230片最多是潜在非空候选，实际完整数可为0–230，最终必须逐片验收。

现prepare的判定路径明确：

1. `validate_acquisition`要求545片都有已登记身份、完成请求、无failure、bars/result/audit及哈希/quality一致；**没有要求545片非空或全部rth_complete=true**。
2. 已支持的recovery空成功必须有canonical零行表、真实成功零行raw、末页has_more=false和匹配审计。空月写入successful_empty_parts及unknown_coverage_gaps，不能叫完整行情。外部metadata-only标记缺这条证据链，不能直接通过。
3. `join_2025_parts`允许五个2025空月，只在2025与冻结2026均空时拒绝。本次读取冻结HONA 2026有13,572行，SPCX有14,347行；二者已有合法短历史输入，不能为了五个月空白移除候选或制造历史。
4. SPCX外部metadata日期与冻结2026最早行情2026-06-09有差异。保留冲突，不用metadata过滤旧2026、改变旧行或作独立上市证明。

这10片须找到可核验真实空响应，或另登记更强的来源证明与缺历史状态。原HONA/2025-08已有成功零页原始证据但未提交result，若承接须显式登记，不能把它追认为原第306完成片。无合格存量证据时，按原边界做有限Local→R2→OpenD补缺；成功空仍标unknown，不把未知缺口变成负例、完整行情或历史资格证明。

当前prepare还只接受原R03与recovery_v1版本，直接传另一库或伪装version/tier不会通过。最小新来源适配与交接见 [REGISTRATION_DRAFT.md](REGISTRATION_DRAFT.md)；尚未授权或执行该草案。
