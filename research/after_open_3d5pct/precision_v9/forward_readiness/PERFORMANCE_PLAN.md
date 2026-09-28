# Forward 性能测量登记：先测全流程，再决定优化

2026-09-27；只规划，不写代码、不跑测量。**执行前置：R03M 有效 metadata 与三群模型/兼容证书正向门禁通过**；当前不满足。使用本地 fixture、不联网、不启调度、不连接交易；测得30秒也不升 G2。

## 实际状态

- 旧 AMD 单 candidate、107依赖、58成熟prior 的三群诊断约22秒，最终报告21.35秒；这是三条群路线一次离线 fixture，不能当作单模型耗时或全21候选估算值。当前门禁修后没有全池计时。
- `local_adapter.run_local_e2e` 每次处理一条路线；循环 forecast rows 时逐candidate调用 `build_features_asof`。features 每次循环依赖symbol，重新 `_select/_view`；每路线还会重新加载 bundle/model、执行前后全引用 hash/seal 检查。重复成本存在于源码，实际占比尚未测，不能预定它就是主瓶颈。
- adapter 的 `total_before_report_write` 不含最终summary写入，亦不是五路线整体时间；21/16/58项测试耗时不是运行SLA。缺模型/元数据导致快速拒绝不能算五路线性能成功。
- 操作群union固定21股：AAOI、ALAB、AMD、ANET、ARM、AVGO、AXTI、CIEN、COHR、CRDO、CSCO、FN、INTC、LITE、MRVL、MU、NVDA、QCOM、SNDK、STX、WDC。全池仍108 candidates，QQQ独立benchmark；所有需要的peer从该完整池解析，不按21股缩群。

## Backlog与方案矩阵

| 优先级 | 待解决问题 | 实际验收 |
|---|---|---|
| P0 | 冻结合法可推理的同一11:30 snapshot、21候选×5独立路线、全池依赖/模型/日历/prior | 105条candidate-route audit齐全；每条需要的全部源在场，不能因提前拒绝少执行工作 |
| P0 | 测当前实现完整冷/暖路径及分阶段成本 | 首轮固定一冷一暖，不用最快结果代替整轮；保存原始trace、阶段计数、总时长与失败 |
| P1 | 将超时归因到实际资源/函数/重复工作 | 记录读取bytes、hash bytes、_select/_view次数、每路线candidate数、峰值RSS；由证据排序 |
| P2 条件后续 | 仅在失败后登记一项缓存或向量化改动 | 同输入/同输出/同拒绝/同哈希门禁，前后配对，再测总耗时；本文件不授权提前优化 |

| 方案 | 能回答什么 | 代价 / 决定 |
|---|---|---|
| 当前串行五路线全流程测量 | 当前真实总成本与瓶颈；成功/失败均有增量证据 | **采用**；不先改执行次序或实现 |
| 先缓存历史views/共享解析 | 可能消除重复，但可能误用未来版本、漏掉重新验证 | 仅结果表明此项占主导后另登记 |
| 先向量化/并发五路线 | 可能改善CPU，但改变峰值内存、线程争用和账本时序 | 本轮不做；先看阶段/资源trace |
| 只测一股、一条路线或少peer，再线性外推 | 不能证明105条操作与完整依赖截止成本 | 不作为通过依据 |
| 降候选池、删hash/seal/因果门禁、用未来数据预热 | 改变任务或使证据失效 | 禁止 |

## 冻结测量对象

1. R03M完成后先写独立测量registration/输入清单/source_snapshot，再启动计时。登记冻结模型/schema/校准/阈值、metadata及完整候选集合、官方calendar、bars/actions、PriorStore cut与PriorView来源、adapter/features/inference代码SHA、环境/线程/硬件。现源码指纹仅作计划参考：adapter `0987c1d1425addf81fffcc18cb35d725b46352f142e44a55c2669e41eccdbf98`，features `528344d95bab167fb520584e854c0223e2ccba2b592b3fcaa5a2a9818229e646`，inference `e2a0667c5d44cc62c5760a10fa6d5571f7c236e78488d4b2f5d5cf349bb9ded3`。
2. 决策日期在见性能结果前确定：默认取R03M中21股均有旧合法key的最后官方session；若不存在则记录数据前置失败，不换成更小池。五路线使用同一截止、全21 forecast、同一PriorStore cut、相同原始快照，按固定顺序 `B_group/B_no_group/C_group/C_no_group/C_no_daily`；不可投票合并模型。
3. 先物化完整108候选+QQQ来源清单和每candidate/route的实际依赖矩阵。benchmark不进入peer，未分类合法mask不等于缺源；全部已声明依赖都须可查，重复成员/漏成员仍拒绝。行情只含截止可见版本，未来bar/当前未来标签不进入预测snapshot。
4. prior维持全候选九标签完整成熟、最后63行规则；登记实际候选/历史行数、每股先验counts与视图mode，不用空prior或删peer人为缩小负载。任何fixture注入时钟/接收时间只能明确标合成，不变成历史receipt；真实historical import仍按真实时间隔离。

## 时钟、阶段与冷暖边界

- **T0**：本次最后一个所需原始bar/metadata/outcome版本实际在本地可用的单调时钟屏障。若此时尚需生成/核验manifest、捕获seal PriorView、构建特征或加载模型，其耗时都算截止后；不能把T0推迟到这些衍生产物准备完。同时保存UTC与fixture逻辑cutoff，明确历史回放不证明真实11:30:30接收。
- **T1**：第五条路线结束，105条audit已完成、应记ledger事件已提交、前后hash/seal复核通过、运行summary/audit已fsync；外层测 `T_total=T1-T0`。测量报告自身的额外落盘单列，不能拿内部timer或各模型纯推理时间替代T_total。
- 分段至少包含：截止后输入/视图封存、文件hash、读取/解码、版本筛选、own/peer views、group/tabular构建、各路线model load、inference、preledger hash/seal、ledger提交、postledger hash/seal、结果落盘。记录每段calls/bytes与峰值RSS，保持嵌套计时不重复相加；五路线分项与外层总时钟都保存。
- **冷运行**：新进程、不预先持有模型/解码数据；包括进程/导入准备发生在T0后的成本。不得清全机OS缓存或影响共享采集；若OS page cache未知，报告“process-cold / OS cache uncontrolled”，不能伪称磁盘完全冷。
- **暖运行**：只允许在T0前准备固定模型/代码和当时已可知的历史源（默认截至前一官方session close）；列出预备耗时、bytes/hash和各缓存实际可复用状态。当前实现若仍重读/重载/_view就照实计时，不用monkeypatch跳过；不预读本次尚未到达的11:30输入，不以新实现的缓存假设报告当前暖耗时。
- 首轮固定一冷一暖，各为新fixture ledger/run_id、相同输入/状态，顺序先冷后暖；重试只因测量自身失败并保留原结果。报告两者原值，不用两次样本计算p95；不重复大量试跑挑最快。外部占锁任务不暂停，记录当时系统负载作为限制。
- 单次观测上限20分钟，先保留全部阶段trace；若超过30秒仍继续至完整收尾以定位，若触及20分钟则记录timeout/未完成candidate数，不能按已完成子集宣称通过。不得由性能限时删掉应有候选审计。

## 成败与后续边界

全21×5正常预测路径、合法依赖、前后验证与账本收尾齐全，冷/暖各自报告是否≤30秒；任何缺模型、输入拒绝导致未跑满或未最终持久化，均不是完整性能通过。fixture ledger预期的data_unverified拒绝需明确：它测到的是当前计算/审计/拒绝写入路径，尚不证明真实BUY提交与外部展示的截止SLA；不能强改G2来制造BUY。

若当前实现失败，先写实际STATE/BACKLOG/MATRIX，按占比只选一项优化另登记；要求相同105条预测、九目标数值容差、拒绝原因、依赖/版本/先验lineage及分母不变，再跑同一冷暖方案。若通过，只接受已测fixture规模/环境的工程延迟，不承诺未来负载、不升G2/G3、不声称precision改善。正式日历cohort coverage与11:35前真实发布仍独立验收。
