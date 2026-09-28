# 给Sol的前向证据工程交接

2026-09-27；这是接口与验收建议，尚未实现。先读 STATE/BACKLOG/MATRIX、v9 PROTOCOL与工程规范。仅在根代理安排后编码；保持五路线研究决定与R03运行源不变。建议新增 `precision_v9/forward/{spec,features,inference,ledger,labels,evaluate,cli}.py` 及定向测试，避免重构正在使用的data/baseline。

## 1. 冻结输入与三层门槛

- `ForwardSpec`：protocol/version、五route IDs、九target顺序/主目标、price_basis与公司行动策略、ET时区、11:30 cutoff、11:30:30 information_deadline、11:35 entry、1170 RTH cooldown、发布deadline、官方calendar完整sessions、三操作群8/12/4、更新/缺失/拒绝规则。
- 保持当前30秒出结果契约的首版 `publish_deadline=11:30:30 ET`，实际 `published_at` 必须≤deadline且严格早于11:35。若现实需要延后发布，先另登记版本并确保120秒人工延迟与固定11:35入场仍兼容（最晚11:33），不能运行后放宽。
- `RouteArtifact`：route/version、checkpoint(s)、完整feature/tensor顺序及shape、变换和模型构造代码、训练/拟合监督行ID、原始/校准输出、校准参数、单调投影、行动score_column、每群threshold、成熟先验规则/初始库、group输入规则、冻结时间和逐文件sha256。
- 默认行动列沿用当前v9的 `raw_3d_5pct`；若某路线研究改列，须随版本明确冻结，不能拿校准列悄悄替代。阈值可为null，此群永远拒绝，0信号precision=null。
- `CohortSpec`：启用时间早于首session、固定起止和官方日期全集（≥60）、30信号/20日期工程门槛、两个半段、5/10日期块与15格同时区间规则。完整期末一次验收，不因期中达到90%而提前停。
- G1仅验证可预测；G2另验真实输入/发布时间；G3另验cohort全分母与结果。制品缺失或真实receipt缺失可先交fixture工程，但不得填写通过。

## 2. 只看当时的特征/推理接口

```text
load_route(frozen_artifact) -> verified model + inference_schema
build_features_asof(snapshot, decision, schema, metadata, mature_prior_store) -> X + lineage + rejections
predict_route(artifact, X) -> nine raw + processed outputs
issue_once(spec, route, cohort, run_id, predictions, quality, ledger) -> immutable issued/refused events
mature_signals(signal_ids, outcome_snapshot, calendar, evaluation_asof) -> appended outcome events
evaluate_cohort(cohort_id, ledger_cut, preregistration) -> full denominators + gates + report
```

`build_features_asof/predict_route`没有当前/未来标签、未来entry价参数；`data.build/_outcomes/focus_v8.load_data`不是这个接口。先过滤版本再编码，所有输入bar_end≤cutoff且available_at/非空真实received_at≤information_deadline，包括历史相对量分母、同伴与QQQ；迟到修订只能影响之后的决策。

保留训练的x5/x60/xday/group_seq形状、缺失mask、tabular/path/curve列顺序与消融约束，逐route白名单防止B无群/C无群/C无日级通过辅助输入偷带被移除信息。不得为了实时可用暗改窗口或填补规则。

成熟先验按原版本读取所有合格本股历史行，不只读曾buy者；history IDs及每项label_end/label_available均早于decision，九标签完整成熟条件不改变。初始历史快照现在实际读到可作为未来已知输入，但其receipt就是现在，不能回填成历史bar结束时间。

固定操作群与动态模型输入群分开。前向动态群版本需真实created/published/received及effective时间；仅历史重建的feature_cutoff不足以证明当时可用。冻结股票/同伴成员、公司行动和source snapshot的已知版本；未来新split不得倒删旧buy，无法一致计价记unresolved。

## 3. 账本事件与发布

- 每事件含 `event_id/route_id/model_version/cohort_id/run_id/sample_id/symbol/session_date/cutoff/decision_at/recorded_at/event_type/payload_sha/prev_hash`；输入快照、模型、schema、协议、日历、成员、源码、runtime/dependency哈希随run归档。
- `prediction`保存九raw/processed数值、reference时间、source IDs、quality；`buy`另存score、各群threshold、`member_groups/trigger_groups`、实际published_at、planned_entry_at、cooldown_end；预测时间不冒充发布时间。
- 拒绝状态显式为 `no_trigger/cooldown/missing_input/data_unverified/expired/model_unavailable/calendar_unavailable`；每候选都有结果。已运行无buy与整日missing_run分别记，不能合并成0%命中。
- 对每route按symbol与decision排序，计算全部触发群；同股只发一次union buy，每群只记实际跨过其阈值者。不同route独立发信号；另报信号/误报交集。
- 事务内用持久route+symbol状态检查 `decision_at < previous_cooldown_end`；end从前一11:35起累计1170 RTH分钟。提前命中/缺标签不释放cooldown，重启、模型换版、重复run也不重置；旧cohort尾部占用传到后继版本。
- 唯一键至少覆盖route/cohort/cutoff/symbol及事件类型；同一route只启用一个动作版本。并发/重试不得出现两条buy；新run_id不是重新发信号的理由。
- 先可靠落盘再公开结果，数据库事务与不可变artifact核对；实际发布越期即expired，禁止回写published_at。若错误buy已经实际公开，保留issued记录并标protocol_violation，不能撤回分母使cohort过关。
- `live`入口以实际系统时钟取时，不能接可回填的--as-of；历史/合成模式独立路径且无法写live账本。记录时钟健康和实际receipt审计，hash链提供篡改检测，不宣称外部公证时间证明。

## 4. 标签、分母与验收

先从已发buy全集取signal IDs再左连接最新有效outcome；标签事件只追加，修订有supersedes和新sourcehash，不改预测。每目标保存entry/Open及源bar、threshold_price、完整expected grid、first-touch bar区间、label_end/available/evaluated_at和缺失原因。

RTH严格使用官方日历78/234/390根5m窗口；同一目标正负例等完整窗口后成熟，早触及但窗口未齐仍pending。源标签11:40的Open才是11:35 entry；不能偷换11:35结束柱、延后找到第一根有利bar或把夜盘触及算成功。

对每route×trigger_group：`N_issued=TP+FP+U`；U按互斥主原因分pending/missing/halts/invalid/evidence_unknown，所有buy恰归一类。`known_precision=TP/(TP+FP)`仅为已知子集，已知分母为0时null；另报总N和U、全体可能范围 `[TP/N,(TP+U)/N]`，N=0时范围null。U>0、评价期未结束或任何forward证据违规时，cohort不得pass；违规数另列，不重复加入N。

供给=`N_unique_issued/(all_official_sessions/5)`，分母取冻结期全部官方session，包含无信号、缺数据、停机与半日；不可从frame.min/max或成功运行日期推导。违规issued单列且阻止通过，不能靠它们凑机会数。

期末仅对完整主目标cohort判断90%/供给/30信号/20日期，其他八目标分别报告成熟状态，不替代主目标。前后半段、每股/日期、拒绝/漏报覆盖、成本、同股AUC/校准诊断按有完整证据的范围报告；缺失不能填0。5/10-session共同日期块重采样15格，保留跨股/交叉成员相关性，零信号重采样不算100%。

## 5. 必须交付的端到端证据

1. 无未来行情和y的当前fixture可得九输出；加/改未来bar、迟到修订、未来labels、未来成员或split后，旧X/概率/action字节不变；群同伴与历史量分母同样覆盖。
2. 五路线真实checkpoint静态重放配对；固定输入同raw/processed/threshold。保存容差与最大差值；C的normalizer/support buffers、树prior残差及全部列顺序都在制品里，不能靠当前源码默认值猜。
3. 跨run/重启/两个并发调用同股只一buy；1170边界、提前命中不解锁、跨周末/假日/半日/DST/跨年、换模型承接cooldown；低于某群threshold的交叉成员不进入该群。
4. 五个已发buy构造TP=2/FP=1/pending=1/missing=1：N=5、known_precision=2/3、U=2、区间[2/5,4/5]、pass=false；以后补标签N仍5。提前正例、报告损坏、缺entry、停牌/split不能被inner join删掉。
5. 60官方sessions含无信号/停机/半日均在分母；只有1条全胜buy不通过30/20门槛；期中90%不可通过；历史回填/live --as-of、收到晚于deadline、发布晚于deadline均不能产生合格live buy。
6. 注入落盘失败/进程崩溃/模型或schema/hash不符，保留失败且不重复发出；更改报告/source/model字节后评估失败，旧预测不被重写。每次归档登记、before/after sourcehash、依赖锁/源码字节、测试结果与逐条审计。

先完成上述fixture及静态制品测试，再分开报告G1/G2/G3。没有安排真实接收实验就如实交G2待验证。此交接不授权启动采集/调度/部署/账户/订单/R2上传；不改变模型研究路线或PROTOCOL的最终目标。
