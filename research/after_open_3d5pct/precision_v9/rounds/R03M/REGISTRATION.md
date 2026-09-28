# R03M · 冻结 2026 群元数据重建与全键兼容性登记

2026-09-27，状态 `registered_for_engineering_reconstruction`；本文件不登记训练。独立于 R03，不修改其来源、采集/恢复、三臂扩量计划。新输出建议 `runs/precision_v9_20260927/R03M_group_only_v1`，必须原先不存在；本次尚未创建 run。

## 冻结输入与角色

旧根 `research/after_open_3d5pct/runs/focus_v8_20260927`；以下相对该根，读取 `inputs` 里的源，不回退共享 `market_data`、R2/OpenD 或未完成 R03。

| 输入 | SHA-256 |
|---|---|
| config.json | `7da26f04c6c0fc3d296df932a1da153ac4541ceb821f4f1800085f53f4296057` |
| source_provenance.json | `e2f92d6742a58c46e836cb6b30bcccfacbae57696ddfdb211e2b9ff410869d84` |
| dataset/rows.parquet | `b6220c3645ff6477ff59763d75deaff26b6f60c7b1ad34f6d0b2b913a4fed2fe` |
| dataset/features.npz | `6bce8471011f1c2c54ce2551ba7584f18196fb5b056580e528d95532d5c0e744` |
| dataset/manifest.json（含 223 个来源 SHA） | `43bbf25f88c62313b610657b77b1ae99e4b45973fa41a91cef4f8279a067d9e9` |
| inputs/market_data/universe/qqq_retrospective_v1.json | `cd556bc08e0315d1243baa3817440ed969cdbeb17e8debb493b84867adea5218` |
| inputs/market_data/calendars/nasdaq_sessions_2026_v1.json | `1de4f9b4a472b9995fa6a7e8649fa1b39ce9e6b75e07e544f1b099569fd6989e` |
| 冻结群 versions.json | `fcc5c218947c666a850e1984695d1c8259b795a3514a7cbc856429a74549888b` |
| 冻结群 memberships.json | `1a23e39efd8c0e3b2a6d793e8eb0ebc0e2c7c619a17abb77d7027cabe5c86a7d` |

群目录为 `inputs/research/group_expectation_matrix/outputs/20260925_v5`。全池固定 108 candidates；QQQ 额外作为唯一加载的 benchmark，共 109 行情与 109 公司行动源。冻结 universe 其余 benchmark 不自动加入 peers/加载池。GOOG/GOOGL 保持 issuer 去重。操作三群、九目标、主目标与原监督规则不变。

## 冻结重算算法与时间

1. **先原样快照源码与运行配置，再重算**。原 101 candidates = 108 减 v8 `added` 七股；从冻结原始 float64 bars 构造官方 RTH 5m grid，复用原 `group_expectation_matrix.build.prepare_bars/daily_history/classify_history/corporate_dates` 口径。保留 duplicate invalid、NONE、完整 RTH、available>=bar_end 的检查；原研究总 asof 为 `2026-09-24T20:00:01Z`，每周历史仍须严格早于该周 cutoff。
2. 新增 `AAOI/ANET/AXTI/CIEN/COHR/CRDO/FN` 复用冻结 `focus_v8.prepare.new_memberships` 的计算路径：v6 float32 view、daily_full、以 seqday 重建 close、既有 `_split_days` 判断。仅复用计算，不调用会把新结果追加进旧成员文件的写入入口。不得把两条路径偷偷统一为 `v9.make_groups`。
3. 交易日按冻结官方 calendar，周 cutoff 为该 ISO 周首个官方 session 的 open_at；不用 weekday 猜。分类窗口仅用周首日前最后 need 个官方 sessions，逐日完整且实际 `available < cutoff`，公司行动跨越按上述既有路径拒绝；history_end **必须取实际参与成功分类的最后交易日**。
4. trend15/63/126 需 n+1 个 close，以 n 个 log-return 的 sum/(sample std × sqrt(n)) 分类，阈值 ±1；volatility 用 21 close 的20 returns，年化 sqrt(252)，分界 .2/.4；liquidity 用20日 RTH turnover 中位数，分界 50m/200m。保留各原路径零方差与数值精度，禁止 parity 后改阈值/选路径。
5. 成功行保留实际 window_dates、source hashes、history_end、最大 available、classifier value、动作检查与算法版本；不足/缺失/动作冲突保留 `group_ids=[]` 及明确 reason/窗口证据。不能复制旧类别再推算日期。不能重建原始 received_at/first_seen/published_at；新 created_at 只记本次真实重建时间，标 `causal_weekly_reconstruction`、`exposed_development`。
6. 群聚合取 candidate-only map，QQQ 仅独立 benchmark 槽，使用保留 peer history_end 检查的 v9 `_group_state`。代表周取原每 key 当前日及此前各周最后已知官方 session，共最多六周；11:30 prefix、11:30:30 信息截止、group mask/issuer 去重不变。own tensors 不重编码。

## 源码与分类定义指纹

以下为本登记读到的源码，实施新增入口后另冻结完整 imports/实际运行字节；并行 forward 修复必须另做版本快照，不能只记路径。

| 路径（research 下） | SHA-256 |
|---|---|
| group_expectation_matrix/build.py | `cdcc28163beb61ddef68a9eb9e940ff46d5963d0c25d956198cccdb65a7c9d93` |
| group_expectation_matrix/outputs/20260925_v5/config.json | `d976fa5f36eb4acce1482d45e8c29ad255e486bd37aadd8829f034518bf5ff18` |
| 同上 strategies.json | `4336add67c55c500941974ed63809b86e08461f518fb14abec6bd9dd99325700` |
| 同上 manifest.json（原101股202源已全匹配） | `83b78ff7b6c47198cbc27b9e1dc02b3118f0986c992330f03c5e134ab99c50cb` |
| after_open_3d5pct/focus_v8/prepare.py | `ab99c7c8296986f2e190da76f6637a65661c049c10dab9dcc4d1f6ad966a4780` |
| after_open_3d5pct/v6_data.py | `c50fee6d0a0a4a3b4ff9103d415e4d1022646ba0425f6833b1cf63bb624b60c7` |
| after_open_3d5pct/precision_v9/data.py | `06207586233fea7153094dbdb40e314cc7608bf5b7c96b269e21bbc083607a3e` |
| after_open_3d5pct/precision_v9/prepare.py（仅审计/helper参考） | `0dffa738a804bafce87b962352878d7901b5600f88869384840cf7476969cca1` |

## 全键门禁与复盘分支

- 新数据行集合及顺序须是全部 14,424 旧 sample_id，old_only=new_only=duplicate=0；`x5/x60/xday/y` NPY 解压字节、shape、dtype/NaN 掩码逐行完全相同。截止/decision/entry/entry_price、label_end/available、terminal1/3/5、own support 字段全等；仅允许 group_valid_count 随重建 mask 更新并独立报告。group 与 group_seq 必须一致对应，不能只替换其中一个。
- 重新计算自身成熟 prior 并核对全部 history IDs、counts、数值；fit/tune/cal/eval keys、正权重监督 IDs、权重、种子3566、incumbent recipe、旧 split/purge 与支持计算规则全冻结。先验不因本次新建 metadata 引入新历史标签。
- 按每 key/周/策略/通道报告 old/new 分类与群差异、缺口原因及 source lineage。先做全键重建对旧张量；随后以真实 feature-only 路径覆盖最早有效周、分类变化、合法空群/动作边界等代表样本并逐路线比较九目标，拒绝测试不能冒充正向 G1。
- **A：全部 group/group_seq 及模型有效输入全等** → 不训练。保存版本化 metadata 与明确 `exact_tensor_equivalence` 兼容证书：旧模型 SHA/真实版本、原训练 dataset 和原 schema SHA、新重建 membership/dataset 与目标 schema SHA、全键输入 parity 报告 SHA、校准 SHA、作用路线/源/日期/keys 范围及复放误差。旧权重/校准/阈值不变；每路线全旧有效键 raw/p 与原 checkpoint 复放误差≤1e-7。当前 loader 的 causal 路径仅支持 `group_training_provenance`，并拒绝 `focus_v8_` model_version；**仅在证据通过后另登记最小 certificate 路径扩展**，默认旧模型拒绝仍保持。不能改版本字符串绕过、伪造重训 provenance 或冒称旧权重由新 metadata 训练；本证书限定历史兼容性，G2=false。
- **B：仅群输入不同，且逐项因果可解释** → 先把实际影响交 B_group/C_group/C_no_daily 各自 Astra 复盘，另登记匹配 refit；本登记不能自动训练或调整分类器。后续使用同旧 keys/recipe/seed/折，C support 仅按同一 fit 规则重算；两无群保留旧权重/输入并复放为不变对照。禁止借用 R03 的 `new_expanded_fit`。
- **C：非群输入变化、源不匹配、时序证据不足或无法解释差异** → 不发布新兼容声明、不训练；记录失败与具体所缺日期/字段，修实现或等待 R03；不得缩池、挑匹配 keys、复制旧类别以强行通过。

既有 `baseline.py` 目前仅认 R03/同快照工程 purpose；若进入 B，需单独 R03M purpose/薄适配复用原 fit 实现，不能以错误登记名绕过门禁。本轮不调用训练器、不修改原制品、不接网络/交易/调度。原本股缺126日历史、当前池回溯、历史 available 假设、公司行动覆盖未认证等限制全部保留；G2/G3 不通过，不声称 precision 提升。
