# 冻结 v8 群路线真实同伴配对（G1 诊断）

2026-09-27。新增 `test_group_parity.py`，只读冻结 `focus_v8_20260927/inputs/market_data`、群版本/成员、旧样本和三个 dev1 checkpoint。测试取 AMD 2026-06-01 的六个周代表日 W18–W23，按群成员关系加载 106 只候选同伴及 QQQ，共 107 个实际行情来源；同伴仅保留从 2026-04-02 起足以计算 20 日基准的 09:30–11:30 前缀，AMD 保留完整历史可见源。`build_features_asof` 分别重建 B_group、C_group、C_no_daily，输入不含当前/未来 `y` 或入场价。成熟先验来自冻结 **全部候选行** 中 58 条在本决策前九目标均已成熟的 AMD 自身历史记录。冻结 5m/60m/日线及群张量只用于比较基准，不作为 forward 特征输入。

真实 feature-only → 九目标预测配对 **FAIL**，原 `1e-7` 数值门禁保留。一次最终运行耗时 21.35 秒，三路线特征构建各约 6.8–6.9 秒，模型加载与推理各约 0.02 秒。最大绝对差：

| 路线 | group_seq | xbase | relative | raw 九目标 | processed 九目标 |
|---|---:|---:|---:|---:|---:|
| B_group | 0.51166809 | 0.50910628 | 0.01135526 | 0.01150029 | 0.01249919 |
| C_group | 0.51166809 | 0.50910628 | 0 | 0.00023618 | 0.00027002 |
| C_no_daily | 0.51166809 | 0.50910628 | 0 | 0.00054628 | 0.00047913 |

三路线的 `x5/x60/prior/paths/curve` 最大差均为 0；含日线的两路线 `xday` 差也为 0。六周 `group_seq` 变动槽位按 `[trend_15, trend_63, trend_126, volatility, liquidity, QQQ]` 为 `[25,25,0,0,30,0]` 个单元，槽位最大差分别为 `[0.06219339,0.02363133,0,0,0.51166809,0]`。旧五路线用 **冻结 batch 张量**重放 checkpoint 的静态误差≤`1e-7`；它不能代替本次从真实同伴行情独立构造群特征的失败证据。

已确认一处契约差异：冻结 v8 的 `v6_data._group_state` 对同群 peer 不要求 `facts.history_end`；forward 使用的 v9 函数要求它早于代表日，缺失时保守排除。六周 AMD 的 liquidity 同群共有 568 个 peer×周记录（95 只不同证券）缺该字段。例如 AAPL 在 W23（代表日 2026-06-01）的 `liquidity:high` 记录仅含 `median_daily_turnover=12469789656.941502`；对应版本 `feature_cutoff_at=2026-06-01T13:30:00Z`，`source_observed_at=null`。旧 `group_expectation_matrix/build.py::classify_history` 源码用先前交易日并要求数据 `available < cutoff`，但 liquidity 分支未把 `history_end` 写入成员事实。这说明旧成员行欠缺可逐行验的时间证据，**不是已证实发生前视**；为追求旧模型配对而放宽前向因果门禁不成立。

趋势槽差异仍待定位，不能归因于 liquidity。已逐股对照 107 个输入源在六个代表日的旧 `v6_data.load_symbol` 与 forward `_select/_view` 的 `cutoff_return`、`cutoff_rvol`，均一致；冻结 manifest 登记的 223 个 source SHA 与当前冻结 inputs 全部匹配。trend peer 成员集合与原 `_group_map` 的初步集合核对相同；仍需定位实际构造群图时的差异。现有结果足以阻止将旧带群 checkpoint 宣称为真实 feature-only 前向就绪，待补可验证的群元数据/新 route 契约或在 R03 数据上重新训练，再单独验收。

复现命令：`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 research/after_open_3d5pct/.venv/bin/python -m unittest -v research.after_open_3d5pct.precision_v9.forward.test_group_parity`。当前预期为 1 项失败并打印 `GROUP_PARITY_METRICS`；无联网、真实账本写入、行情下载或训练。`historical_fixture` 的 `g2_eligible` 与 `publication_eligible` 均为 false；这是 G1 诊断，不是独立前向效果证据。

稳定源码 SHA-256：`forward/features.py` `65fedae4ddcc2cda359846e99f019b175c01a73ac8eaa0251d2e4b1208002de0`；`forward/inference.py` `e8ef043748b7542bc6cb38a0f891dc96d7456792d20f68856daee55bfe79eae8`；`precision_v9/data.py` `06207586233fea7153094dbdb40e314cc7608bf5b7c96b269e21bbc083607a3e`；冻结 `v6_data.py` `c50fee6d0a0a4a3b4ff9103d415e4d1022646ba0425f6833b1cf63bb624b60c7`（与冻结 dataset manifest 中的 builder SHA 相同）。冻结群 `versions.json` `fcc5c218947c666a850e1984695d1c8259b795a3514a7cbc856429a74549888b`、`memberships.json` `1a23e39efd8c0e3b2a6d793e8eb0ebc0e2c7c619a17abb77d7027cabe5c86a7d`。

## 后续精确归因与证据更正

上文“趋势仍待定位”是首轮报告时的状态，现已关闭。冻结 `v6_data.build()` 把 **108 个 candidate** 传给 `_group_map`，另读 QQQ 作 benchmark view；原 forward 把行情依赖集合中的 QQQ 也放进群 peers。先前一次诊断误把 109 个 candidate+QQQ 也传给旧 `_group_map`，因此未看出集合差；该比较参数与冻结 builder 不同，已撤回。逐周以相同 views、正确 108 candidate 重新计算，旧函数对冻结群张量六周最大差均 0；错误的 candidate+QQQ map 对原 forward map 的旧函数输出也六周全 0。W22 唯一额外成员 QQQ 的 prefix return 为 `0.000650313566`，分别造成 trend_15 的 `0.0621933937` 与 trend_63 的 `0.0236313343` 最大差；五个趋势有差的周均能在 JSON 中逐周看到 QQQ 唯一新增，发行人去重数同步变化。QQQ 的趋势 `history_end` 合法；错误是其 **benchmark 身份进入候选同伴**，不是因果日期筛选误杀。liquidity 的 568 个缺 `history_end` 仍独立存在，不能以修 QQQ 解锁旧带群 checkpoint。

修前独立诊断输出 [GROUP_PEER_DIAGNOSTIC.json](GROUP_PEER_DIAGNOSTIC.json) SHA-256 `9ba039c5d5cec0f8b79a432582729b47a07a4363d219df742d28ade747b560f1`，脚本原字节及四个关键源文件见 `evidence_baseline/`。首失败数值另由这些保存的原始 feature/inference/test 字节重放到 [GROUP_PARITY_FIRST_FAIL.json](GROUP_PARITY_FIRST_FAIL.json)（SHA-256 `f32e20f1ae67355fe85e9a1166a1d89bd8ebe8c5350c9adadc03eeaa04ab6ad1`），其重放耗时约 21.19 秒；此文件是**修后执行的修前字节重放**，非冒充最初那次 stdout。上文原始 21.35 秒记录仍保留。重放 stdout/stderr 亦单独保存。后续新 `test_group_parity.py` 改测明确拒绝旧元数据，其通过不是三条群路线九预测配对通过。
