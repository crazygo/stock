# 群同伴与时间证据门禁实施登记（先于代码修改）

2026-09-27。采用决策见 `../forward_readiness/GROUP_PARITY_REVIEW.md`，当前字节 SHA-256 `6b201e4c5c33a183f09e7f49f3ea47c203e545ba63099c0bd3d937665aab03f6`。目标仅修正已证实的 QQQ 同伴混入，并在真实依赖的已分类群成员缺有效 `history_end` 时审计拒绝；不改 `data._group_state`、旧 checkpoint、数据、标签、训练器或因果条件。

首轮真实 feature-only 的 1e-7 配对 FAIL 及预测差已固定在 `GROUP_PARITY_REVIEW.md`。修前逐周诊断输出固定为 `GROUP_PEER_DIAGNOSTIC.json`，SHA-256 `9ba039c5d5cec0f8b79a432582729b47a07a4363d219df742d28ade747b560f1`；**复跑应重定向到新路径，不覆盖此文件**。其脚本与源代码原始字节另存 `evidence_baseline/`：features `65fedae4ddcc2cda359846e99f019b175c01a73ac8eaa0251d2e4b1208002de0`，inference `e8ef043748b7542bc6cb38a0f891dc96d7456792d20f68856daee55bfe79eae8`，诊断脚本 `a9be304547b35a85fcdaf9d50d5b2fb7e0510325bdd924a78d539a12d12b9899`，配对测试 `44b12471670901e16d425da2e13a6bd99fe2e1219e0340515b5d86f6c32a8398`。冻结 versions/memberships 分别为 `fcc5c218947c666a850e1984695d1c8259b795a3514a7cbc856429a74549888b`、`1a23e39efd8c0e3b2a6d793e8eb0ebc0e2c7c619a17abb77d7027cabe5c86a7d`。

正确核验入口：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 research/after_open_3d5pct/.venv/bin/python -W ignore -m research.after_open_3d5pct.precision_v9.forward.group_peer_diagnostic > /tmp/group_peer_diagnostic_rerun.json
```

冻结 `v6_data.build()` 将 108 个 candidate 传给 `_group_map`，QQQ 另作 benchmark view；把 QQQ 也传进 map 是此前错误对照。六代表周用同一 forward views，candidate-only 旧函数对冻结 `group_seq` 最大差均 0，candidate+QQQ 旧函数对 forward map 旧函数最大差也均 0。W22 额外 QQQ 的 cutoff_return 为 `0.000650313566`，trend_15/trend_63 分槽残差分别 `0.0621933937/0.0236313343`。另一已证实的旧 liquidity 568 个 peer×周缺 `history_end`，属于元数据字段证据欠缺；不推断它们实际前视。

变更验收：

1. 行情依赖可含 QQQ，但 candidate 同伴 map 只取 universe 候选。保持发行人去重规则，趋势六周旧群函数对冻结值的对照仍为 0。
2. 群 schema 显式标明 `v9_causal_history_end_v1`，来源 metadata 同样标明；旧静态 schema 明确为 `legacy_static_only` 诊断，不因张量维度一致而成为前向群输入。模型 manifest 与 feature lineage 语义须握手；没有匹配新训练来源的旧 checkpoint 不可接新群特征。无群不受影响。
3. 六代表周实际依赖的已分类 own 与同群 candidate peer，`facts.history_end` 必须是严格 YYYY-MM-DD 且早于代表日；缺失、畸形、当日或未来返回 `X=None`、`group_metadata_incompatible` 和 symbol/strategy/week/date/version/字段原因及内容 SHA。合法未分类 `insufficient_history` 继续空群 mask。
4. 原 v8 成员缺字段的真实 fixture 预期审计拒绝；合成完整合法历史、每种坏字段、空群、无群和推理拒绝各有测试。后验测试绿只能证明拒绝与局部映射正确，不宣称三条带群 G1 成功；正向配对仍等 R03 新 metadata/模型。
