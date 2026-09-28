# 群同伴边界与时间证据门禁实施结果

2026-09-27。实施范围对应先前 [登记](GROUP_GATE_REGISTRATION.md)。仅改 `forward/features.py`、`forward/inference.py` 和必要测试/诊断文档。`data._group_state` 的 `history_end` 因果排除、旧元数据、旧 checkpoint、标签、训练器、采集和账本均未修改。

## 已确认与实现

冻结 v6 builder 的群 map 只接收 108 个 candidate，QQQ 单独作为 benchmark view。修前 forward 错把 QQQ 当 candidate peer。六周同一份 views 的 candidate-only 旧函数与冻结 `group_seq` 差均 0；candidate+QQQ 旧函数与修前 forward map 旧函数差均 0。独立核验见 [QQQ 更正](../forward_readiness/GROUP_PARITY_QQQ_CORRECTION.md)，逐周成员、发行人计数、QQQ prefix 与 source hashes 见固定 [诊断 JSON](GROUP_PEER_DIAGNOSTIC.json)。当前 features 的群 map 明确只接收 `universe_symbols - {'QQQ'}`；QQQ bars 仍供独立第 6 槽使用。合成测试即使错误把 QQQ 写进 universe，也不能把它变成 peer。趋势两槽的本样本根因已关闭。

群 feature-only 新 schema 与 metadata 都须声明 `group_peer_semantics=v9_causal_history_end_v1`。对六个代表周**实际生效且已分类**的自身与同群 candidate peer，逐行要求 `facts.history_end` 为严格日期且早于该代表日，并核对 group/strategy/version；缺失、畸形、当日或未来返回 `X=None`、`group_metadata_incompatible`，lineage 带 symbol、strategy、week、代表日、version、字段/原因、派生内容 SHA 与 issue_count。合法 `insufficient_history` 空群仍 mask；无群路线无需群 metadata。成功 lineage 新增 `used_market_symbols`，供本地 adapter 严格核对真实行情依赖。内容 SHA 是调用时元数据对象的规范化诊断指纹，不冒充源文件字节哈希。

旧 v8 群 schema 明确 `legacy_static_only`，只允许 historical fixture 静态 replay；输出九预测仍可复放，但 quality `g1_ready=false`、`g2_eligible=false`。其 schema 不能进入 feature-only 构造；旧 metadata 即使被错误宣称新语义，也因原 liquidity 568 个 peer×周缺 `history_end` 而在推理前拒绝。`predict_route` 要求 schema/feature lineage 语义一致；新因果群 manifest 还须有模型、schema、训练 dataset 和 membership hash 的 `group_training_provenance`，且旧 `focus_v8_*` model version 不可冒称新群模型。此门禁是契约隔离，不是旧模型可迁移性证明；未来若完成独立等价证书须另行登记和验收。

## 验证与证据层级

- 50 项 forward 联合 unittest 全过（features、inference、group parity/gate、local adapter、prior store、ledger、labels、evaluate），约 10.55 秒；其中本任务定向 16 项全过。合成测试覆盖 classified own/peer 缺失、NaN/非法日期/数字、当日/未来、非法 group ID、完整合法历史、空群、QQQ benchmark 与无群不受影响；真实 AMD fixture 验证旧群在模型调用前审计拒绝。
- 旧静态五路线 checkpoint 重放的九目标 raw/p ≤`1e-7` 测试仍通过；但三条旧带群 feature-only 的**首轮 FAIL 未转为 PASS**。首 FAIL 的 group_seq `0.511668086`、B_group raw/p `0.011500286/0.012499189` 等数值固定于 [GROUP_PARITY_FIRST_FAIL.json](GROUP_PARITY_FIRST_FAIL.json)，它是从 `evidence_baseline/` 保存的修前 features/inference/test 原始字节，在修后环境重放的预期失败，stdout/stderr 另存。原始报告中的首次 21.35 秒记录与重放约 21.19 秒有自然耗时差。新 `test_group_parity` 的 PASS 只代表**正确拒绝**不兼容元数据。
- 修前 [GROUP_PEER_DIAGNOSTIC.json](GROUP_PEER_DIAGNOSTIC.json) SHA-256 `9ba039c5d5cec0f8b79a432582729b47a07a4363d219df742d28ade747b560f1` 未覆盖。当前诊断脚本加载 `evidence_baseline/features.py`，重跑到 `/tmp/group_peer_diagnostic_rerun.json` 与修前 JSON **字节相同**（SHA 亦相同）；它不是用修后 feature gate 假造旧成功。

复验命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 research/after_open_3d5pct/.venv/bin/python -W ignore -m research.after_open_3d5pct.precision_v9.forward.group_peer_diagnostic > /tmp/group_peer_diagnostic_rerun.json
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 research/after_open_3d5pct/.venv/bin/python -W ignore -m unittest -q research.after_open_3d5pct.precision_v9.forward.test_features research.after_open_3d5pct.precision_v9.forward.test_inference research.after_open_3d5pct.precision_v9.forward.test_group_parity research.after_open_3d5pct.precision_v9.forward.test_group_gate research.after_open_3d5pct.precision_v9.forward.test_local_adapter research.after_open_3d5pct.precision_v9.forward.test_prior_store research.after_open_3d5pct.precision_v9.forward.test_ledger research.after_open_3d5pct.precision_v9.forward.test_labels research.after_open_3d5pct.precision_v9.forward.test_evaluate
```

实施快照：features SHA-256 `810c7fd9f7c7d996534e02ed7c42d549a1f18c5754a14e8a78c4f599b418a5c1`；inference `e2a0667c5d44cc62c5760a10fa6d5571f7c236e78488d4b2f5d5cf349bb9ded3`；修前 features/inference 分别 `65fedae4ddcc2cda359846e99f019b175c01a73ac8eaa0251d2e4b1208002de0`、`e8ef043748b7542bc6cb38a0f891dc96d7456792d20f68856daee55bfe79eae8`。`forward_readiness/GROUP_PARITY_REVIEW.md` 采用决定字节 SHA `6b201e4c5c33a183f09e7f49f3ea47c203e545ba63099c0bd3d937665aab03f6`，未被本任务反写。

## 留待新证据

本次只关闭 AMD 六周的趋势归因与缺字段静默放行问题。旧 liquidity 成员缺时间证据不等于已证实前视；全样本等价、三条带群新模型 feature-only 九目标 G1、真实接收 G2、独立效果 G3 均未通过。任何旧模型复用须另以原群生成过程和 2026 全样本等价证书验证，不能凭本局部修复去掉因果门禁或宣称可交易。
