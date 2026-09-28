# 趋势差异独立核验补充：benchmark QQQ 混入候选 peers

2026-09-27。补充 [原失败审查](GROUP_PARITY_REVIEW.md) 中“趋势两槽待查”，不修改原诊断或代码。已亲读旧 builder 实际调用、新 forward 实际过滤与冻结 universe，并独立重算；不是转述 Sol 的结论。

## 确证的调用边界

- [v6_data.py](../../v6_data.py) `build` 只从 `role == candidate` 提取 `symbols`，加载 `symbols + ['QQQ']` 的 views，但调用 `_group_map(config, dates, symbols)` 时仍仅传 candidate。加载 benchmark 行情不等于把它纳入候选 peer。
- 冻结 `focus_v8_20260927/inputs/market_data/universe/qqq_retrospective_v1.json` 有 **108 candidates、7 benchmarks**；QQQ 的 role 明确为 `benchmark`，不在 108 候选集合中。
- 失配版 `forward/features.py` 先令依赖 `symbols = (members ∩ universe_symbols) ∪ {'QQQ', symbol}`，随后用 `member['symbol'] in symbols` 构造 peers。这里把行情依赖集合当成候选群成员边界，实际允许 QQQ 进入同类 peer。
- 冻结 QQQ 的 trend_15/trend_63 `history_end` 均早于六个代表日。故此差异不是这些趋势行缺时间字段，也不能靠删除 v9 因果检查解释或解决。

## 独立重算结果

未执行 Sol 的诊断脚本：本次用一次只读 Python 进程读取冻结 metadata、calendar 和 107 个实际源的 09:30–11:30 prefix，建立同一份 views；分别调用真实 v6/v9 `_group_state`。以旧 `build` 的 108 candidate 参数调用真实 `_group_map`，并按亲读的失配版 forward 过滤另建 dependency map；冻结 `group_seq` 仅作比较基准。

- **旧函数 + candidate-only map + 同 views：六周完整 group_seq 最大差全部为 0。**
- **v9 函数 + candidate-only map + 同 views：六周所有非 liquidity 槽最大差全部为 0。** peer `history_end` 因果检查保留。
- dependency map 相较 candidate-only map，相关趋势群没有丢失旧 peer；新增项仅为 QQQ。把该 map 送入 v9 函数，逐周精确复现原趋势差异：

| 周 / 代表日 | trend_15 新增 QQQ / 最大差 | trend_63 新增 QQQ / 最大差 | QQQ 趋势 history_end |
|---|---:|---:|---|
| W18 / 05-01 | 是 / 0.012570500374 | 是 / 0.012780606747 | 04-24 |
| W19 / 05-08 | 是 / 0.036708474159 | 否 / 0 | 05-01 |
| W20 / 05-15 | 是 / 0.008511900902 | 是 / 0.007389187813 | 05-08 |
| W21 / 05-22 | 否 / 0 | 是 / 0.012096762657 | 05-15 |
| W22 / 05-29 | 是 / 0.062193393707 | 是 / 0.023631334305 | 05-22 |
| W23 / 06-01 | 是 / 0.009758055210 | 是 / 0.010752677917 | 05-29 |

两策略各 5 周多 QQQ。固定同 views 与其余 metadata，只改变 map 的候选边界即可移除全部趋势差异；**因此该 AMD/2026-06-01 六周样本的趋势根因现已确证为 QQQ benchmark 混入 candidate peers**。此前若给旧 `_group_map` 传入 candidate+QQQ，比较的是被改变的参数，不能代表原 builder 的实际调用。

## 独立问题与未证明边界

- v9 candidate-only map 下 liquidity 仍最大差 `0.5116680860519409`：原 568 peer×周 / 95 证券缺 `history_end` 的不兼容仍成立。QQQ 自己的 liquidity facts 也缺字段，v9 已将其保守排除；修 candidate 边界不能消除旧 liquidity 缺证据。
- 允许 Sol 修正候选边界并实现显式 metadata/schema 拒绝及模型语义握手；本核验不要求其等待批准。QQQ 仍须加载为独立 benchmark 槽，不能为了修 peer 集合而删掉行情依赖。
- 这里关闭的是本样本趋势归因，不是已验收新实现、全日期/全证券 parity、三带群 feature-only G1、G2/G3 或 90% 效果。修复版源码/测试正在并行变化，本次未验收；旧 grouped checkpoint 与严格新 metadata/schema 的兼容门禁仍须独立通过。
- 未训练、采集、改旧数据、启调度或发布信号；直接调用内部函数只用于控制变量诊断，不是绕过 live metadata 门禁的执行方案。

## 证据指纹

| 对象 | SHA-256 |
|---|---|
| 冻结 universe | `cd556bc08e0315d1243baa3817440ed969cdbeb17e8debb493b84867adea5218` |
| v6_data.py / 实际旧 builder 与群函数 | `c50fee6d0a0a4a3b4ff9103d415e4d1022646ba0425f6833b1cf63bb624b60c7` |
| precision_v9/data.py / 因果检查保留 | `06207586233fea7153094dbdb40e314cc7608bf5b7c96b269e21bbc083607a3e` |
| 失配版 forward/features.py（原审查快照，现有并行修复） | `65fedae4ddcc2cda359846e99f019b175c01a73ac8eaa0251d2e4b1208002de0` |
| Sol [GROUP_PEER_DIAGNOSTIC.json](../forward/GROUP_PEER_DIAGNOSTIC.json)，已读并与独立结果核对 | `9ba039c5d5cec0f8b79a432582729b47a07a4363d219df742d28ade747b560f1` |
| Sol [group_peer_diagnostic.py](../forward/group_peer_diagnostic.py)，另附复核入口 | `a9be304547b35a85fcdaf9d50d5b2fb7e0510325bdd924a78d539a12d12b9899` |

版本/成员 SHA 仍见原审查；未将随后正在修改的 features 文件 hash 冒充为失配版执行证据。
