# 10 · 训练、预测、策略回放的版本化产物契约

本文件记录旧 `manual_three_cutoffs_v1` 导出契约，以下“本次不重新训练”仅指该旧回放交付。后续 `hourly_once_v3` 已按 [13–16](13_hourly_once_implementation.md) 另建训练、校准、快照、推理与前向纸面制品；旧运行及本文件的旧字段不被回填改写。
新一次运行的 `report.json.artifact_registry` 采用 `research_artifacts_hourly_v3`，仍关联 `ModelSpec → TrainingRun → PredictionSet → PolicyRun → Report`，并附模型、配置、训练 manifest、预测、快照和阈值文件哈希。前向追加日志另外保存最终 report SHA-256，以免报告自引用。

## 五种关联实体

导出的 `overview.json.artifact_registry` 使用 `research_artifacts_v1`。所有关联 ID 明确，不靠文件名猜测。`EXPORT_MANIFEST.json` 给出 `overview.json` 与所有模型流水、路径文件的 SHA-256；原运行 `manifest.json` 和 `predictions.parquet` 的哈希也在 overview 中。导出只读原始运行；目标输出目录必须不存在。

| 实体 | 当前字段与关联 | 扩展规则 |
|---|---|---|
| `ModelSpec` | `id`、H/A/B 输入组、模型结构、关键输入、设计假设、label/feature schema、`spec_version/spec_sha256` | 新模型或新输入新增 ID/schema，不改旧解释 |
| `TrainingRun` | `source_run_id`、fold、model_spec_id、配置哈希、状态、耗时、模型文件及 SHA | 后续增加每轮训练/内层验证指标、累计耗时、失败原因；旧空缺保持 `not_recorded` |
| `PredictionSet` | 固定源 `predictions.parquet` SHA、model_spec_id、外层 fold 范围和行数 | 未来区分训练内、时间 OOF、外层、前向纸面；默认只评分外层 |
| `PolicyRun` | prediction_set_id、`manual_three_cutoffs_v1`、policy 配置哈希、决策/操作流水文件及 SHA | 改时点、阈值或退出规则生成新 policy run；不覆盖旧预测或旧回放 |
| `Report` | 关联 overview、导出完整性清单和上述实体 | 新报告保存新路径和哈希；不改既有运行 |

当前 `ModelSpec` 是描述训练方案的元数据；它没有声称特征因果重要性。旧运行的 LightGBM 每折只记录 `best_iteration`、最终内层 Brier、特征名和总秒数；TCN 记录每个 epoch 的 train loss/inner Brier、最佳 epoch 和总秒数。没有的曲线/每轮耗时不得反推或插值。新代码已为**未来** LightGBM 训练加内层验证曲线和累计迭代时间，为未来 TCN 加每轮与累计耗时；本次不重新训练，旧运行仍为空。

## 时间、输入与评估门禁

导出器核对配置版本、`manifest.scope`、源文件 SHA、日历 SHA 和每个读取的 5m 源文件 SHA。`features/outcomes` 按唯一 `sample_id` 关联，预测中的目标值必须与冻结 outcome 一致。每条预测的 `session_date` 必须在其 fold 的外层区间内；未成熟 outcome 拒绝导出。静态看板不从模型制品重新推理，也不把数据集中的训练行评分。

现有原运行只落盘了**已成熟、输入/标签合格**的样本。源级缺失和未成熟原因是聚合审计，不能重建未入库行的股票、日期、模型分数。后续数据集应输出逐条 `eligibility/exclusion` 表，包含 `symbol, session_date, cutoff, reason, source_snapshot_id`，看板才能把未成熟/缺失按任意切片重分配。

## 操作状态机

`policy_replay_v1.py` 只接收预测、日历和 5m 正常时段 OHLC。它不 import 训练器或券商模块。每个 symbol 按 UTC `decision_at` 排序维护仓位：

```text
flat ── 过阈值 ──> recommend_entry ── 有入场代理 ──> held
  └─ 低于阈值 ──> below_threshold（仅观察）
held ── 任意新分数 ──> observe_held（不重复买入）
held ── 确定目标触及/完整窗口结束 ──> flat
held ── 缺失路径/未成熟 ──> unresolved（不假定已退出）
```

12:30 记作 `scored_only`，不进操作计数。恰好等于阈值视为过阈值。已持仓时分数高低只更新观察；不把“没有买入”解释为模型负例。若退出与新决策在同一 5m bar，只有 bar **结束**不晚于决策时才视为空仓，避免利用 bar 内未知先后。缺失后持仓状态保守保留，直到另一个带证据的独立清算协议处理。

交易状态：`target_touch_proxy`、`window_end_proxy`、`entry_bar_missing_or_invalid`、`entry_wait_expired`、`indeterminate_missing_path`、`pending_window`、`pending_calendar`、`decision_time_mismatch`。前两者属于已决操作分母；其他均不填作亏损或成功。后续若添加真实逐笔、止损、仓位或资金约束，应升 policy 版本并重放。当前每个建议按一股单位比例模拟，**不构成组合收益曲线**。

目标触及用 5m high 判定，退出代理记恰好 `entry_open × 1.05`；未触及在第 234 根常规 5m bar 的 Close 退出。触及时间只报正常交易分钟区间 `[bar 起点, bar 终点]`，不伪造精确秒数。high 触及时同根 low 是否发生在卖出前未知，故 `mae_lower_bound/mae_upper_bound` 形成风险边界，并标 `mae_order_uncertain`。净变化按入场/退出两侧滑点和费用算：

`net = exit_proxy × (1 − sell_slip − sell_fee) / [entry_open × (1 + buy_slip + buy_fee)] − 1`。

默认每侧滑点 5 bp、费用 1 bp，仅为演示假设。Open、high、Close 都不保证实际可成交，缺盘口和排队信息；手续费也不代表用户券商实际费率。`target_hit` 与 `net_return` 是不同指标。

## 文件接口与扩展

命令 `python -m research.after_open_3d5pct.export_dashboard_v1 --source <run> --output <new-dir> --threshold 0.35` 产生；可另传 `--fee-bps-per-side`、`--slippage-bps-per-side` 并另存新目录做执行假设敏感性：

- `overview.json`：模型卡、metrics/trials、五类 registry、总计、来源与审计说明；
- `models/<family>_<input>.json`：该预测集完整外层逐条预测、三时点 action、独立 trades；
- `paths/<symbol>.json`：从冻结源行情取得的正常时段 OHLC，按股票按需加载；
- `EXPORT_MANIFEST.json`：导出文件和源文件哈希。

后续新模型只需生成相同主键与时间字段的 `PredictionSet`，添加相应 `ModelSpec/TrainingRun`，即可复用 policy 和页面。新 policy 与新报告写新目录；前向预测集须区分 `unmatured` 与 `mature`，并保留真实当时可得时间、模型启用时间及冻结阈值来源。当前导出格式仍是研究用首版，生产服务/API 契约和实时日志待单独设计。
