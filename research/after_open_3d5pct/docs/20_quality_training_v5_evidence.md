# 20 · 因果质量股票池 v5：运行、评价与复核

本页记录 [19 · 拟合前登记](19_quality_training_v5_registration.md) 所冻结的 `causal_quality_training_v5` 实验。**工程和有限实验已完成；概率改善目标、行动证据目标均未通过。** 本实验读取本地冻结数据，没有联网更新行情，没有修改 v4 模型或看板，也没有下单。2026 年两个外层月份都已暴露，只能称开发期时间外诊断，不能称独立验证或当前可执行建议。

## 固定问题与真实运行

每个 `symbol × decision_at` 仅用此前 63 个官方交易 session 中、六个不同小时标签均完整成熟且已经可用的股票日。至少 40 个完整日，日等权历史触及率严格大于 60% 才进质量组；数学上恰好 60% 不进组。目标仍为延迟入场价起未来 1170 个正常交易分钟触及 +5%。没有中途止损或浮亏扣分；触及固定 +5% 代理，否则三日期末 Close 估值；两侧各 6bp 成本。期末估值不是用户授权的到期强制卖出，也不是实盘成交。

使用相同前向日期折完成 LightGBM 和小型 TCN：每折 10 个首轮新候选（HB/HAB × 两家族的无质量列 pooled、质量感知 pooled，外加 HAB 质量专用），两折共 20 个；冻结 v4 的四个同样本参考只评分。按登记的**内层相同质量行**早停和训练/早停差距，August LightGBM、September LightGBM 和 TCN 各触发一个正则化第二轮候选，August TCN 未触发。总计 **23/23 个新 trial 完成，0 失败，训练约 62.57 秒**。第二轮触发、每轮 Brier/耗时、校准器、选择结果在 `trials.jsonl`、`selection.json` 和 `manifest.json`。质量专用模型只用每条训练行当时符合资格的样本；TCN 的九个质量静态列、缺失掩码和日数归一化均进入检查点 schema。HB 触及头仍共用 v4 的 HAB 风险头，因此完整收益管线的信息范围是 HAB。

最终冻结运行为 [`runs/causal_quality_v5_20260925_r2/`](../runs/causal_quality_v5_20260925_r2/)。此前 `r1` 已完成 23/23，但其训练器存在“假如未来某 trial 在评分中失败，部分预测行可能先落盘”的原子追加缺口；`r1` 实际没有失败 trial。修复只将完整评分后的行写入结果，没有改配置、输入、候选或选择门槛，因而以全新目录 `r2` 复跑；**保留 r1 及修复原因，不把技术复跑计作额外超参搜索**。`r2` 的 48 个模型/校准制品、四个结果制品及全部 101,062 行质量特征均已由独立完整性脚本按哈希、源和检查点 schema 复核。

## 质量池本身与冻结内层选择

下表的“全池同期”是按质量组所在日期×小时加权的全池描述性对照；也另列不加权的外层全池。不能把历史筛选差异解释成模型因果增益。`ES95` 对**全部机会**收益的最差 5% 计算，跨尾部边界使用分数权重；不只对未触及者计算。同期全池 ES95 使用与触及率、净值相同的日期×小时横截面权重。

| 已暴露外层 | 质量组触及 | 同期全池触及 | 质量组净评价 / ES95 损失 | 同期全池净评价 / ES95 损失 | 完整全池净评价 / ES95 损失 | 质量组证据 |
|---|---:|---:|---:|---:|---:|---|
| August 2026 | 431/900 = 47.89% | 20.91% | −0.679% / 16.35% | −0.078% / 10.86% | −0.153% / 10.69% | 150 股票日、21 ET 日、13 股票 |
| September 2026（截至 18 日的外层） | 86/174 = 49.43% | 22.13% | −0.635% / 15.34% | −0.457% / 10.83% | −0.337% / 10.50% | 29 股票日、13 ET 日、5 股票；最大单股占 44.83% |

质量历史率本身能筛出高于同期全池的触及人群，但**没有把平均净评价变成正数，尾损也更大**。质量池含六小时同日重叠窗口，900/174 小时行不能当成同数独立交易；September 的日期和股票覆盖尤其薄弱。此前独立可行性盘点含 September 21 日、185 行，不是这里外层截至 18 日的 174 行；二者不得混用。冻结源共有 101,062 小时行、16,859 股票日，其中 16,775 个有六个完整成熟时点；84 个不完整股票日集中在 9 月 21 日，均未被填成负例。质量身份行 5,825、历史不足行 26,058、已知但历史率 ≤60% 行 69,179；原始源无重复 sample 或股票日时点。

**选择只用内层 selection**。在“概率研究模型”候选里，August Brier 最低的是 `tcn_HAB_quality_aware`，September 是 `tcn_HB_quality_aware`。这两个名称表示内层相对候选的最低 Brier，**两者均未通过预登记的“比同段校准历史率低至少 0.005 且日期块区间下界大于零”开发目标**。两折均没有同时通过概率和行动两道门的候选，`best_inner_action_candidate=null`、`formal_action_enabled=false`。内层没有借外层表现重新选模型；稀疏或零推荐不会迫使填满每小时 5% 的上限。

## 内层预选模型在外层的实际结果

| 外层 | 内层预选模型质量组 Brier | 同池历史率 Brier（原始 / 内层校准） | 相对同家族同输入无质量列 control Brier 改善 | 该模型按冻结场景筛出 | 实际触及 / 平均净评价 / ES95 | 同日期小时同数量历史率排序触及 |
|---|---:|---:|---:|---:|---:|---:|
| August | 0.26681 | 0.27722 / **0.25216** | +0.01270 | 123 | 57/123 = 46.34% / −1.671% / 21.36% | 54/123 = 43.90% |
| September | 0.26711 | 0.26911 / **0.25096** | **−0.02530** | 0 | 无推荐；胜率、收益、ES 不计算 | 无推荐 |

两折预选模型的外层 Brier 分别比主基线**差 0.01465 和 0.01615**。August 质量特征相对同族 control 的 Brier 有 0.01270 的描述性改善，但仍不及简单校准历史率；September 同族增量反向。August 123 条仅有 57 条触及，净评价为负，ES95 超过 15% 的预登记开发场景；同覆盖历史率触及增量仅 +2.44 个百分点，其 5 日块区间下界为负。September 预选模型零推荐，记录为 `no_recommendation`，**不记成 0% 胜率**。概率研究目标与行动证据目标分别未达；不能因某个**非内层预选**候选在已暴露外层有漂亮的小样本结果而替换选择或宣称高置信。

模型相对仅历史率的公平比较采用**同一质量池**、同一内层质量校准段的历史率 sigmoid 基线。校准原始历史率的 Brier 在 August 由 0.27722 改至 0.25216，September 由 0.26911 改至 0.25096；因此超过原始率不能直接归因为行情或质量特征。内层另有值得登记的下一项诊断：对 LightGBM，强制 sigmoid 在选择期反而损坏概率，August HAB control 的原始/校准 Brier 为 0.24601/0.28318，September 为 0.25718/0.26960；TCN 不一定同向。后续应**另开版本**只在内层比较 identity/raw 与 sigmoid 的校准策略，再用冻结选择评分外层；本 `r2` 不更改，也不把 raw 冒称已校准概率，更不降低 60% 质量阈值或发布风险/置信门槛。

## 可复算文件与命令

从仓库根目录运行，使用尚不存在的新运行目录；训练只读本地冻结源，不隐式联网。需项目 Python 3.12 虚拟环境及 `requirements-hourly-v3.txt` 已安装。实际 `r2` 的全量制品、模型和源读取哈希在其 `manifest.json`、`provenance_at_launch.json`；`predictions.parquet` 保留全池/质量组的逐股逐日预测和评分，`evaluation_v2.json` 包含每个候选按小时、ET 日期、股票、历史率档、概率校准箱拆分，并单列内层预选候选的外层结果。最初的 `evaluation.json` 也保留；`evaluation_v2.json` 仅补齐同期全池加权 ES95，不修改训练或原始评分。

```bash
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.train_quality_v5 \
  --config research/after_open_3d5pct/configs/quality_training_v5.json \
  --output research/after_open_3d5pct/runs/<new-v5-run>
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.evaluate_quality_v5 \
  --run research/after_open_3d5pct/runs/<new-v5-run> \
  --output research/after_open_3d5pct/runs/<new-v5-run>/evaluation_v2.json
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.audit_quality_v5 \
  --run research/after_open_3d5pct/runs/<new-v5-run> \
  --output research/after_open_3d5pct/runs/<new-v5-run>/integrity_audit.json
research/after_open_3d5pct/.venv/bin/python -m unittest discover \
  -s research/after_open_3d5pct/tests -v
```

实际 `r2` 哈希：注册配置 `beeba64527284c715ef9149c6ea1ca1ab3022d1d206735304d3e615d18655f6d`；拟合前登记 `6639d64697ed3879805ab046d0582a683d54b68e69626b34b8327ea925416563`；启动 provenance `8f02599112b1224db52d2d796dbd7812e12bc8b196130e13f8b79a8ac04d3943`；运行 manifest `ecc5ae11479c610bdb3b9b52b5df3af321d26b53a858c680f39a430d258f2985`；外层详细评价 `597b7ae2176441203f8d467f8c62f251052d693a20918d5bf98ce348e583280b`；完整性审计 `2b7eabd5746d9a5ef70cabec4a6e61c61c9ab75e9fd561f9349edefbd5472f5d`。审计结果为 `integrity_passed_not_effectiveness_release`：逐行重算 101,062 个质量特征，核验 23 个完成的 checkpoint schema、48 个模型/校准文件哈希、四个训练结果文件哈希。仓库完整测试运行 **73/73 通过（本次验收时仓库全套）**；这属于合成/契约和冻结历史回放的工程证据，不会把外层改名独立验证。

## 尚未满足的效果门禁

当前池是**现有股票名单回溯**，不是历史 PIT 成分；历史可用时间以 `bar_end+1s` 假设为主，未实测真实逐年接收延迟；夜盘覆盖不足。这些假设使外层指标不能作为实盘概率。两个外层日期均已暴露，重叠三日期间的六小时样本和低股票数放大不确定性。失败条件期末分布沿用 v4 的 HAB 风险头，本轮只对触及头加入质量特征；预测净值与 ES 的开发场景不是用户批准风险预算。正式高置信/行动发布仍需先冻结风险容忍度和最低成功率、覆盖及有效独立样本要求，再在新的前向标签成熟后验证；当前保持**研究分数、无正式行动**。
