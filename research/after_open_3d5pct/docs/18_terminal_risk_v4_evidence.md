# 18 · 三日触及与期末收益风险：真实开发期结果和运行手册

本轮工程协议为 `three_day_touch_terminal_valuation_v4`，训练前定义见 [17](17_terminal_risk_v4_registration.md)。**效果结论：两个已暴露 2026 外层折的内层候选选择均为 `no_candidate_passed`，没有可发布的正式手动行动。** 不能因为看到期末收益、风险模型或概率数字就称预测达到“高置信”或能盈利。本次不改 dashboard，不管理实际账户/持仓/成交，不自动下单。

## 评价是什么

目标事件沿用原标签：延迟代理 5m Open 入场后的 234 根常规 5m bar（1170 分钟），任一 High≥入场价×1.05。必须完整窗口成熟，且 234 根、原价 `NONE`、时间顺序和可用时刻均核准才评分。中途深跌后仍触及算成功，不设止损、不扣 MAE；+20% 不额外奖励，触及后的价格不进入评价。触及的研究毛评价收益固定 +5%；未触及用第 234 根 Close 计算期末**可变现估值**，不是用户已授权强制卖出。统一假设买卖每侧 6bp：`net=(1+gross)×0.9994/1.0006−1`。因此 +5% 触及不等于净赚 5%，也不是实际成交保证。

两个预测部分分开：HB/HAB × LightGBM/TCN 触及头训练二分类概率，以内层 Brier 早停、独立内层 sigmoid 校准，外层分别报 Brier/logloss；共享的 HAB tabular 失败条件风险头预测未触及的期末毛收益均值、q10/q50/q90 和失败条件净亏概率。失败条件外层用均值 MSE/RMSE、分位数 pinball 与经验覆盖、亏损 Brier 校验。`HB touch + HAB risk` 的整条管线是 HAB 信息范围，不能算纯 HB 消融。混合预期净评价收益为 `p_touch×touch_net + (1−p_touch)×failure_mean_net`；期末亏损概率为 `(1−p_touch)×P(net_failure<0|failure)`。这三个条件分位数不足以推算每股混合分布 ES95，故逐股 `predicted_es95_loss` 明确不可用。报告中的 ES95 用**全部实际评价机会**的净收益最差 5% 做分数边界平均，而非只在未触及者求尾损。

## 冻结实验和实测证据

训练输入是 v3 冻结的 101,062 个候选×日期×六小时成熟样本及同源未复权 5m。新重算器对 101 只个股的源行情，逐条验证完整 234 根、entry、target、label_available_at；核对了旧运行引用的 105 份 5m 源哈希、日历与股票池元数据哈希。`_r3` 是**首次真正拟合模型**的 v4 运行；此前两个目录均在模型拟合前因 Brier 早停代码勘误和 Pandas 微秒/纳秒对齐异常而终止，失败证据保留。训练在 50.6 秒完成 8/8 触及 trial，并分别拟合两折的共享失败风险头，28 个模型/校准制品哈希已验。注册配置、代码、旧训练实际 `config.json`、依赖与源哈希在拟合前的 `provenance_at_launch.json` 固定。

| 时间外开发折 | 成熟外层样本 / 日期 | 全池触及 | 全池净评价均值 | 全池实际 ES95 loss | 失败风险头 RMSE / q10经验以下占比 / q10–q90覆盖 | 内层保留候选 |
|---|---:|---:|---:|---:|---:|---|
| August 2026 | 12,636 / 21 | 2,564 / 12,636 = 20.29% | −0.1534% | 10.69% | 3.28% / 5.71% / 84.84% | 无 |
| September 2026 | 7,878 / 13 | 1,811 / 7,878 = 22.99% | −0.3373% | 10.50% | 3.48% / 4.32% / 85.08% | 无 |

内层选择已把新收益风险评价实际纳入：每个 ET 日期×小时横截面，先筛预测净评价收益>0、失败条件 q10 净收益≥−15%，再按触及概率最多取该横截面 5% Top-K；预算未填满即候选失败。入围后按**内层成熟真实结果**验证净评价均值>0、全体实际 ES95≤15%、至少 100 个选择和 10 个日期，再在同覆盖预算比较触及率。风险限额只是预登记开发场景，未经用户批准。August 四候选的内层入围净评价均值均为负，且全体 ES95 过高；September 多数候选无法填满每小时预算，故两折都未通过。小样本的高触及率不能替代覆盖、真实收益与尾损约束。

外层只检验冻结规则，不能回头挑赢家。August 外层四方案的逐时规则入围 508–666 条，触及率 44.7–51.1%，但净评价均值仍为 −0.44% 至 −0.13%，实际全体 ES95 loss 为 14.9–17.6%。September 外层逐时规则入围仅 0、2、0、10 条；这不足以声称 50%/60% 小样本触及率有用。失败条件 10% 分位经验覆盖仅 4.3–5.7%，与 10% 名义值偏离；概率校准在若干外层候选甚至比 raw Brier 更差。逐股 q10 和预期收益须作为研究估计，不可解释为经独立验证的精确分布或可执行风控。

同日期/小时并以训练期拟合的波动和流动性分箱抽 100 次匹配对照，August 严格同箱匹配有正触及率差，但净评价均值与尾损未过开发场景；未来研究需解释高波动/流动性的共同暴露。按 5/10 个连续交易日做成块 bootstrap；入围日期不够两个完整块时区间标 `insufficient_independent_calendar_blocks`。同日六小时和三日重叠窗口有相关性，不能将 12,636/7,878 当独立交易次数。2026 历史是已暴露外层、当前股票池回溯、历史 `available_at=end+1s` 假设、无夜盘观测且无真实成交，因此仅为开发期负结果，不是独立高置信验证。

## 可复现命令

从仓库根目录，以本工程 `.venv` 运行。每次重训、评估或预测给**新**输出路径，不能覆盖这里的运行证据。训练完全离线，只读既有数据；一次运行 `--acquire auto` 才按 Local→R2→OpenD 只读刷新到独立 snapshot，绝不上传 R2。

```bash
research/after_open_3d5pct/.venv/bin/python -m unittest discover -s research/after_open_3d5pct/tests -v

research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.train_terminal_risk_v4 \
  --config research/after_open_3d5pct/configs/terminal_risk_v4.json \
  --output research/after_open_3d5pct/runs/<new-v4-run>

research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.audit_terminal_risk_v4 \
  --run research/after_open_3d5pct/runs/<new-v4-run>

research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.evaluate_terminal_risk_v4 \
  --run research/after_open_3d5pct/runs/<new-v4-run> \
  --output research/after_open_3d5pct/runs/<new-v4-run>/evaluation.json

research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.run_once_terminal_risk_v4 \
  --model-run research/after_open_3d5pct/runs/terminal_risk_v4_20260925_r3 \
  --output research/after_open_3d5pct/runs/<new-once-run> --acquire auto
```

`--as-of` 接收带时区 ISO 时间；未来时间直接拒绝。历史时间可做回溯模型应用，但早于模型产出时标为 `retrospective_model_application`，绝不冒充当时已经可用的预测。默认省略 `--as-of` 时读取**实际系统时钟**并只向过去取当日最近已完成且模型支持的 10:30–15:30 ET 小时。首个时点前、休市、半日收市、缺市场/行业/个股 bar 或晚到数据均给不可用/过期的具体状态。`requested_at`、`effective_cutoff_at`、`resolved_at`、实际 `knowledge_at`、`generated_at`、各数据水位/接收时间、模型/快照/可见视图/特征哈希和每股分数写入一次运行的 `report.json` 与 `per_symbol.csv`。原始 snapshot 完整保留；推理自动只选截止与知识时刻均可见的版本，已观测接收版本优先于无接收时间的历史假设版本，绝不要求调用者删除未来数据。`--snapshot-from <先前一次运行目录>` 会逐文件校验既有原始快照哈希，再为本次请求派生新的可见视图；不会把复用时刻冒称新采集时刻。没有未来三日数据照样可推理，标签保持 pending。即使算出研究分数，也没有获批风险预算或独立前向达标，所以没有正式行动。

实际证据路径：[`terminal_risk_v4_20260925_r3`](../runs/terminal_risk_v4_20260925_r3) 下的 `manifest.json`、`provenance_at_launch.json`、`terminal_outcomes.parquet`、`trials.jsonl`、`selection.json`、`predictions.parquet`、`evaluation_daily.json` 和 `integrity_audit.json`。`evaluation_daily.json` 是在源制品哈希校验后只追加的最终评价，包含按 ET 日期和小时的全池/筛选分子分母，零入选日期也保留；此前 `evaluation.json`、`evaluation_verified.json` 原样留存。训练 manifest SHA256 为 `6ad731f365cc681065a8ce0f5b0340262bffe2a6cf17184591e9fea4a21251d5`，最终评价 SHA256 为 `e59d8b0060a3a3bbd9421268fa5f3275becde13e113c3ff44694c124ff480259`，完整性审计 SHA256 为 `b629bd06b42e0c3e417e4890a62b97c80f03b0c5b58037cb93c0e4bb44bc894c`。

历史接口复核在 [`terminal_risk_v4_historical_probe_20260924_1330_r4`](../runs/terminal_risk_v4_historical_probe_20260924_1330_r4)：2026-09-24 13:31 ET 的 AAPL/NVDA 使用新 v4 触及检查点及失败风险头，各有四个研究分数，但状态均是 `retrospective_model_application`，因为模型直到 2026-09-25 09:20:15 UTC 才可用；它们不是当时的实时建议，推理也没有读取 outcomes。58 项工程/边界测试通过。

最新一次全池只读刷新在 2026-09-25 09:24:17–09:34:04 UTC 执行 Local→R2→OpenD，采集审计为 105 个 Local copied、105 个 R2 downloaded、103 个 OpenD received、2 个 OpenD empty，全部写入独立原始 snapshot，没有修改既有训练源或上传 R2。修复版本选择后，于实际系统时间 **2026-09-25 09:35:49 UTC／05:35:49 ET** 逐文件哈希校验该快照并派生 105 个可见源文件，生成 [`terminal_risk_v4_now_verified_20260925_0936Z`](../runs/terminal_risk_v4_now_verified_20260925_0936Z) 中的 `report.json`、`per_symbol.csv`、`snapshot_manifest.json`、`visible_view_manifest.json`。101 只候选股全部为 `unavailable / before_first_supported_completed_cutoff`，`effective_cutoff_at=null`、`selected_model=null`，没有当前可执行预测或建议。首个 10:30 ET 支持时点尚未发生，不能把前一日或历史回溯分数伪装成现在信号。原始 snapshot manifest SHA256 为 `b2a5a3ba1c89a7b5666ef58019f20457accd70079c600ba60ec87233dcf9892d`；本次报告 SHA256 为 `8d40d621ab331f6992c225f828112f2e0bbb4e6e118750f72b435b5be55bce88`，逐股 CSV SHA256 为 `f91d606836b2c15a1632da80682d744c4df76a404d9e1d7d8ccd4daea22532d8`。这是当前时间的真实不可用结果，而不是失败标签或 0% 胜率。
