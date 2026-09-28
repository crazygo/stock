# 实验登记：model_registry_v1 五个既有模型重放

状态：FROZEN（2026-09-28，运行前写入。运行后只追加 `runs/model_registry_v1/ACCURACY.json`，不改本文件的规则）

## 身份与假设

- experiment_id：`model_registry_v1_incumbents`
- 本轮不做新的 `market_dual_track_v2` 八格。五个模型沿用 focus_v8 已冻结配方，只换输出目录。
- 假设：从各自 `spec.json` 启动后，reserved 折的 eval score 与 `runs/focus_v8_20260927/final/<route>/selected_3566/result.json` 的差不超过 1e-4。
- 只改变调用位置和输出目录。不改 `focus_v8` 的配方、种子、标签、切分。
- 种子 3566。数据是已经暴露的 2026 年 focus_v8 数据集。

## 数据与时点

- 数据：`research/after_open_3d5pct/runs/focus_v8_20260927/dataset/`。训练时记录 manifest 的 sha256。
- 股票池：该数据集里的 108 只候选。验收另按 spec 中的三群切片报告，不把三群当成预测范围。
- 决策、入场、1170 分钟边界沿用 focus_v8：11:30 截止，11:35 五分钟 Open 代理，最高价触及。
- 折：`protocol.json` 的 reserved。fit 在 2026-07-13 之前且标签已成熟；eval 为 2026-08-24 至 2026-09-18（左闭右开）。这段已经暴露，不是独立前向。

## 冻结评估计划

- 主报告数字：eval 上主目标 `3d_5pct` 的分类准确度，阈值 0.5，即 `(p >= 0.5)` 与标签一致的比例。
- 同时报告：基准率、多数类准确度、p≥0.5 的精准率、Brier、eval score，以及与原 checkpoint 的 score 差。
- 阈值 0.5 在运行前固定，不用 eval 再选。
- 五个模型全部跑完才算这一轮完成。某一个失败则保留失败记录，不改其他模型的输出。
- 准确度不是 90% 买入达标，也不写成未来保证。

## 运行后附录

见 `runs/model_registry_v1/ACCURACY.json`。
