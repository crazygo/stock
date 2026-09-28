# 实验登记：五路线 × 三组

状态：FROZEN（2026-09-29，运行前写入）

15 个模型是 `B_group`、`B_no_group`、`C_group`、`C_no_group`、`C_no_daily` 各复制一份，只把预测和拟合行限制到一个组。目标、入场、配方、种子 3566、reserved 折都与对应父模型相同。父模型的 checkpoint 和 `runs/model_registry_v1/<父模型>/` 不覆盖。

组的成员来自 `focus_v8/protocol.json`：

- 半导体组 `chips`：ALAB、AMD、ARM、AVGO、INTC、MRVL、NVDA、QCOM
- 光通信组 `optics`：LITE、COHR、CIEN、AAOI、FN、AXTI、CRDO、MRVL、AVGO、ALAB、ANET、CSCO
- 存储组 `storage`：MU、SNDK、STX、WDC

交叉成员同时出现在半导体组和光通信组里，两个模型都会给它出概率。准确度仍是 eval 上主目标 `3d_5pct`、阈值 0.5。
