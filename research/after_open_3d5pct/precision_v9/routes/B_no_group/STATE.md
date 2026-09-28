# B_no_group 状态与复盘

2026-09-27；本次仅有界只读诊断与文档交接，未编码、训练或获取数据。目标及时间口径继承 [v9 协议](../../PROTOCOL.md)：11:30 ET cutoff → 11:35 Open 代理 → 1170 RTH 分钟 +5% High 触及。90% 是去重买入信号 TP/(TP+FP)，三群各自同时满足、每群每 5 官方 session 平均 ≥1 信号；High 触及不是成交。总目标仍未完成。

## 已证事实

- Incumbent：v8 R9，`representation + regularization + calibration`，669 个本股特征（585 摘要 + 84 路径摘要），九目标各一个 LightGBM；未保留 prior、capacity、curves、balance、recency、specialize、bagging。原配方是最大 120 树、7 叶、深 4、叶子最少 150 行、L2=30、学习率 .035、种子 3566，tune 早停 20。
- B 无群不输入其他股票、群或 QQQ 行情。旧实现仍按操作群等权计算 tune 早停与 cal 校准权重，须披露；这是训练/选择权重，不是逐行模型特征。原配方匹配轮保持它不变，不能顺便改权重。
- R01 固定 0.25 同股 pairwise 辅助已回滚：两开发折×三群平均 within-stock AUC `0.6281465 → 0.6305480`，Brier `0.2189728 → 0.2239218`，通过单元仍 `0/6`。AUC 增量 .0024015 未达到登记 .02；不能解释为所有排序损失都无效。

| 制品/时期 | chips：TP/n，供给 | optics：TP/n，供给 | storage：TP/n，供给 |
|---|---|---|---|
| R00 dev1（19 session） | 0/0，0 | 24/40=60.0%，10.53 | 0/0，0 |
| R00 dev2（20 session） | 0/0，0 | 0/0，0 | 0/0，0 |
| R00 diagnostic（18 session） | 0/0，0 | 0/0，0 | 6/9=66.7%，2.50 |
| R01 dev1 / dev2 | 均 0/0，0 | 均 0/0，0 | 均 0/0，0 |
| R01 diagnostic（18 session） | 0/0，0 | 18/34=52.9%，9.44 | 10/17=58.8%，4.72 |

供给单位为每 5 官方 session 去重信号数。零信号 precision=null。阈值只由各自 cal 冻结；不能按上表 diagnostic 回选。R00 dev1 optics 阈值 .70、diagnostic storage .40；R01 diagnostic optics .50、storage .45，其余群拒绝。

| 旧 incumbent 折 | fit 日期/行数 | cal 日期 | fit 完整 126 日率 | eval 完整 126 日率 |
|---|---:|---:|---:|---:|
| dev1 | 29 / 3064 | 9 | 0% | 0% |
| dev2 | 58 / 6090 | 9 | 0% | 98.05% |
| diagnostic | 86 / 8997 | 10 | 0% | 98.04% |

## 下一假设及边界

训练/评价的历史支持突变、短 cal 与行情漂移可能共同限制迁移；这里只证明存在差异，未证明它导致误报。R03 扩历史后先冻结旧配方重建匹配基线；随后才检验本股波动条件残差。新数据、架构、阈值搜索不能同轮改变。

R03 批量采集由共享任务负责，本次未验收其完成；最终训练 manifest/覆盖/split 尚待注册。2026 全部 exposed，新补历史也不自动成为 untouched holdout。最终通过仍需冻结后的连续 ≥60 官方 session、每路线每群 ≥30 成熟去重信号及 ≥20 信号日期等 v9 门槛。

## 证据索引与身份

- `runs/focus_v8_20260927/round_10/summary.json`、`round_09/B_no_group/{dev1,dev2}/result.json`、`final/B_no_group/selected_3566/result.json`。
- `runs/precision_v9_20260927/R00/{metrics,selection}.json`；`R01/B_no_group/decision_before_diagnostic.json` 及三折 `precision.json/result.json`。
- 路径均相对 `research/after_open_3d5pct/`。旧 dataset 身份：features `6bce8471011f1c2c54ce2551ba7584f18196fb5b056580e528d95532d5c0e744`；rows `b6220c3645ff6477ff59763d75deaff26b6f60c7b1ad34f6d0b2b913a4fed2fe`。
- 当前 `precision_v9/train.py` SHA256 `65ab3544d0eca645beb9d95f04e936c0a4f03d1e04c03fd7f1fc7d5a4bfad7a0`；v9 协议 `21dde0982b2f24edb006fc32f43d32b5e73f099e12b71e0408a2f07bea331a7a`。后续登记必须重新读取，勿把本次哈希当未来代码身份。
