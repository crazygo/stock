# C 无日级 · 已证实状态

2026-09-27。本次只做旧证据诊断及研究交接，未编码、未拟合。状态为 **P0等待共享数据验收；P1为条件方案**。共同目标、供给及前向验收引用 [PROTOCOL](../../PROTOCOL.md)，不另造90%定义。

## 路线身份与现存能力

- incumbent 为 v8 R10：`balance + regularization + calibration + support`；`representation/prior/curves/capacity` 均未启用。真源：[final_selection](../../../runs/focus_v8_20260927/final_selection.json)。
- `FocusC` 以 `daily=False, group=True` 构造：本股7个完整日加当日5m前缀、30日加当日60m前缀；没有本股 `xday` 分支。**仍有长期群摘要，不是完全没有长期信息。**
- 群输入 `group_seq=(6周,6槽,10通道)` 包含趋势15/63/126、波动、流动性、QQQ。前五周取各周最后session，当前周取当前session；全部是对应日11:30状态。按类型投影、群内汇总后用GRU编码，并以群relation调节本股5m表示。
- 本股短序列已有相对收益/价位/成交及mask。`relative_coordinates` 是已有可选变换，v8 R2在本路线试过后回退；不能称尚未尝试。`group_relative_features` 是本股减当前群收益的既有函数，但 `fit_c` 不接收该表，它只接入 B 有群的 `curves` 配方。
- v7 C 有群已试“当前群状态减历史周均值”，迁移失败；并非本路线直接否定证据，也不是全新能力。现有GRU已能表达周变化；尚缺的是同一日10:30与11:30的同伴配对变化。

源码证据：[core](../../../focus_v8/core.py)、[support](../../../focus_v8/support.py)、[run](../../../focus_v8/run.py)、[CModel](../../../train_multiscale_v6.py)、[群构造](../../../v6_data.py)、[群周差分](../../../iterations_v7/c_group/v7_c_group.py)。

## 真实结果与决定

v8十轮全部未准出。R00只复用概率；R01只加5m/60m跨patch MixingPatch。R01开发六个群×折单元仍0/6通过，**回退到上述incumbent**。

| 指标：两开发折×三群均值 | incumbent | R01候选 |
|---|---:|---:|
| Brier | 0.240142289 | 0.238628195 |
| 同股AUC | 0.493525485 | 0.473235895 |
| 90%且供给通过单元 | 0/6 | 0/6 |

微小Brier下降不支持短周期择日改善。决定真源：[decision_before_diagnostic](../../../runs/precision_v9_20260927/R01/C_no_daily/decision_before_diagnostic.json)。

| 已暴露最后诊断段 | 群 | cal阈值 | TP/n | precision | 信号/5官方sessions |
|---|---|---:|---:|---:|---:|
| R00原配方 | chips | null | 0/0 | null | 0 |
| R00原配方 | optics | .55 | 4/8 | 50.0% | 2.22 |
| R00原配方 | storage | .35 | 10/17 | 58.8% | 4.72 |
| R01回退候选 | chips | null | 0/0 | null | 0 |
| R01回退候选 | optics | null | 0/0 | null | 0 |
| R01回退候选 | storage | .40 | 13/20 | 65.0% | 5.56 |

两开发折的R00和R01三群均因cal无合格阈值而拒绝，不是未展示其失败。R00光通信cal为8/8却向后降至4/8；小cal不能承诺90%。诊断段18个官方session，全部2026已暴露。
真源：[R00 metrics](../../../runs/precision_v9_20260927/R00/metrics.json)、[R01 precision](../../../runs/precision_v9_20260927/R01/C_no_daily/reserved/precision.json)。

## 数据支持：事实与尚未证明之处

| 折 | fit行/日期 | cal日期 | fit学习到的群槽位 |
|---|---:|---:|---|
| dev1 | 3064 / 29 | 9 | 15日、波动、流动性、QQQ |
| dev2 | 6090 / 58 | 9 | 上述加63日 |
| diagnostic | 8997 / 86 | 10 | 上述加63日；126日仍屏蔽 |

所有旧fit的本股126日完整率为0，但本路线不读取xday，不能把它直接当作本路线失败根因。相关问题是群类型支持迁移、有限训练日期和短cal；支持恢复是否改善仍待P0。
证据：[support_diagnosis](../../../runs/focus_v8_20260927/support_diagnosis.json)、[v8 R10 dev1](../../../runs/focus_v8_20260927/round_10/C_no_daily/dev1/result.json)、[dev2](../../../runs/focus_v8_20260927/round_10/C_no_daily/dev2/result.json)、[final](../../../runs/focus_v8_20260927/final/C_no_daily/selected_3566/result.json)。

读取R03进度时为 `running, 145/545` 月分片；是本次时点快照，继续以 [progress](../../../runs/precision_v9_20260927/R03/progress.json) 为真源。21操作股加QQQ先齐不等于全109证券同伴/训练池齐。共享builder修复和测试也不能代替完整数据验收或模型结果。

下一步入口：[BACKLOG](BACKLOG.md)、[MATRIX](MATRIX.md)、[HANDOFF](HANDOFF.md)。当前池回溯、历史available_at假设、High触及而非成交边界继续保留。
