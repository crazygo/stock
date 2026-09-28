# C_no_group：已证实状态

更新：2026-09-27。状态：未达标；本次只读诊断与方案交接，没有编码或训练。共同定义引用 [v9 协议](../../PROTOCOL.md)，本路线不另改标签、群成员或阈值规则。

## 当前配方与输入边界

- incumbent：v8 R8，`prior=true, specialize=true, calibration=true`；机器真源为 [final_selection.json](../../../runs/focus_v8_20260927/final_selection.json)。R01 的 `within_stock_rank` 已回退。
- 每行仅使用本股 7 个过去日及当日 5m、30 个过去日及当日 60m、126 个过去日日线和本股成熟九任务先验。无其他股票、QQQ、群行情或群身份输入；共享跨股训练参数不等于跨股行情特征。
- `mature_prior` 是最近最多63条本股完整成熟决策，Beta(1,1)，严格检查 decision / label_end / label_available；不是63个自然日，也不是实时当日达成率。
- `specialize` 只保留三操作群并集21股的 fit 权重；tune 提前停止及 cal 拟合仍按操作群等权，交叉成员分摊。此群元数据用途沿用旧配方并披露，不能称完全不使用操作群信息。
- 11:30 ET cutoff、11:35 Open 代理、1170 RTH 分钟 High 触及；九输出保留。90% 是 TP/n，不是 p≥0.9，也不是实际成交收益。

## 已运行证据

R01 相对 incumbent：两开发折×三群通过数仍为0/6；平均 Brier `0.234988275→0.234958957`，同股 AUC `0.453956112→0.468684666`，提升0.014729，未到0.02保留线。决定文件：[decision_before_diagnostic.json](../../../runs/precision_v9_20260927/R01/C_no_group/decision_before_diagnostic.json)。

下面为 cal 所选 raw 阈值的后续成熟信号；每格写 TP/n、precision、每5官方session供给。零信号 precision=null。

| 版本/评价段 | chips | optics | storage |
|---|---|---|---|
| R00 dev1，19 session | 0/0，null，0 | 0/0，null，0 | 0/0，null，0 |
| R00 dev2，20 session | 0/0，null，0 | 2/4，50.0%，1.00 | 0/0，null，0 |
| R00 diagnostic，18 session | 0/0，null，0 | 19/33，57.6%，9.17 | 9/14，64.3%，3.89 |
| R01 dev1，19 session | 0/0，null，0 | 16/22，72.7%，5.79 | 0/0，null，0 |
| R01 dev2，20 session | 0/0，null，0 | 2/4，50.0%，1.00 | 0/0，null，0 |
| R01 diagnostic，18 session | 0/0，null，0 | 19/33，57.6%，9.17 | 9/13，69.2%，3.61 |

来源：[R00 metrics.csv](../../../runs/precision_v9_20260927/R00/metrics.csv)、[R01路线制品](../../../runs/precision_v9_20260927/R01/C_no_group/)。R00 processed≥0.9 三段三群全部零信号；raw≥0.9仅dev1 optics有1/2、供给0.53。加阈值不能修复当前迁移。

## 数据支持与真实结构

| 折 | 全池fit行 / 本路线正权重fit行 | fit日数 | cal日数 | fit完整126日比例 | eval完整126日比例（全池） |
|---|---:|---:|---:|---:|---:|
| dev1 | 3064 / 609 | 29 | 9 | 0% | 0% |
| dev2 | 6090 / 1218 | 58 | 9 | 0% | 98.05% |
| diagnostic | 8997 / 1806 | 86 | 10 | 0% | 98.04% |

本次按冻结 [rows.parquet](../../../runs/focus_v8_20260927/dataset/rows.parquet)、[splits/weights](../../../focus_v8/core.py)及 [training_weights](../../../focus_v8/run.py)复算；不能把全池行数当有效训练样本数。dev1最早59个、dev2最早30个日线位置在fit从未有效，见 [support_diagnosis.json](../../../runs/focus_v8_20260927/support_diagnosis.json)。这是输入分布差异证据，尚非失败的唯一因果解释。

- [PatchBranch/CModel](../../../train_multiscale_v6.py)：5m patch12、60m patch3、日线patch6，宽16；已有 patch 内卷积、位置嵌入、加权池化、融合网络，incumbent共11,961参数。没有跨patch卷积，不能笼统称“没有时序能力”。
- [FocusC.capacity](../../../focus_v8/core.py) 已实现三尺度 `MixingPatch`；v8 C无群未实际选择该候选，R10选择的是 support 且回退。R01 C无群只试pairwise；C无日级的两尺度 MixingPatch 则已失败，作为邻近风险证据，不能转记为本路线实验。
- incumbent dev1/dev2/diagnostic最佳epoch为13/1/1，单折记录约4.94/3.79/4.53秒；较早停和小样本值得诊断，不证明网络必然欠拟合。来源：v8 `round_08/C_no_group/*/result.json`及`final/C_no_group/selected_3566/result.json`。

## 当前依赖

R03在采集109证券2025年8–12月；本次读取 [progress.json](../../../runs/precision_v9_20260927/R03/progress.json)为running、106/545月分片，此数是瞬时快照。未有新基线结果。共享Sol负责日历、146日拆股依赖边界、半日、provenance和旧新行配对；本路线不改采集器。

2026全部已暴露development；旧文件名reserved现在仅是diagnostic。当前池回溯、历史bar_end+1s可用性假设及High触及代理局限不因扩历史消失。下一步见 [BACKLOG](BACKLOG.md)、[矩阵](MATRIX.md)、[实现交接](HANDOFF.md)。
