# C 有群 · 实证状态

2026-09-27；本次只读诊断并写交接文档，没有编码、拟合或改动采集器。**R01 已回退，incumbent 仍为 v8 R10；开发动作通过 0/6，诊断通过 0/3。** 目标、九输出、成熟窗口、去重及独立前向验收引用 [共同协议](../../PROTOCOL.md)。2026 全部 exposed，文件名 `reserved` 仅作 development_diagnostic。

## 证据索引

以下路径相对 `research/after_open_3d5pct/`；数字来自实际制品或本次对其只读聚合。

| 索引 | 来源 |
|---|---|
| E1 | `runs/focus_v8_20260927/final_selection.json`；`round_10/C_group/{dev1,dev2}/{registration,result}.json`；`final/C_group/selected_3566/` |
| E2 | `runs/precision_v9_20260927/R00/{metrics.csv,selection.json,signals.parquet}` |
| E3 | `runs/precision_v9_20260927/R01/C_group/decision_before_diagnostic.json`；同目录各折 `result/precision/selection.json` 与 `signals.parquet` |
| E4 | `runs/focus_v8_20260927/dataset/{rows.parquet,features.npz,manifest.json}`；`support_diagnosis.json` |
| E5 | `train_multiscale_v6.py::CModel`；`focus_v8/core.py::FocusC`；`focus_v8/support.py::SupportedC`；`v6_data.py::_group_state` |
| E6 | `precision_v9/rounds/R03/{BACKLOG,MATRIX}.md`；`runs/precision_v9_20260927/R03/progress.json`（采集实时状态，本文不代替验收） |

## 已保留配方与失败改动

- E1：`balance + representation + calibration + support`，seed3566，14,793 参数。无 prior、curves、capacity 或 group_dropout。三尺度为 7 日5m、30日60m、126日日线；群序列为6周×6槽×10通道。
- balance 是 `0.25×全池等权 + 0.75×三操作群等权`；交叉成员分数权重保留。representation 已变换本股相对坐标；support 已冻结日线位置与群类型的 fit 支持，不能再次包装为新方案。
- E3：唯一新改动是训练期逐样本 group dropout=.5，推理保留群。开发等权 Brier **0.2263337541→0.2264351767**，处理后同股 AUC **0.5407326807→0.5314002052**；0/6→0/6，因此回退。不能据此断言所有群输入都无价值。
- E1/E5：群槽为 trend15/63/126、volatility、liquidity、QQQ，来自因果周分群；不是 chips/optics/storage 行业分支。固定操作群仅用于评价和权重，当前池回溯不能升格为 PIT 行业资料。

## 信号迁移与高分误报

| 来源/阶段/群 | raw 阈值 | cal TP/n | eval TP/n；FP | precision | 信号/5官方session；信号日 |
|---|---:|---:|---:|---:|---:|
| R00 diagnostic/chips | 无 | 无合格阈值 | 0/0；0 | null | 0；0 |
| R00 diagnostic/optics | .55 | 16/16 | 21/37；16 | 56.76% | 10.28；8 |
| R00 diagnostic/storage | .45 | 12/12 | 13/20；7 | 65.00% | 5.56；5 |
| R01 diagnostic/optics | .55 | 见E3曲线 | 22/37；15 | 59.46% | 10.28；8 |
| R01 diagnostic/storage | .45 | 见E3曲线 | 13/20；7 | 65.00% | 5.56；5 |

R00、R01 两开发折三群 cal 选择均无解，不能把拒绝算正确。R00固定raw≥.9在dev1 optics为5/6，在dev2 optics为8/12、storage为1/2；高分不等于90%实测。处理后p≥.9三折三群均无信号。

E2仅筛 `route=C_group, fold=reserved, method=cal_selected`，联合 sample_id 去重后 **57信号、34TP、23FP**。其中21/23个FP集中在08-28与09-10；两日分别1/11和0/11命中。09-10信号raw均值.6755，高于08-24全中日的.5911，不能靠再抬阈值解释成可靠择日。

可逐行复核：`AXTI:2026-09-10:11:30:v6` raw=.740114、p=.887752、y=0；`SNDK:2026-09-10:11:30:v6` raw=.718700、p=.886009、y=0。terminal_3d仅是期末估值，不能当止盈成交结果。

E2按sample_id连接E4最新群槽，09-10信号行的有效五类同伴槽 breadth 中位数约.6733、QQQ截至11:30收益约+.50%；09-03全中日分别约.6234、+.96%。**误报日期集中已确认，“弱群导致误报”未确认。** 这些事后切片仅用于提出假设，不按两天设计拒绝规则；六类群可能重叠，槽位聚合不是独立样本。

## 支持与架构事实

| 折 | fit日期/行数 | cal有效日期 | eval官方session | fit完整126日比例 | 冻结支持（日线位置数；群槽） |
|---|---:|---:|---:|---:|---|
| dev1 | 29/3064 | 9 | 19 | 0% | 67；T15、V、L、QQQ |
| dev2 | 58/6090 | 9 | 20 | 0% | 96；T15、T63、V、L、QQQ |
| diagnostic | 86/8997 | 10 | 18 | 0% | 124；T15、T63、V、L、QQQ |

E4逐行复算：dev2/diagnostic eval 完整126日约98.05%/98.04%；最新trend126槽有效率也约98%，但三个旧checkpoint均屏蔽该槽。`support_diagnosis.json`另统计整个六周序列的有效率，不应与最新槽比例混用。126日输入还含20日因果量分母，实际来源可回看146日，交共享数据审计。

E5当前群链路：类型特定10→12映射→同周有效槽均值→GRU(12)→relation；最终头同时接 relation 和 `five×sigmoid(gate(relation))`。因此群既提供附加向量，也乘性重表达本股5m；这不是只接一个行业标签。当前没有证据分离两条贡献，C无群的不同配方不能直接作因果消融。

当前结论：先完成R03原配方匹配基线；随后检验**有界加性群残差**与匹配零残差臂。群影响过强只是待证伪假设，日期共因、数据支持变化与本股排序都尚可能解释误报。
