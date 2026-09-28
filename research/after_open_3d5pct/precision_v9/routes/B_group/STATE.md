# B有群 · 实际状态与 R01 复盘

2026-09-27，只读既有真实制品；没有新拟合。**R01 回退，仍保留 v8 R9 配方；两个开发折 0/6 单元通过，诊断段 0/3。** R03 长历史由 root 采集中，本路线未验收完成。共同目标、precision/供给、去重、九目标成熟及前向验收全部引用 [PROTOCOL](../../PROTOCOL.md)，不在此修改。

## 精确来源与版本

以下制品路径相对 `research/after_open_3d5pct/`：

- 保留配方：`runs/focus_v8_20260927/round_10/summary.json` 的 `B_group.incumbent_recipe`：representation、regularization、curves、calibration；类别保留。开发模型为 `round_09/B_group/{dev1,dev2}/`，诊断模型为 `final/B_group/selected_3566/`。
- 最新候选：`runs/precision_v9_20260927/R01/B_group/{dev1,dev2,reserved}/` 下 registration/result/selection/precision.json；只新增 `no_group_categories=true`。回退决定在同路线 `decision_before_diagnostic.json`。
- 旧动作对照：`runs/precision_v9_20260927/R00/{selection,metrics}.json`。源码核对：`precision_v9/{evaluate,train}.py`、`focus_v8/{core,run}.py`、`train_multiscale_v6.py`、`iterations_v7/b_group/experiment.py`。

| 冻结项 | SHA-256 |
|---|---|
| features.npz | `6bce8471011f1c2c54ce2551ba7584f18196fb5b056580e528d95532d5c0e744` |
| rows.parquet | `b6220c3645ff6477ff59763d75deaff26b6f60c7b1ad34f6d0b2b913a4fed2fe` |
| manifest.json | `43bbf25f88c62313b610657b77b1ae99e4b45973fa41a91cef4f8279a067d9e9` |
| R01 train.py | `65ab3544d0eca645beb9d95f04e936c0a4f03d1e04c03fd7f1fc7d5a4bfad7a0` |
| R01 evaluate.py | `a09c38e57babca1c1a4f69dc688d47f7aaf4490b58cbb23e84f15527f8707795` |

后三折均种子3566、863列；主目标树数119/69/11；checkpoint复放误差均0。开发等权 Brier **0.2202894482→0.2207367588**，同股AUC **0.6462643868→0.6422211294**；类别捷径不是本轮得到支持的主要根因。

## 动作与校准支持

| 折 | eval 日期（左闭右开） | fit/tune/cal/eval全池行数 | cal有效日 / eval官方session |
|---|---|---|---|
| dev1 | 06-01～06-27 | 3064/1045/936/1975 | 9/19 |
| dev2 | 07-13～08-08 | 6090/934/930/2056 | 9/20 |
| diagnostic | 08-24～09-18 | 8997/1030/1021/1836 | 10/18 |

| 折/群 | cal raw阈值 | cal TP/n（日数） | eval TP/n；FP | precision | 每周信号；信号日 |
|---|---:|---:|---:|---:|---:|
| dev1/optics | .65 | 14/15（7） | 23/44；21 | 52.27% | 11.58；16 |
| diagnostic/storage | .40 | 12/12（3） | 5/7；2 | 71.43% | 1.94；3 |
| 其余7个单元 | 无 | 无合格阈值 | 0/0；0 | null | 0；0 |

R00旧配方 dev1/optics 对应统计同为23/44；diagnostic/storage为阈值.35、13/20、7FP、65.00%、5.56/周。R01 storage少发13条后点估计变好，仍未达标；诊断结果不推翻预先回退。入选信号terminal_3d均值/最差：optics −1.05%/−15.19%，storage +0.61%/−6.00%；这是期末持有估值，未含成本且不是止盈成交收益。

**选择器稀疏是事实，过严为主因未获证明。** cal只有9/9/10日期，受到完整九目标成熟/purge限制。dev1 chips .75有3/3、2日，被最低支持挡住；但dev2 chips支持合格时最高4/7，optics .60有16/23且供给12.78/周，storage .60有8/9，均不能放宽成90%。诊断cal raw≥.45全部零信号，storage .30/.35/.40均12/12，平局选.40；后续7条均分仅.400356，提示分数压缩/阈值边界敏感，尚未证明网格是根因。阈值变化会重跑冷却，集合可能非嵌套，不能只靠曲线反转认定模型排序反向。

## 同股、跨股及迁移

| R01 eval | chips pooled/同股AUC | optics pooled/同股AUC | storage pooled/同股AUC |
|---|---:|---:|---:|
| dev1 | .6956/.6176 | .6507/.5192 | .8085/.8212 |
| dev2 | .6760/.6451 | .7488/.6994 | .5707/.5508 |
| diagnostic | .6568/.5371 | .6799/.5663 | .4795/.4490 |

上表来自result.json的处理后p（Platt+九目标投影）；动作使用raw，尚缺raw对应诊断。同股AUC按同股正负配对数加权；pooled不是纯跨股AUC。诊断cal optics pooled=.8160而同股=.4860，说明总体排序不能代替时机证据。诊断三群实际率47.92/43.98/47.22%，预测均值55.17/57.64/58.91%，均过估。

`focus_v8/REPORT.md`确认所有fit完整126日比例0%、部分长期群类型fit缺失而eval出现。既有B特征已经含群末值、六周均值、末减首及本股减群当前收益；再说“增加群差分”会重复旧机制。当前新假设必须限定为同日同一同伴集合的两段小时变化。2026全部exposed；R03当前池回溯和历史延迟假设不因扩量消失。逐股票/日期FP迁移、raw排序及分数唯一值仍是待产出诊断，不能写成已确认根因。
