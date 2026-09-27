# C 无群三轮迭代结果（2026-09-27）

**确认的瓶颈是R0五轮训练过早结束。** 同架构、同输入和同种子的R1延长至最多40轮，内层3日+5% Brier从0.171569降至0.150788；三尺度手工形态R2与主任务加权R3均未超过R1。全部2026日期已暴露，这属于历史开发结果；没有独立验证通过的概率模型或交易建议。

冻结数据为`runs/multiscale_groups_v61_dataset_20260926_r2`，manifest SHA256 `56b3d04def58b72e388f98e41c0d72df1b996aef3ecf7c8ee6325349082a143b`。11:30 ET 截止、11:35代理入场、1/3/5×390常规分钟与+3/+5/+8%九目标不变。9,222行，按最长标签结束及可用时间purge后的train/inner/outer为5,606/1,521/1,140行，58/16/12个ET日期。只加载本股`x5/x60/xday/y`；没有群、QQQ或其他股票信号。R0为v6.1冻结的`C_patch_no_group`。

| 主目标3日+5% | 参数 | 最优轮/运行轮 | 拟合秒 | 内层 Brier / AUC | 9月外层 Brier / AUC | 外层均值/实际率 |
|---|---:|---:|---:|---:|---:|---:|
| R0 五轮基线 | 11,961 | 5/5 | 4.60 | 0.171569 / 0.6626 | 0.157473 / 0.7190 | 0.2870 / 0.2096 |
| R1 延长优化 | 11,961 | 36/40 | 49.30 | **0.150788 / 0.7497** | **0.140021 / 0.7557** | 0.2420 / 0.2096 |
| R2 因果形态 | 12,729 | 21/27 | 35.10 | 0.154024 / 0.7392 | 0.149276 / 0.7436 | 0.1993 / 0.2096 |
| R3 目标加权，seed3566 | 12,729 | 9/15 | 19.92 | 0.156756 / 0.7310 | 0.149521 / 0.7403 | 0.2069 / 0.2096 |
| R3 目标加权，seed81173 | 12,729 | 9/15 | 20.79 | 0.155034 / 0.7374 | 0.149012 / 0.7405 | 0.2200 / 0.2096 |

R1为按内层证据得到的最强主目标候选；R3-3566是预先固定的同种子比较，81173只作重复性诊断。没有按9月外层从R0–R3或双种子里挑模型。R1相对R0主目标配对Brier差，内层−0.020781（5日日期块95%区间[−0.030249,−0.013933]），9月外层−0.017452（[−0.024662,+0.000442]）；后者跨0。R2相对R1内层差+0.003236（[−0.001485,+0.010390]）。所有区间属于选择后的已暴露开发诊断，不能当独立显著性检验。

九目标Brier如下；每个格子保留了实际负例与完整同ID样本，详细logloss/AUC、均值、10个可靠性桶及原始/投影输出见`runs/v7_c_no_group_final_20260927/results.json`。

| 目标 | 内层 R0 | R1 | R2 | R3-3566 | R3-81173 | 外层 R0 | R1 | R2 | R3-3566 | R3-81173 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1日+3% | 0.1626 | 0.1453 | 0.1486 | 0.1500 | 0.1509 | 0.1382 | 0.1221 | 0.1248 | 0.1254 | 0.1264 |
| 1日+5% | 0.0817 | 0.0738 | 0.0754 | 0.0766 | 0.0775 | 0.0591 | 0.0536 | 0.0550 | 0.0557 | 0.0552 |
| 1日+8% | 0.0365 | 0.0342 | 0.0335 | 0.0346 | 0.0351 | 0.0154 | 0.0155 | 0.0146 | 0.0149 | 0.0149 |
| 3日+3% | 0.2387 | 0.2293 | 0.2271 | 0.2273 | 0.2281 | 0.2281 | 0.2138 | 0.2090 | 0.2111 | 0.2104 |
| 3日+5% | 0.1716 | 0.1508 | 0.1540 | 0.1568 | 0.1550 | 0.1575 | 0.1400 | 0.1493 | 0.1495 | 0.1490 |
| 3日+8% | 0.0917 | 0.0784 | 0.0813 | 0.0833 | 0.0845 | 0.0916 | 0.0762 | 0.0878 | 0.0884 | 0.0877 |
| 5日+3% | 0.2446 | 0.2424 | 0.2375 | 0.2369 | 0.2365 | 0.2490 | 0.2431 | 0.2298 | 0.2313 | 0.2333 |
| 5日+5% | 0.2138 | 0.1979 | 0.1957 | 0.1982 | 0.1966 | 0.1981 | 0.1840 | 0.1865 | 0.1841 | 0.1890 |
| 5日+8% | 0.1372 | 0.1211 | 0.1251 | 0.1272 | 0.1276 | 0.1351 | 0.1161 | 0.1288 | 0.1282 | 0.1279 |

R1的主目标内层预测均值0.2461/实际0.2281，外层0.2420/实际0.2096；ECE10为0.0441/0.0604。R2外层ECE10降至0.0459，却有更高Brier与更低AUC，因此分桶均值好看并不代表抓住了更好的机会。R1的0.1–0.2概率桶，外层605行、预测均值0.159、实际率0.102；0.3–0.4桶79行、预测0.342、实际0.443，低概率偏高、高概率偏低。没有额外独立时间段供拟合校准器，保留原始模型估计；九目标单调投影仅是确定性嵌套约束，非概率校准。R0/R1原始九目标无顺序违例；R2内层1.18%、R3两种子2.37%/2.04%，投影后均为0，主目标投影改变很小。

路径语义测试覆盖完整Open→Close收益、跨根开盘跳空、同端点异回撤、内部缺口和无效尾部。R2已强烈使用新增形态：将形态张量置零的分布外推理，内层平均概率变化0.189，Brier由0.1540升至0.2105；这个敏感性不能被解释为重训后的消融增益。R2相对R1改善较长窗口的低阈值目标，却退化主目标；R3三倍主目标损失权重仍未补回。先拟合高阶路径函数或给预测输入未来走势标签没有依据；若未来测试过去趋势分型，应保持历史当时可知并在新时间折独立验证。完整诊断见`DIAGNOSIS.md`与`diagnosis.json`。

冻结覆盖的关键变化：完整126日历史仅占训练行23.47%，内层/外层约97.9%；日级126槽位平均有效率0.855→0.986→0.989。这可能让模型利用历史覆盖识别时期。当前5m前缀每行恒90有效根、60m恒8有效桶；其全窗口有效率约0.469/0.471来自决策后padding，不是缺行情。原价通道量纲比5m收益大得多，但R1同输入可改善，尚不能认定标准化是主要瓶颈。既有当前股票池回溯、历史到达时刻假设、有限日期及重叠标签使结果仍属研究层。

模型`model.pt`、`source_at_fit.py`、`transform.json`、每轮学习曲线和train/inner逐行九目标预测在各`runs/v7_c_no_group_R*_seed*/`目录；外层逐行预测同处，R0重放和全量度量在`runs/v7_c_no_group_final_20260927/`。各预测保存sample_id及逐目标y/raw/p。源代码、基础配置、数据清单、特征NPZ、行表和模型都有SHA256记录。独立重放目录`runs/v7_c_no_group_replay_20260927/`与原结果主目标逐轮精确相同，全部模型train/inner/outer预测最大绝对差0；R0外层与冻结v6.1预测一致。三轮合计拟合约125.1秒，低于15分钟路线预算，无失败拟合或额外网格。

从仓库根目录复现，所有`<new-...>`必须是尚不存在的目录；数值库单线程。最后命令可直接重放已存模型，也可通过`--r1-run`等参数指向新拟合目录。R1/R2的原始拟合代码快照保存在各运行目录，当前runner只增加了R3损失分支与自动保存源快照，不改变R1/R2模型/损失定义。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 research/after_open_3d5pct/.venv/bin/python -m unittest research.after_open_3d5pct.iterations_v7.c_no_group.test_shape -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.iterations_v7.c_no_group.runner --round R1 --seed 3566 --output research/after_open_3d5pct/runs/<new-R1>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.iterations_v7.c_no_group.runner --round R2 --seed 3566 --output research/after_open_3d5pct/runs/<new-R2>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.iterations_v7.c_no_group.runner --round R3 --seed 3566 --output research/after_open_3d5pct/runs/<new-R3a>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.iterations_v7.c_no_group.runner --round R3 --seed 81173 --output research/after_open_3d5pct/runs/<new-R3b>
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.iterations_v7.c_no_group.finalize --output research/after_open_3d5pct/runs/<new-final> --r1-run research/after_open_3d5pct/runs/<new-R1> --r2-run research/after_open_3d5pct/runs/<new-R2> --r3a-run research/after_open_3d5pct/runs/<new-R3a> --r3b-run research/after_open_3d5pct/runs/<new-R3b>
```
