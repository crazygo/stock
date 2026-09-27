# Pugh方案选择记录

符号依次为概率偏差、同股择日、跨期稳定、样本/过拟合风险、计算成本；权重3/3/2/2/1。符号是相对当前配方的待验机制判断，不是已经观测到的收益。

| 轮次 | 路线 | 当前平均校准差 | 同股AUC | 选中机制 | 五项符号 | 分数 | 下一候选 | 实验后决定 |
|---|---|---:|---:|---|---|---:|---|---|
| R1 | B有群 | 9.0% | 0.568 | balance | + / + / + / 0 / + | 9 | representation | 回退 |
| R1 | B无群 | 10.8% | 0.518 | balance | + / + / + / 0 / + | 9 | representation | 回退 |
| R1 | C有群 | 10.3% | 0.439 | balance | + / + / + / 0 / + | 9 | representation | 保留 |
| R1 | C无群 | 8.5% | 0.457 | balance | + / + / + / 0 / + | 9 | representation | 回退 |
| R1 | C无日级 | 11.0% | 0.455 | balance | + / + / + / 0 / + | 9 | representation | 保留 |
| R2 | B有群 | 9.0% | 0.568 | representation | + / + / + / 0 / 0 | 8 | prior | 保留 |
| R2 | B无群 | 10.8% | 0.518 | representation | + / + / + / 0 / 0 | 8 | prior | 保留 |
| R2 | C有群 | 7.3% | 0.583 | representation | + / + / + / 0 / 0 | 8 | prior | 保留 |
| R2 | C无群 | 8.5% | 0.457 | representation | + / + / + / 0 / 0 | 8 | prior | 回退 |
| R2 | C无日级 | 6.1% | 0.490 | representation | + / + / + / 0 / 0 | 8 | prior | 回退 |
| R3 | B有群 | 8.9% | 0.591 | prior | + / + / + / 0 / 0 | 8 | regularization | 回退 |
| R3 | B无群 | 11.5% | 0.610 | prior | + / + / + / 0 / 0 | 8 | regularization | 回退 |
| R3 | C有群 | 11.0% | 0.501 | prior | + / + / + / 0 / 0 | 8 | curves | 回退 |
| R3 | C无群 | 8.5% | 0.457 | prior | + / + / + / 0 / 0 | 8 | curves | 保留 |
| R3 | C无日级 | 6.1% | 0.490 | prior | + / + / + / 0 / 0 | 8 | curves | 回退 |
| R4 | B有群 | 8.9% | 0.591 | regularization | + / 0 / + / + / + | 8 | bagging | 保留 |
| R4 | B无群 | 11.5% | 0.610 | regularization | + / 0 / + / + / + | 8 | bagging | 保留 |
| R4 | C有群 | 11.0% | 0.501 | curves | 0 / + / + / 0 / 0 | 5 | regularization | 回退 |
| R4 | C无群 | 13.5% | 0.452 | curves | 0 / + / + / 0 / 0 | 5 | regularization | 回退 |
| R4 | C无日级 | 6.1% | 0.490 | curves | 0 / + / + / 0 / 0 | 5 | regularization | 回退 |
| R5 | B有群 | 8.5% | 0.641 | bagging | + / 0 / + / + / 0 | 7 | curves | 回退 |
| R5 | B无群 | 8.7% | 0.630 | bagging | + / 0 / + / + / 0 | 7 | curves | 回退 |
| R5 | C有群 | 11.0% | 0.501 | regularization | 0 / 0 / + / + / + | 5 | specialize | 回退 |
| R5 | C无群 | 13.5% | 0.452 | regularization | 0 / 0 / + / + / + | 5 | specialize | 回退 |
| R5 | C无日级 | 6.1% | 0.490 | regularization | 0 / 0 / + / + / + | 5 | specialize | 保留 |
| R6 | B有群 | 8.5% | 0.641 | curves | 0 / + / + / 0 / 0 | 5 | specialize | 保留 |
| R6 | B无群 | 8.7% | 0.630 | curves | 0 / + / + / 0 / 0 | 5 | specialize | 回退 |
| R6 | C有群 | 11.0% | 0.501 | specialize | + / + / 0 / − / + | 5 | recency | 回退 |
| R6 | C无群 | 13.5% | 0.452 | specialize | + / + / 0 / − / + | 5 | recency | 保留 |
| R6 | C无日级 | 6.1% | 0.493 | specialize | + / + / 0 / − / + | 5 | recency | 回退 |
| R7 | B有群 | 8.5% | 0.646 | specialize | + / + / 0 / − / + | 5 | recency | 回退 |
| R7 | B无群 | 8.7% | 0.630 | specialize | + / + / 0 / − / + | 5 | recency | 回退 |
| R7 | C有群 | 11.0% | 0.501 | recency | + / 0 / + / − / + | 4 | calibration | 回退 |
| R7 | C无群 | 9.9% | 0.453 | recency | + / 0 / + / − / + | 4 | calibration | 回退 |
| R7 | C无日级 | 6.1% | 0.493 | recency | + / 0 / + / − / + | 4 | calibration | 回退 |
| R8 | B有群 | 8.5% | 0.646 | recency | + / 0 / + / − / + | 4 | calibration | 回退 |
| R8 | B无群 | 8.7% | 0.630 | recency | + / 0 / + / − / + | 4 | calibration | 回退 |
| R8 | C有群 | 11.0% | 0.501 | calibration | + / 0 / + / − / + | 4 | bagging | 保留 |
| R8 | C无群 | 9.9% | 0.453 | calibration | + / 0 / + / − / + | 4 | bagging | 保留 |
| R8 | C无日级 | 6.1% | 0.493 | calibration | + / 0 / + / − / + | 4 | bagging | 保留 |
| R9 | B有群 | 8.5% | 0.646 | calibration | + / 0 / + / − / + | 4 | capacity | 保留 |
| R9 | B无群 | 8.7% | 0.630 | calibration | + / 0 / + / − / + | 4 | capacity | 保留 |
| R9 | C有群 | 3.1% | 0.503 | bagging | 0 / 0 / + / + / 0 | 4 | capacity | 回退 |
| R9 | C无群 | 6.4% | 0.454 | bagging | 0 / 0 / + / + / 0 | 4 | capacity | 回退 |
| R9 | C无日级 | 6.7% | 0.493 | bagging | 0 / 0 / + / + / 0 | 4 | capacity | 回退 |
| R10 | B有群 | 6.5% | 0.646 | capacity | 0 / + / 0 / − / − | 0 | — | 回退 |
| R10 | B无群 | 6.1% | 0.628 | capacity | 0 / + / 0 / − / − | 0 | — | 回退 |
| R10 | C有群 | 3.1% | 0.503 | support | + / 0 / + / + / + | 8 | capacity | 保留 |
| R10 | C无群 | 6.4% | 0.454 | support | + / 0 / + / + / + | 8 | capacity | 回退 |
| R10 | C无日级 | 6.7% | 0.493 | support | + / 0 / + / + / + | 8 | capacity | 保留 |

R10新候选的独立登记见 [R10_SUPPORT_AMENDMENT.md](R10_SUPPORT_AMENDMENT.md)。完整候选矩阵和训练前登记保留在运行目录各轮registration.json。
