# 三组预报模型 v8

用户目标：提高计算与通信芯片、光通信产业链、数据与存储的预报准确度，五条路线均保留，最多十轮瓶颈诊断→Pugh比较→训练→评价。目标名单已获用户确认。

已完成五路线各10轮，共125次路线/时间折拟合。没有路线达到预登记准出标准；不是生产模型交付。结果入口：[结果与瓶颈](REPORT.md)、[Pugh矩阵](PUGH.md)、[逐轮表](ROUNDS.md)、[九目标/个股概率](PROBABILITIES.md)、[审计](AUDIT.json)。

入口：[冻结协议](PROTOCOL.md)、[机器配置](protocol.json)。实验运行目录为 `../runs/focus_v8_20260927`，数据和模型不会提交Git。过程先以两段开发结果选配方，最后才开启本轮保留段；2026历史此前已暴露，任何保留段结果都不属于独立前向证明。

## 如何复现

从仓库根目录使用工程`.venv/bin/python`，设置`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`。准备命令拒绝已经存在的目录：

```bash
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.focus_v8.prepare research/after_open_3d5pct/runs/NEW_RUN
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.focus_v8.run research/after_open_3d5pct/runs/NEW_RUN --round 0
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.focus_v8.run research/after_open_3d5pct/runs/NEW_RUN --round 1
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.focus_v8.evaluate research/after_open_3d5pct/runs/NEW_RUN --gate-round 1
```

依次完成2–10轮，每轮先检查门禁和记录；不能跳轮。每条路线从当前最佳配方出发测试一个尚未试过的机制，未改善则回退。十轮完成后使用`run ... --final`冻结配方并完成两种子保留段复验，最后用`python -m research.after_open_3d5pct.focus_v8.deliver NEW_RUN`生成结果表及最终门禁。`evaluate --gate-round`仅负责开发段门禁，不能替代最终准出。准出资格与运行完成是不同字段。R10新增候选依据另见[R10登记](R10_SUPPORT_AMENDMENT.md)。

## 五条路线

| 路线 | 数值摘要/模型 | 三个价格尺度 | 群输入 |
|---|---|---|---|
| B有群 | LightGBM，九个独立目标 | 5m、60m、日线摘要 | 六类市场群/QQQ的六周状态 |
| B无群 | LightGBM，九个独立目标 | 同上 | 不输入 |
| C有群 | 小型patch网络，多目标 | 三个并行分支 | 同上 |
| C无群 | 小型patch网络，多目标 | 三个并行分支 | 不输入 |
| C无日级 | 小型patch网络，多目标 | 5m、60m两个分支 | 保留群摘要，包括长期趋势分类 |

C无日级的准确含义是“无个股日线分支”，不是“完全没有长历史信息”。若某轮加入成熟本股标签先验，另行明确登记。

每根价格量输入包括局部收益、High/Low相对Open、跨根缺口、成交额/成交量绝对与相对值、时段与间隔、可见性mask等14个通道。数学摘要不是新增市场事实，而是让有限样本模型更容易识别路径的归纳偏置；是否有价值以逐轮开发对照判断。不拟合高阶多项式，不用事后拐点或未来趋势作输入。

数据快照有14,424个完整股票日；原101股加7只光产业链股票，共108候选股。11:30信息、11:35入场的同一成熟样本用于全部路线。原有5m数据覆盖04:00–20:00 ET，隔夜以真实缺口表示，不假造夜盘K线。早期样本的126日日线不完整：新时间折fit段完整覆盖比例均为0%，最后两段评价接近98%，这个输入分布变化必须随结果报告。

测试：`python -m unittest research.after_open_3d5pct.focus_v8.test_core -v`，并按上级README运行84项回归测试与synthetic smoke。回放检查针对保存模型的数值一致性，不证明真实收益或线上到达延迟。
