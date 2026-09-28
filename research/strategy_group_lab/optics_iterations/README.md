# 光产业链：Pugh 矩阵与三轮迭代

先读 [冻结协议](PROTOCOL.md)。本轮保留3日+5%、5日+8%、21日+10%，补齐12股光产业链；原58群保持原义，新增7群。原工程、数据源和旧结果没有被覆盖。

## 本轮交付

- [交互报告](../runs/optics_3round_20260926/index.html)：事实走势、覆盖诊断、设计及实测Pugh矩阵、全路线结果、最新概率、学习曲线。
- [结论与证据](../runs/optics_3round_20260926/REPORT.md)、[全量3960行结果](../runs/optics_3round_20260926/results.csv)。
- [审计](../runs/optics_3round_20260926/audit.json)、[成对对照](../runs/optics_3round_20260926/comparisons.json)、[具体漏报及盈亏分解](../runs/optics_3round_20260926/diagnostics.json)。

**结果没有通过策略验收。** 三轮216个模型全部完成，48项可检验的成对概率改进，Holm通过0项；21日独立日期块不足。共同成熟窗口光产业链整体账户的84个轮次/路线/期望组合全部亏损。9月部分改善单独展示，不代替共同窗口与独立验证。

主路线R3（LightGBM、5m、全池、12股产业链），共同窗口三目标扣费收益分别为−22.29%、−15.19%、−8.44%；分类准确率52.78%、57.05%、75.21%。未达标时损失明显超过多次止盈收益。多数模型内层验证选择更早停止，不支持普遍增加训练次数。

## 文件及口径

| 文件 | 职责 |
|---|---|
| `acquire.py` | 隔离缓存内 Local→R2→OpenD 获取12股+QQQ；不上传 |
| `prepare.py` | 扩展候选与群版本，调用原工程因果特征/成熟标签构建 |
| `facts.py` | 20日/5日价格事实与旧覆盖、展示遗漏的证据 |
| `features.py` | 历史路径、当日已知走势、同cutoff且排除自身的同群上下文 |
| `train.py` | R0旧权重重放、R1/R2/R3训练、成熟边界、校准、R3内层早停 |
| `portfolio.py` | 从观察到的行情模拟成交，未到期持仓估值，现金/盈亏对账 |
| `evaluate.py` | 全部绑定结果，同群B0、同股票日期成对对照、多重检验 |
| `audit.py` | 源文件哈希、训练边界、检查点重放、账户守恒 |
| `render.py` / `report.html` | 自包含HTML、结论Markdown、漏报和损失分解 |
| `test_invariants.py` | 5项时间、同群、资金、pending不变量测试 |

所有期望按 `days × 390` 常规盘分钟计时，从11:35 ET Open代理价出发，允许跨天；不是“第三天收盘”的定义。截止11:30，历史available_at=end+1秒是假设，因此恰好11:30结束的柱不会提前进入特征。

R2使用20个历史日槽位；20槽位之间覆盖19个收盘变化区间。显式收益滞后为1/3/5/10/19日对数变化，另有当前参考价相对前日收盘。协议中的“20日”描述历史路径长度，不额外声称存在第21个收盘锚点。

R1/R2/R3均训练全池及12股产业链模型。其他群是分层评价绑定；本轮没有为每个子群另训模型。原群内模型记录仍在原运行目录。共同窗口07-01—08-25，估值09-25；近期窗口09-01—09-25，未成熟标签保留pending，已发生目标触及可以退出，未平仓收盘估值。

## 复现

依赖本仓库已有研究虚拟环境、原运行`20260926_v1_checked`（含旧模型检查点）及原群源。运行制品与行情均位于忽略的`runs/`，不进入Git。首先确认数据和OpenD读权限；`acquire`会获取行情，其余步骤离线。

```bash
PY=research/after_open_3d5pct/.venv/bin/python
OPTICS_RUN=research/strategy_group_lab/runs/optics_3round_20260926
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 新实验应选择新的 OPTICS_RUN，避免改变现有冻结输入。
$PY -m research.strategy_group_lab.optics_iterations.acquire "$OPTICS_RUN"
$PY -m research.strategy_group_lab.optics_iterations.prepare "$OPTICS_RUN"
$PY -m research.strategy_group_lab.optics_iterations.facts "$OPTICS_RUN"
$PY -m research.strategy_group_lab.optics_iterations.features "$OPTICS_RUN"
for ROUND in R0 R1 R2 R3; do
  $PY -m research.strategy_group_lab.optics_iterations.train "$OPTICS_RUN" --round "$ROUND" || break
done
$PY -m research.strategy_group_lab.optics_iterations.evaluate "$OPTICS_RUN"
$PY -m research.strategy_group_lab.optics_iterations.audit "$OPTICS_RUN"
$PY -m research.strategy_group_lab.optics_iterations.render "$OPTICS_RUN"
$PY -m unittest research.strategy_group_lab.tests.test_lab research.strategy_group_lab.optics_iterations.test_invariants -v
```

`train`在每轮训练前登记代码、特征、标签、配置与原始源清单哈希；任何差异拒绝复用已有模型。失败记录保留，评价入口遇到失败则拒绝生成完整结论。一次Pandas 3只读数组兼容修复的初始尝试保存在`infrastructure_attempt_before_pandas_fix/`；没有读取外层结果后调整设计。

## 证据边界

当前业务分类及股票池为快照回溯，所有外层为已接触历史上的开发期回放。R0对新增股票仅为迁移诊断；9月仍用旧8月检查点。R1同时改覆盖/时效，不能分离两者的单因素贡献。近期只有12股+QQQ补齐，其他群不输出近期资金结果；26只ETF仍不可评价。

只预测触及不能约束未达标损失，下一研究优先考虑独立保留触及概率并联合建模下行和持仓时间。本次严格结束于三轮；不按9月收益选择上线赢家、不接真实订单。
