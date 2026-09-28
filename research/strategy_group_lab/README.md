# 多策略 × 股票群实验室

本工程把 **收益期望 × 算法 × 输入粒度 × 训练范围 × 股票群** 作为独立可比较的实验行，保存概率预测与资金回测。原模型与原股票群输出保持独立。

先读 [冻结协议](PROTOCOL.md)。初始配置 [v1.json](configs/v1.json)：3日+5%、5日+8%、21日+10%；LightGBM/TCN；5m/60m；58个版本化股票群。共有 **1,380 个模型组合 + 174 个同群发生率基准**。每组合两个时间折，训练不足与失败也登记。

## 查看本轮

- **光产业链事实核查与三轮迭代（09-26）**：[工程入口](optics_iterations/README.md)、[交互报告](runs/optics_3round_20260926/index.html)。保留原期望，完成216个新模型；覆盖修复，但尚未建立稳定预测优势。

- [交互结果表](runs/20260926_v1_checked/index.html)：搜索群、切换期望/算法/粒度/训练范围，查看分类准确率、Brier、候选触及率、资金收益；点击行查看月份、校准、逐笔交易与资金曲线。
- [完整 CSV](runs/20260926_v1_checked/results.csv)、[逐条预测](runs/20260926_v1_checked/predictions.parquet)、[全部训练记录](runs/20260926_v1_checked/trials.json)。
- [整体周选择器](runs/20260926_v1_checked/router.json)：只用当周之前已成熟的预测证据选组合，多群重复股票去重后使用一个账户。
- 结果的具体解释见 [本轮证据](EVIDENCE.md)。

`runs/` 是本地可再生制品，不进 Git；报告内嵌数据，可独立离线打开。Parquet、模型检查点、CSV、账本、源文件哈希保留在运行目录。

## 复跑

在仓库根目录使用已安装研究依赖的虚拟环境：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
research/after_open_3d5pct/.venv/bin/python -m research.strategy_group_lab.run \
  --output research/strategy_group_lab/runs/<新的运行编号>
```

默认离线，不下载数据、不请求券商、不推送R2。若本地行情与冻结群源哈希不一致，拒绝运行；更新行情应先形成新的群版本并修改配置。首次运行目录必须不存在。显式 `--resume` 只允许在配置与源数据一致时续跑。可用 `--stage data|train|evaluate` 分阶段执行。

数据和模型实现哈希改变也会拒绝复用已有训练；评价或页面修复单独记录执行代码版本，不改变已保存概率。首轮运行先做了数据预检，发现粗粒度末段需要保留55分钟有效信息，修复后使用 `20260926_v1_checked` 完整重建再开始训练；预检目录没有训练或选优。

```bash
research/after_open_3d5pct/.venv/bin/python -m unittest discover \
  -s research/strategy_group_lab/tests -v

OMP_NUM_THREADS=1 research/after_open_3d5pct/.venv/bin/python \
  -m research.strategy_group_lab.audit research/strategy_group_lab/runs/20260926_v1_checked
```

## 文件结构

| 文件 | 职责 |
|---|---|
| `data.py` | 当时可知的多粒度 H/A/B 特征、周版本群归属、分离的多期限标签 |
| `models.py` | 固定超参 LightGBM/TCN、仅训练集预处理、内层概率校准 |
| `run.py` | 冻结源、断点续跑、完整实验登记和模型检查点 |
| `portfolio.py` | 同一资金约束、成本、持仓去重、5m净值与逐笔对账 |
| `evaluate.py` | 同群基准、概率与分类指标、日期块比较、因果周选择器 |
| `report.html` / `render.py` | 单文件研究对照表及内嵌账本 |

## 解释边界

这些是已暴露历史上的开发期前向回放，不是未接触独立测试。行业/ETF/候选池当前名单回溯；部分群只含一只证券或缺训练历史，群内模型明确跳过，全池模型仍可分群评价。不可把最高回测收益组合追认成事先选中的策略。真正可执行的整体选择结果看周选择器的单账户记录。

概率表示未来窗口内触及目标的机会，资金收益还受入场、等待时间、持仓约束、未触及时的跌幅与费用影响。原先单目标模型的负结果不会被覆盖；本轮用于发现下一轮应冻结检验的组合。
