# 增长区间提前触发规律研究（21 只置顶股）

对应 backlog：`backlog/2026-09-23-growth-trigger-research.md`。

## 结论（先说）

**未找到通过事前门槛的触发规律。** 详见 `report.md` §1 与 §8。

## 复算顺序

```bash
cd /Users/admin/Code/stock
PYTHONPATH=. python3 analysis/growth_trigger/build_sample.py         # 标签快照 + 预测点样本
PYTHONPATH=. python3 analysis/growth_trigger/verify_page_labels.py   # 标签口径与页面一致
PYTHONPATH=. python3 analysis/growth_trigger/make_report.py          # 指标 + 报告 + 审查图
PYTHONPATH=. python3 analysis/growth_trigger/make_audit_sample.py    # 人工核验抽样
```

`run_candidates.py` 是被 `make_report.py` 复用的指标库（也可单独跑，输出 `eval_report.json`）。

## 文件

| 文件 | 内容 |
|---|---|
| `spec.md` | 冻结规格 v1：口径、特征、候选规则、阈值、分组、通过标准（看评估结果前写定） |
| `labels_8pct_v1.json` | 8% 区间标签快照：参数、数据版本、生成时间、逐股区间 |
| `points.jsonl` | 全部预测点（约 5.4 万行）：特征原料 + 两种结果判定 |
| `events.json` | 冻结区间及奇偶归属 |
| `sample_stats.json` | 样本量、基准命中率、无法判定计数 |
| `eval_report.json` | 全部规则 × 训练/评估 × 三种去重口径的指标 |
| `signals.jsonl` | 逐条触发信号：触发时证据、入场价、是否达成、最大涨幅 |
| `signals_review.html` | 逐股日 K 线图：绿色阴影=历史区间，三角=预测触发点 |
| `human_audit_sample.json` | 正例/负例/边界例/被拒例抽样，供人工核验 |
| `report.md` | 结论与全部指标 |

## 关键口径

- 预测点 = 21 只置顶股、每个合格交易日、小时 bar 标签 05:00–20:00（美东）。
- 入场价 = 信号 bar 收盘价；达成 = 入场后 3–14 个自然日内，含盘前/盘后/夜间的最高价 ≥ 入场价 ×1.08。
- 分组：偶数日训练、奇数日评估；评估事件剔除与同一股票训练期区间相距 ≤14 个自然日者。
- 门槛（事前固定）：≥15 个去重信号、召回率 ≥25%、精准率 ≥1.3×基准且 bootstrap 95% 下界高于基准。

## 数据注意事项

- 小时线 `time_key` 是**美东 bar 结束时刻**；常规时段 = 标签 10:30–16:00。
- Futu 小时线成交量约为日线档案的 91%；分红股前复权价与日线档案差 0.1%–0.5%。
- SNOW/SPCX/TSM 在 point-in-time 资格门槛下没有合格股票日（SPCX/TSM 无 SEC 年报事实、SNOW 净利润为负），实际有效股票池是 18 只。
