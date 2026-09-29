# 模型名册

每个子目录是一个模型，互不覆盖。四项都写在该目录的 `spec.json` 里，现在数值可以相同，以后只改自己的文件：

1. `target`：买入动作看的标签，例如 3 个交易日最高价触及 +5%。
2. `inputs`：这个模型吃哪些行情。
3. `universe`：对哪些股票出概率。全体 QQQ、一个群、或一只股票，都写在这里。
4. `code`：训练入口和配方。新点子使用新目录里的新模块，不改父模型的入口。

已有的 `focus_v8/` 与 `precision_v9/` 仍是原训练和原证据，本目录不搬动它们。新的拟合写到 `runs/model_registry_v1/<模型 id>/`。

五个父模型各有三份组模型，id 是 `<父模型>_chips`、`<父模型>_optics`、`<父模型>_storage`，分别只拟合和预测半导体组、光通信组、存储组。`SOXS_SOXX_3d5pct` 每天只在买入 SOXS、买入 SOXX、不行动里选一个，目标仍是 3 天触及 +5%。

增加一个模型：新建目录，写 `spec.json`，把 id 追加到 `registry.json`，然后运行

```bash
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.models.launch --only <id>
```

父模型的 checkpoint 先重放。重放最大误差超过 `1e-4` 时，不采用这次新拟合的准确度。

尚未训练的点子记在 `ideas/`。交易启动时点是否适合预测，见 [`ideas/clock_time_predictability.md`](ideas/clock_time_predictability.md)。

Pugh 矩阵里 5 分及以上的四项，以及跑完锁定周后的复盘，见 [`pugh_ge5/PROTOCOL.md`](pugh_ge5/PROTOCOL.md) 和 [`pugh_ge5/REVIEW.md`](pugh_ge5/REVIEW.md)。2024-01-01 起的加长重训见 [`pugh_ge5/REPORT_2024.md`](pugh_ge5/REPORT_2024.md)。个股九成买入的筛选见 [`per_stock_90/REPORT.md`](per_stock_90/REPORT.md)。下一轮训练维度见 [`per_stock_90/PUGH_NEXT.md`](per_stock_90/PUGH_NEXT.md)。微观曲线和宏观曲线在训练、验证、2026 测试三段上的作用见 [`curve_split/REPORT.md`](curve_split/REPORT.md)。按周滑动的微观 × 宏观见 [`weekly_scale/REPORT.md`](weekly_scale/REPORT.md)。八周窗口上每个模型的破局见 [`break_v1/REPORT.md`](break_v1/REPORT.md)。30 日趋势乘上午推动的势能见 [`potential_v1/REPORT.md`](potential_v1/REPORT.md)。滑动微观窗口走完再进场的两轮见 [`micro_entry_v1/REPORT.md`](micro_entry_v1/REPORT.md)。
