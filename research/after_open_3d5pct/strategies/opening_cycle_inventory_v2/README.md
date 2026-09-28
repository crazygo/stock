# 上涨窗口机会清点 v2

按用户最新要求，先清点不同 T/P 在每只股票中实际发生的次数，并查看前两根5m后还剩多少上涨空间。独立于 opening_continuation_v1，不改变共享模型或执行策略。

- [结果与全101股表](REPORT.md)
- [2026-09-26记录：机会供给偏少，32次的口径与频率](DECISION_2026-09-26.md)
- [运行前定义](PROTOCOL.md)
- [交互证据页](http://127.0.0.1:8768/research/after_open_3d5pct/runs/opening_cycle_inventory_v2_20260925/index.html)

```bash
research/after_open_3d5pct/.venv/bin/python research/after_open_3d5pct/strategies/opening_cycle_inventory_v2/inventory.py --output /absolute/path/to/new-run
research/after_open_3d5pct/.venv/bin/python research/after_open_3d5pct/strategies/opening_cycle_inventory_v2/verify_inventory.py --run /absolute/path/to/new-run
research/after_open_3d5pct/.venv/bin/python research/after_open_3d5pct/strategies/opening_cycle_inventory_v2/build_page.py --run /absolute/path/to/new-run
```

统计脚本拒绝复用既有输出目录。页面可单独以 `--replace-html` 重新渲染，不修改冻结的研究结果。延迟10/15/20分钟是补充描述性敏感性，不是新的预测试验。页面自带原始路径和数据，可离线使用。
