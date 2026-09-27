# 15 · 每小时一次运行：启动、复现与异常处理

本手册对应 `hourly_once_v3`，不改变旧三时点回放或旧训练制品。所有命令从仓库根目录执行。程序只读行情和模型、写新的研究产物；不接交易账户、订单接口、定时器或 R2 上传。运行目录必须不存在。默认时点是 ET 10:30、11:30、12:30、13:30、14:30、15:30；用户传入的非对齐时间只向过去 floor 到模型制品支持且已完成的同日时点。

## 运行前检查

```bash
git status --short
research/after_open_3d5pct/.venv/bin/python -m unittest discover -s research/after_open_3d5pct/tests -v
lsof -nP -iTCP:8768 -sTCP:LISTEN
lsof -nP -iTCP:11111 -sTCP:LISTEN
```

本地服务通常已在 8768，先检查再决定是否启动，切勿重复启动。OpenD 仅用于必要行情读取，实际连接与响应以当次审计为准。历史源来自 `market_data/us_5m`，源文件无真实逐 bar 接收时间；`available_at=bar_end+1秒` 是回溯假设。当前名单回溯及无夜盘覆盖也会随报告呈现。
一次运行的 Python 3.12 环境需安装 [`requirements-hourly-v3.txt`](../requirements-hourly-v3.txt)，其中在训练依赖外增加 `futu-api`；只装旧 pilot requirements 会使 OpenD 补源不可用。安装命令：`uv pip install --python research/after_open_3d5pct/.venv/bin/python -r research/after_open_3d5pct/requirements-hourly-v3.txt`。

## 训练与校准

训练前先读 [13](13_hourly_once_implementation.md)、[14](14_hourly_v3_training_registration.md) 和配置 [`hourly_v3.json`](../configs/hourly_v3.json)。切分、校准段和候选预算已经登记；训练器只读冻结本地源，不在训练中联网。复跑换新目录，不能覆盖旧版或已存在的 v3 运行：

```bash
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.train_hourly_v3 \
  --config research/after_open_3d5pct/configs/hourly_v3.json \
  --output research/after_open_3d5pct/runs/<new-training-run>
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.export_hourly_v3_summary \
  --run research/after_open_3d5pct/runs/<new-training-run>
```

检查 `manifest.json` 的 completed/failed trials、源与模型哈希、`trials.jsonl` 的每轮内层指标/耗时、`metrics.csv` 的 raw/calibrated 外层指标、`predictions.parquet` 的逐条结果。外层月份全部是已暴露的 development，校准只在内层成熟日期训练；模型选择及阈值不能用外层揭盲结果回填。

## 一次运行

现在调用（程序自动读取当前时间）会尝试在新的 run-local 快照中依 Local→R2→OpenD 次序补充必要的当日源。OpenD 采用限流；真实接收晚于信息截止或最新 5m 缺失时，输出 `data_unverified` 或 `unavailable`，不能冒充实时 fresh。OpenD/R2 不可用时也记录失败来源。默认模型制品由配置的 `enable_fold`、`enable_model` 指定，旧 v2 制品不支持新增小时。

```bash
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.run_once_v3 \
  --output research/after_open_3d5pct/runs/<new-once-run>
```

研究历史指定时刻可用 `--as-of 2026-09-24T13:32:00-04:00 --acquire local`；必须带时区，结果标历史传参，不产生当前可执行建议。`--symbols AAPL NVDA` 可小范围验收；`--exclude-held AAPL` 是用户手动排除列表，不读取或维护真实持仓。`--acquire local` 禁止网络补源，适合复现。输出 `report.json`、`snapshot_manifest.json`、`snapshot_sources/`、可见特征和预测。原始完整源可包含未来和迟到版本，推理内部按 event end / available / received 过滤；不得让调用方先删未来行情。

报告同时列 `requested_at`、`effective_cutoff`、`information_available_deadline`、`generated_at`、参考价时刻和有效期。cutoff 后 30 秒信号、120 秒人工延迟、最多 5 分钟等待第一个合格 5m Open 的标签定义没有改。报告的参考价是 cutoff 最后一根完整 5m Close，最高接受价以“不高于该 Close”的开发期保护规则表示；这**不是**真实入场价。实际晚买、价差、窗口到期或参考价不合条件时旧预测失效，必须重新评估。真正 +5% 目标始终是实际入场价 ×1.05。

## 看板与后续成熟

已有静态服务可看 [每小时研究页](http://127.0.0.1:8768/research/after_open_3d5pct/dashboard/hourly.html)；特定运行加 `?run=<new-once-run>`。页面给逐股状态和可复算特征、行情水位、阈值、同小时开发期成熟分子分母、逐小时/逐日模型结果与训练曲线。旧三时点回放仍在 `dashboard/index.html`。

不传 `--as-of` 的前向调用只追加 `runs/forward_hourly_v3.jsonl`，包含报告哈希和每股状态，不代表用户真实买入。等待窗口成熟并补齐未来 5m 行情后，运行：

```bash
research/after_open_3d5pct/.venv/bin/python -m research.after_open_3d5pct.evaluate_forward_hourly_v3 \
  --output research/after_open_3d5pct/runs/<new-forward-evaluation>.json
```

评估器只写新文件：未满 1170 分钟为 `pending`，日历覆盖不足、缺正常时段 bar、报告哈希不匹配分开记录；无可评估推荐不显示为 0% 胜率。评估仍需核对历史源的真实接收/可用时间。高置信发布另需事先冻结最低触及率、覆盖、风险、样本与校准误差，采集未暴露的前向独立数据并通过门禁。代码和开发期结果本身不构成通过。

## 异常排查

| 状态 | 查看什么 | 处理 |
|---|---|---|
| `unavailable` | 日历、是否在首个合法 cutoff 前、三类源水位及缺口 | 保留无信号；补数据后新建运行，不改旧报告 |
| `data_unverified` | bar 的真实 `received_at`、成员可知时刻、信息截止 | 只作研究观察；不要称 fresh 或买入建议 |
| `expired` | 请求时间和延迟入场有效期 | 下一合法小时重算；旧概率不能条件化到新价格 |
| 训练 trial `failed` | `trials.jsonl` 的 traceback、source hash、calibration class/count | 保留失败，修复后新版本/新目录重跑 |
| 看板无数据 | 8768 服务是否在、`?run=` 是否为已存在目录、`report.json`/摘要是否生成 | 先用 CLI 产出证据，再刷新页面；不重启已有服务 |

所有实际账户持仓、可用资金、下单、成交、退出和跨运行仓位管理见 [12](12_future_research_register.md) 的 backlog，本轮不实现。
