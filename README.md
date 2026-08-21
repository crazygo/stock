# U.S. Stock Drop → Flat Screener

这是一个“候选生成器”，用于每天扫描美国普通股：先在 7 个完整交易日内显著下跌，再在随后 7 个完整交易日内进入低波动平台。它输出形态候选，不直接输出买入建议或收益概率。

## 为什么使用 15 根 K 线

原始示例用 14 根 K 线分成 7+7，但前半段 `index 0 → index 6` 只有 6 个收盘到收盘的收益区间，还漏掉 `index 6 → index 7` 的边界变化。这里使用 15 根：

- `C0 → C7`：7 个下跌期收益区间；
- `C7 → C14`：7 个平台期收益区间；
- `C7` 同时作为平台锚点，第一天再次跳空下跌时不会被误判为横盘。

## 默认筛选条件

- 最新价：$50–$200；
- 7 个交易日跌幅：18%–45%；
- 后续平台总高低区间：≤7%；
- 平台拟合趋势绝对值：≤3.5%；
- 平台日收益标准差：≤2.2%；
- 平台任一日绝对涨跌：≤3.5%；
- 平台平均成交额：≥$20M；
- 默认仅包含 Massive 类型 `CS`（美国普通股），排除 ETF、权证、优先股等。

阈值有意比“4% 总振幅”稍宽，先保证召回率，再用 `shape_score` 排序。`shape_score` 只描述形态质量，不是回涨概率。

## 运行

1. 申请 Massive Stocks API key。免费 Basic 档即可做 EOD 原型。
2. 将 `.env.example` 复制为 `.env`，把新 key 只写入本地 `.env`。`.env` 已被 Git 忽略。
3. 执行：

```bash
python3 screener.py run --config config.example.json
```

也可以不使用 `.env`，直接通过运行环境注入 `MASSIVE_API_KEY`。

第一次运行需要下载至少 15 个交易日的全市场日线和股票目录。默认将未缓存 API 请求间隔设为 12.2 秒，以兼容免费档 5 次/分钟的额度：首次初始化通常需要约 4–6 分钟。股票目录缓存复用 7 天；稳定运行后，绝大多数交易日只需 1 次全市场行情调用，目录刷新日再增加若干分页调用。

输出位于 `results/`，其中固定入口用于自动读取，日期目录用于历史追踪：

- `results/latest.md`：最近一次人类可读报告；
- `results/latest.json`：最近一次机器可读报告；
- `results/YYYY-MM-DD/screen.md`：当天归档 Markdown；
- `results/YYYY-MM-DD/screen.json`：当天完整结构化数据；
- `results/YYYY-MM-DD/candidates.csv`：候选明细。

## 全市场每日归档

筛选结果和源行情数据分开保存：

- `.cache/massive/bars/YYYY-MM-DD.json`：运行缓存，只用于减少 API 请求，不属于长期数据资产；
- `market_data/us/YYYY-MM-DD/grouped.json.gz`：Massive grouped daily 接口的完整原始 JSON 响应，经确定性 gzip 压缩后提交进 Git；
- `market_data/us/manifest.json`：归档索引，记录交易日、ticker 数量、未压缩/压缩大小与原始数据 SHA-256。

每日 workflow 在筛选完成后执行：

```bash
python3 archive_market_data.py \
  --all-cached \
  --results-json results/latest.json \
  --cache-dir .cache/massive \
  --archive-dir market_data/us \
  --min-results 1000
```

`--all-cached` 会把 Actions cache 中已有且通过完整性校验的历史交易日一并回填。当天 `results/latest.json` 对应的交易日必须成功归档，否则 workflow 失败。默认要求至少 1,000 个 ticker，防止接口异常或部分响应被误写成完整市场历史。

归档保存的是 Massive `/v2/aggs/grouped/locale/us/market/stocks/{date}` 的完整响应，而不是当前筛选器抽取后的字段。因此以后新增 VWAP、交易笔数、ETF/普通股重新分类或重新计算筛选参数时，可以直接读取历史源数据，无需重新请求行情接口。

运行测试：

```bash
python3 -m unittest -v test_screener.py test_archive_market_data.py
```

## 每日工作流

推荐分两层：

1. 本代码在美股收盘且 EOD 数据就绪后运行，生成候选列表，并把完整当日全市场源数据归档；
2. ChatGPT 只研究候选股的暴跌原因、消息是否破坏长期逻辑、估值、量化业务权重、未来 30/60 日催化与下降旗形风险。

先手动检查前 3–5 次输出，再开启无人值守任务。正式用资金前，应增加两年回测，至少比较：20 日内先涨 8% 的概率、20 日收益中位数、最大不利波动，以及相对“所有同期跌 18% 的股票”的增量优势。

## GitHub Actions

仓库 Secret 中添加 `MASSIVE_API_KEY`。任务在周二至周六 05:30 UTC 执行（美东午夜后），等待 Basic 免费版释放前一交易日 EOD 数据。若某个日期尚未释放，脚本会跳过并寻找最近可用交易日。报告和 `market_data/us/` 归档由 `github-actions[bot]` 一起提交回仓库。`results/**` 与 `market_data/**` 被 workflow 的 push trigger 忽略，避免机器人提交再次触发自身。

后续 ChatGPT 可以固定读取 `results/latest.md` / `results/latest.json` 做每日候选研究，也可以读取 `market_data/us/manifest.json` 和对应日期的 `grouped.json.gz` 做历史窗口、参数敏感性和回测分析。

`daily_research_prompt.md` 已包含第二层研究的固定提示词，重点防止把漂亮形态误当成上涨概率，并保留“业务权重必须量化”的要求。
