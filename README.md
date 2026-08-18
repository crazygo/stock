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

输出位于 `reports/`：

- `candidates_YYYY-MM-DD.csv`
- `candidates_YYYY-MM-DD.json`
- `latest_candidates.json`

运行测试：

```bash
python3 -m unittest -v test_screener.py
```

## 每日工作流

推荐分两层：

1. 本代码在美股收盘且 EOD 数据就绪后运行，生成候选列表；
2. ChatGPT 只研究候选股的暴跌原因、消息是否破坏长期逻辑、估值、量化业务权重、未来 30/60 日催化与下降旗形风险。

先手动检查前 3–5 次输出，再开启无人值守任务。正式用资金前，应增加两年回测，至少比较：20 日内先涨 8% 的概率、20 日收益中位数、最大不利波动，以及相对“所有同期跌 18% 的股票”的增量优势。

## GitHub Actions

仓库 Secret 中添加 `MASSIVE_API_KEY`，复制 `.github/workflows/daily-screen.yml`。任务每天 23:15 UTC 执行；全年都晚于美股常规交易收盘，并自动跳过没有新交易日数据的周末/假日。报告保存在 workflow artifact 中。不要把真实 key 写入 `.env.example`、workflow YAML、源码或 Git 历史。

如果希望 ChatGPT 的定时任务继续做第二层研究，可以让 GitHub Actions 把 `latest_candidates.json` 保存到固定分支/对象存储，再让任务读取该文件。不要把 API key 写入仓库或聊天。

`daily_research_prompt.md` 已包含第二层研究的固定提示词，重点防止把漂亮形态误当成上涨概率，并保留“业务权重必须量化”的要求。
