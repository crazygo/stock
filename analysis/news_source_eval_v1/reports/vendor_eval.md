# Finnhub vs Massive（原 Polygon）新闻/行情供应商评估 — 2026-10-07（北京时间）

## 范围与方法
- 只用官方文档、定价页（含页面内嵌 JSON）、官方知识库，以及少量独立来源。没有注册任何账号。
- 两家都没有不注册就能用的 demo key：不带 key 调 Finnhub `/company-news`、`/press-releases` 和 Massive `/v2/reference/news` 都返回 HTTP 401（"Please use an API key." / "API Key was not provided"）。**所以这三个事件（FRVO、CEG、BKH）在两家都还没测。**
- EODHD 有公开的 `demo` key，但只开放 6 个代码。CEG.US 返回 403。拿 AAPL.US 取样：9/1–10/7 的 200 条新闻里，183 条链接是 finance.yahoo.com，16 条 seekingalpha.com，1 条 cnbc.com（证据：vendor/eodhd_demo_AAPL_200.json）。
- 原始页面存在 vendor/ 目录。

## Finnhub
| 项目 | 结论 | 来源 |
|---|---|---|
| Company News（按代码查） | 只覆盖北美公司；免费版有 1 年历史加新数据；字段 `datetime`（UNIX 秒）、`url`（文档写"原文链接"）、`related`、`source` | https://finnhub.io/docs/api/company-news |
| 新闻来源 | 文档示例里 source 为 "Yahoo"、url 是 finance.yahoo.com（转载），说明有相当部分不是原文链接；没有公开来源清单 | 同上（News websocket 示例） |
| Major Press Releases（Premium） | "sourced from the exchanges, BusinessWire, AccessWire, GlobeNewswire, Newsfile, and PRNewswire"；`datetime` 格式为 `YYYY-MM-DD HH:MM:SS`，**没写时区**；全文仅限 Enterprise | https://finnhub.io/docs/api/company-news（Press Releases 一节） |
| 实时推送 | News websocket 要 Premium；Press Release websocket 仅限 Enterprise | 同上 |
| 已知问题 | GitHub issue #574（2026-02）：用户有 Fundamentals-1 + US Market Data，websocket 新闻只收到旧数据，REST 正常 | https://github.com/finnhubio/Finnhub-API/issues/574 |
| 价格（官方定价页内嵌 serverData） | Fundamental-1 $50/月/每个市场（**按季付**，$150）；Fundamental-2 $200/月（按年付）；Market Data Basic $49.99（按季）、Standard $129.99（按季）、Professional $199.99（按年）；All-In-One $3,500 | https://finnhub.io/pricing （HTML 内 `window.serverData.price`） |
| Fundamental-1 包含 | Company News 3 年加实时、Press Releases ✓、SEC Filings ✓、News Sentiment ✓；300 次/分钟 | 定价页 JS 对比表（vendor/finnhub_main.js） |
| Market Data Basic 包含 | 美股日线 10 年、1 分钟 K 线 10 年、Tick 5 年；150 次/分钟 | 同上 |
| 免费版 | 60 次/分钟；没有 K 线（Candles 标 Premium）；SEC Filings 免费 | 定价页 JS 和 docs |
| 许可 | 所有自助套餐均为 Personal Use | 定价页 JS（"Personal Use. Terms apply"） |
| 限流 | 所有套餐之上另有 30 次/秒的硬上限 | https://finnhub.io/docs/api （Rate Limits） |

## Massive（原 Polygon.io）
| 项目 | 结论 | 来源 |
|---|---|---|
| 新闻接口 | `GET /v2/reference/news`，可按 ticker 和 published_utc 过滤；字段 `published_utc`（RFC3339，精确到秒，UTC）、`article_url`、`publisher`、`tickers`、`insights`（情绪） | https://massive.com/docs/rest/stocks/news.md |
| 套餐/时效/历史 | **所有 Stocks 套餐都包含，免费 Basic 也有**；**各套餐都是"Updated hourly"**；历史 Basic 2 年，Starter 及以上全部（从 2016-06-22 起） | 同上 |
| 新闻来源 | 聚合多家出版方；官方抽样 50 条看到 GlobeNewswire、The Motley Fool、Zacks。打 ticker 标签的规则是"文中提到"，不一定是"讲这家公司"，官方举了 AAPL 被误标的例子 | https://massive.com/knowledge-base/article/where-does-massive-source-ticker-news-from |
| Benzinga 实时新闻（合作数据） | `GET /benzinga/v2/news`，"Updated in real time"，历史从 2009-04-27 起，有全文、tickers、channels；价格 $99/月/每个数据集 | https://massive.com/docs/rest/partners/benzinga/news.md ；https://massive.com/pricing |
| Benzinga 新闻稿覆盖（Benzinga 自家 API） | 合作通稿渠道：ACCESSWIRE、Business Wire、GlobeNewswire、PRNewswire、Newsfile。**但 Massive 转售的 Benzinga News 是否包含 Press Releases channel，没有核实** | https://www.benzinga.com/apis/cloud-product/press-releases/ |
| 行情套餐 | Basic $0：5 次/分钟、2 年历史、日终数据、分钟聚合、公司行动；Starter $29：不限次数、5 年、延迟 15 分钟、Flat Files、WebSocket；Developer $79：10 年加逐笔成交；Advanced $199：实时、20 年以上、报价、财务 | https://massive.com/pricing |
| 分钟/日线 K 线 | Custom Bars 所有 Stocks 套餐都有；Basic 日终、Starter 延迟 15 分钟；历史 Basic 2 年、Starter 5 年 | https://massive.com/docs/rest/stocks/aggregates/custom-bars.md |
| 其他打包数据 | 拆股、分红、8-K Text、8-K Disclosures、SEC EDGAR Index、Form 3/4、10-K Sections、13-F（这些接口分别在哪个套餐里，**没有逐个核实**） | https://massive.com/docs/llms.txt |
| 许可 | 个人套餐仅限个人使用、非专业用户；商用 Stocks Business $2,499/月（第三方引用） | https://massive.com/pricing ；https://www.moneyflock.com/contents/articles/free-stock-api-commercial-use-licensing |

## 备选
| 供应商 | 事实 | 来源 | 结论 |
|---|---|---|---|
| Tiingo | 个人 $30/月；News API 可查 3 个月历史加之后的新数据（更多历史需商用） | https://www.tiingo.com/about/pricing | 历史太短，不适合回补 |
| EODHD | News API 有 demo key；实测链接 92% 是 Yahoo 转载 | https://eodhd.com/financial-apis/stock-market-financial-news-api ；vendor/eodhd_demo_AAPL_200.json | 不满足原文链接的要求 |
| Benzinga 直连 | 新闻稿来自 5 家主要通稿渠道，REST 加 TCP 推送 | https://www.benzinga.com/apis/cloud-product/press-releases/ | 公开页面没找到价格；通过 Massive 买是 $99/月 |

## 建议
1. **先免费验证，再付费。** 请 Ray 自己注册 Massive Basic（免费）和 Finnhub Free 两个 key。Massive Basic 有新闻（2 年）和分钟 K 线（2 年），5 次/分钟足够拿这三个事件做实测，同时补上 FRVO 9/1、CEG 10/6、BKH 10/6–7 的分钟线。
2. **验证通过就付 Massive Stocks Starter，$29/月。** 它能补上：分钟线和日线（不限次数、5 年、Flat Files），解决 Yahoo 429 和 R2 缺口；按 ticker 查新闻，带 UTC 秒级时间和 publisher，历史从 2016 年起，适合回补（Google News RSS 的老新闻只剩日期）；公司行动。
3. **只有需要分钟级的新闻稿时效时**，再加 Massive 的 Benzinga News（$99/月），合计约 $128/月。前提是先确认它包含 Press Releases channel。
4. **Finnhub 只作为第二选择。** 新闻稿要 Fundamental-1（$50/月，按季付 $150），K 线要再加 Market Data Basic（$49.99/月，按季付），合计约 $100/月，而且都是个人许可；press release 的时间没写时区，company news 有不少是 Yahoo 转载链接。
5. **哪个都替代不了现有的免费层：** EDGAR（官方确认）和 GlobeNewswire 通稿（官方发布时间）还要保留。Bloomberg 爆料这类传闻，两家新闻源大概率不会以原文形式收录（没核实），仍然要靠 X 和 Google News RSS 抓。"首次取得时间"任何供应商都给不了，必须自己记录。

## 有 key 之后要实测的
- 三个事件里，Massive `/v2/reference/news` 和 Finnhub `/company-news`、`/press-releases` 能不能查到，`published_utc`/`datetime` 和官方时间（FRVO 08:00 ET、CEG 06:30 ET、BKH 16:10 ET）差多少。
- 链接是不是原文（GlobeNewswire / Business Wire），还是 Yahoo、Benzinga 转载。
- 有没有收录 CEG 的 Bloomberg 爆料（10-05 约 21:12 ET），或者转述它的报道。
- 新闻"Updated hourly"的实际延迟。
- Finnhub press-release 时间用的是什么时区。
- Massive 分钟线是否含盘前盘后（用来定位 CEG 06:30 发布后的盘前反应）。
- Massive 8-K 接口在哪个套餐里。
