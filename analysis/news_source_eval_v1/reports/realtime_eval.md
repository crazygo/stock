
# 实时性 + 事实性评估（Task 3）· 2026-10-07 实测
时间均为北京时间（BJT, UTC+8），括号内附 ET。实测窗口：2026-10-07 22:11–22:46 BJT（10:11–10:46 ET，美股盘中）。
判定标准：事实性 = 一手来源（公司稿/监管披露/交易所公告/通讯社事实报道）；快 = 一手发布后 ≤30 分钟可拿到。
方法：box 上 `realtime/poller.py` 每 5 分钟抓一次，记录每条首次出现时间；滞后上界 = 首次看到 − pubDate，下界 = 上一次成功抓取 − pubDate。原始快照在 `realtime/snap/`、`realtime/bw/snap/`、`realtime/nf/snap/`；汇总在 `realtime/lag_summary.txt`、`first_seen.jsonl`。

## 1. 结论表
| 来源 | 事实性 | ≤30min | 证据（实测/文档） | 成本/账号 | 备注 |
|---|---|---|---|---|---|
| SEC EDGAR getcurrent Atom（8-K/6-K/全部） | ✅ 监管 | ✅ 入库后 ≤5.3 min | 112 条新增，滞后上界中位 2.9、最大 5.3 min（受 5 分钟轮询限制）。`edgar_all`/`edgar_8k` | 免费，需声明 UA | 公司**提交** 8-K 本身可能晚：FRVO 稿后 +5 min，BKH 稿后 +4h09m，CEG **没交 8-K** |
| PR Newswire RSS | ✅ 公司稿 | ✅ 约 1.5–7 min | 连续两次成功抓取之间：发布 22:35 的条目在 22:36 没出现、22:41 出现了 → 滞后 1.6–6.7 min。21 条新增，上界中位 6.6 min | 免费，UA 用 `curl/8` | **8 次里有 4 次返回 HTML**（反爬），需要重试；全量 feed 只有 20 条，约覆盖 20 分钟，轮询需 ≤5 min；RSS 里没有 ticker |
| Business Wire 行业 RSS | ✅ 公司稿 | ✅ 2–13 min | 22:30 发布的条目：tech feed 滞后 7–13 min，networks feed 2–8 min。CDN 头 `s-maxage=300` | 免费 | **通用全量 feed 已停用**；没找到能源/公用事业 key，所以 CEG 这类 BW 独家稿目前**覆盖不到** |
| GlobeNewswire RSS（上市公司/行业/机构） | ✅ 公司稿，带 `NYSE:ACEL` ticker + ISIN | ⚠️ 未测出 | box 和 Mac 的 curl 都 TLS 失败（000）；WebFetch 两次返回同一份缓存（lastBuildDate 22:00 BJT 不变），测不了滞后 | 免费 | 历史时间戳到分钟：FRVO 09-01 20:00 BJT（08:00 ET），BKH 10-07 04:10 BJT（10-06 16:10 ET）。要在**能直连 globenewswire.com 的机器**上轮询 |
| TMX Newsfile 行业 RSS | ✅ | ⚠️ 未观察到新条目 | `feeds.newsfilecorp.com/industry/alternative-energy`，每个行业最近 10 条，时间戳到分钟；有间歇性非 feed 返回 | 免费 | 偏加拿大小盘 |
| Accesswire | ✅ | ❌ 无免费 RSS | rssfeed.aspx：“RSS 请联系分发团队” | 需申请 | — |
| Google News RSS | ❌ 聚合 | ❌ | 通讯稿条目在 Google News 出现时已发布 100–1150 min（24 条，中位约 121 min）；链接是 news.google.com 跳转 | 免费 | 只适合做发现/补漏 |
| Nasdaq Trader 停牌 RSS | ✅ 交易所 | – | pubDate 只到日期（04:00Z），停牌时间在正文里 | 免费 | 是停牌信号，不是新闻 |
| **富途 OpenAPI get_search_news**（Ray 的 OpenD 10.10.7008 已在跑，SDK 10.11） | ⚠️ 混合：GlobeNewswire / PR Newswire / 道琼斯 / MT Newswires / Benzinga / 美股SEC公告 + 观点类（Seeking Alpha / TipRanks / 智通 / 格隆汇） | ✅ 已收录的条目快 | IBM 的 PRN 稿 22:18 发布，22:21:36 搜到（≤3.6 min）；另一条 22:31 发布，22:37:44 搜到；无 ticker 的 PRN 软文大多不收录（14 条里收录 2 条）。ACEL 的 GNW 稿 22:00 发布，22:20 前已收录 | 有富途账号即可，限 10 次/30 秒 | **publish_time 只有“10/7”这种日期（按北京日期）**，没有分钟；URL 多为 news.futunn.com，标题常被机翻成中文；只能按关键词搜。适合做“快提醒 + 按 source 过滤”，原文和时间戳要回到 wire/EDGAR 取 |
| Alpaca News API（免费 Basic 账号） | ⚠️ Benzinga 编辑稿为主 | ✅ 文档称 websocket 实时 | 官方文档：“All news data is currently provided directly by Benzinga”，“average of 130+ news articles per day”，`wss://stream.data.alpaca.markets/v1beta1/news`（docs.alpaca.markets/us/docs/streaming-real-time-news、/historical-news-data）。免费 200 次/分钟（blog，beta 期说法） | 免费注册 | 每天约 130 条 → 不是全量通讯稿；**未实测**（没账号） |
| Benzinga 直连（新闻 + Press Releases API + WS） | ✅ press-release 频道（例子 author=GlobeNewswire，stocks 含 ISIN/CUSIP） | ✅ 文档称实时 WS | benzinga.mintlify.app …/press-releases；docs.benzinga.com/ws-reference | 价格**未公开**，需询价 | 也混有 Benzinga 自家评论，要按 channel 过滤 |
| Massive + Benzinga 附加 | 同上 | ✅ “Updated in real time” | 见 vendor_eval.md | $99/月 | Massive 自带新闻“Updated hourly” → ❌ |
| Finnhub | ⚠️ | ❌/未证实 | press-release 实时 WS 只在 Enterprise；GitHub #574 报告 news WS 推送旧闻 | 免费版延迟未说明 | — |
| MT Newswires | ✅ 通讯社 | 未测 | 没有公开价格；在富途结果里能看到（中文） | 询价 | 通过富途间接可得 |

## 2. 三个历史事件在各源里的样子（BJT）
| 事件 | 一手稿 | EDGAR | Google News | 富途搜索 |
|---|---|---|---|---|
| FRVO × Google | GNW 2026-09-01 20:00 BJT（08:00 ET） | 8-K 20:05:12 BJT（+5 min） | – | 9/1：GlobeNewswire 原稿 / MT Newswires / WSJ / 财联社 / 快讯（只有日期） |
| CEG × Google | Business Wire 2026-10-06 18:30 BJT（06:30 ET）；Bloomberg 泄露在 09:12 BJT（X 上最早一条 01:12:44Z） | **无 8-K** | 条目时间 18:30 BJT（与原稿同一分钟，但实际何时出现未知） | 10/7：道琼斯“Top Energy Headlines”、Utilities Roundup、智通、格隆汇 |
| BKH × Google | GNW 2026-10-07 04:10 BJT（10-06 16:10 ET） | 8-K 2026-10-07 08:19 BJT（+4h09m） | – | 10/7：Benzinga 摘要、Seeking Alpha、美股SEC公告 8-K、MT Newswires 目标价 |

## 3. 覆盖缺口
- 只走 Business Wire 的发行人（如 CEG）：BW 全量 feed 已停用，能源/公用事业 key 未知。CEG 又不交 8-K，所以免费栈目前**只能靠富途/Google News 才看到**。
- 8-K 晚交或不交：EDGAR 不能代替新闻稿（BKH 晚 4h，CEG 没交）。
- GlobeNewswire 从 box 和 Mac 的网络都连不上（TLS 失败），需要海外 VPS 或修好代理。
- PR Newswire RSS 有 50% 的请求返回 HTML（反爬），要重试，抓取间隔要 ≤3–5 min。
- 富途：没有分钟级时间戳，也没有原文 URL。

## 4. 推荐最小栈
1. **免费主干（全部达到事实 + ≤30 min）**：EDGAR getcurrent（8-K/6-K，1–2 min 轮询）+ data.sec.gov submissions（按 CIK 跟踪关注列表）；PRN RSS（2 min 轮询加重试）；BW 行业 RSS（已有 tech/health/M&A/networks，补能源/公用事业）；GNW RSS（Public Companies + 按机构/行业，部署在能直连的主机上）。
2. **富途 OpenD 做快速提醒和兜底**：按关注 ticker 或公司名关键词搜索，只收 source ∈ {GlobeNewswire, PR Newswire, Business Wire, 美股SEC公告, 道琼斯, MT Newswires} 的条目，再回 wire 或 EDGAR 补原文 URL 和分钟级时间。
3. **（可选付费）** 要一条流同时覆盖三家 wire、带 ticker 和实时 WS：Massive + Benzinga 附加 $99/月，或找 Benzinga 直连询价。Alpaca 免费，但每天约 130 条 Benzinga 稿，不算一手全量。

## 5. Ray 需要做的
- 决定 GNW 抓取放在哪：海外 VPS 或代理白名单。box 和 Mac 现在都连不上 globenewswire.com。
- 在浏览器里用 Business Wire 的 feed 定制页（businesswire.com/help/feed-options）生成“Energy / Utilities”RSS key。box 访问该站点返回 403。
- 富途不用额外操作（OpenD 在跑，get_search_news 已可用）。是否允许长期轮询（限 10 次/30 秒）需要 Ray 确认。
- 可选：注册免费 Alpaca 账号测 Benzinga WS；或付 $99/月 开 Massive Benzinga。
