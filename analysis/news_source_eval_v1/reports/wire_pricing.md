# Business Wire / GlobeNewswire 收费结构（2026-10-07 查证，北京时间 23:3x）
原则：只写官方页面能看到的；第三方估价会标“非官方”；查不到就写“未公开”。

## A. 作为读者 / 数据使用方
| 项目 | Business Wire (BW) | GlobeNewswire (GNW, Notified 旗下) |
|---|---|---|
| 公开 RSS | 免费，有预设行业 feed，可按 250+ 关键词组合定制。RSS 只有标题 + 链回 businesswire.com — https://www.businesswire.com/help/feed-options | 免费，按 主题/行业/国家/州 + Public Companies、Earnings、M&A 分类，提供 RSS/ATOM — https://www.globenewswire.com/rss/list |
| 邮件提醒 | PressPass 账号：按公司/主题/行业/语言订阅，邮件或 newsfeed 推送。官方页没写收费（第三方说免费）— https://www.businesswire.com/media-journalist-tools | Reader Account：可设提醒、订阅公司新闻、保存搜索，注册免费 — https://www.globenewswire.com/home/learning-support/reader-support ；https://pnrlogin.globenewswire.com/en/Register |
| 付费机读全文 feed | “Newsfeed Licensing”：全文 Atom（可存进自己的系统）、NX / FTP / SFTP 推送 NewsML+XHTML，按主题/行业/语言/地区定制 — https://www.businesswire.com/help/feed-options ；https://www.businesswire.com/schema/xhtml 。**价格未公开**（只有 “Interested in licensing…” 联系入口） | 媒体合作伙伴 / 定制 feed：格式 ANPA / NITF / NewsML / RSS / ATOM，按主题/地区/行业/交易所定制 — https://www.globenewswire.com/newswire-press-release-content ；https://www.globenewswire.com/home/about/media-relations （联系 media@globenewswire.com）。**价格未公开** |
| 对高频交易 | 2014 年起**不再**向 HFT 公司直接授权 feed — https://www.businesswire.com/news/home/20140220006506/en/ | 未见类似声明 |
| 通过第三方分发 | 官方说经 NX 网络把稿件同时推给媒体和市场参与者（同上 2014 声明）；BW 官方页没列出具体终端名单 | 官方列出的合作方：Bloomberg、Dow Jones Newswires、Factiva、Moody's NewsEdge、COMTEX、FinancialContent、Yahoo Finance、AP、MarketWatch 等 — https://www.globenewswire.com/newswire-press-release-content |
| 个人自动抓取 / 再分发条款 | 官方 Terms of Use：网站用途**限于**提交稿件、“retrieving RSS feeds”、阅读；**禁止**“store, aggregate, reproduce, or distribute information… or compete”，禁止未经书面同意的商业活动，“All scans of internet-facing websites are prohibited” — https://www.businesswire.com/legal/terms-of-use 。→ 个人拉 RSS 是明确允许的；抓网页、建库、再分发不允许 | globenewswire.com 上**没找到**官方条款页（/Home/TermsOfUse 返回 404）。第三方托管的一份 “GlobeNewswire Terms of Use” 写的是 “personal, non commercial use only”，禁止 reproduce / distribute / store — https://editorial.contentenginellc.com/globe-newswire/terms-of-use.pdf （**不是官方域名，仅供参考**）。官方内容页说记者和博主可以用 Reader Account / RSS，商用要联系 |

**对 Ray 的含义**：
- 个人研究用：低频拉 RSS、本地存元数据（标题、时间、URL、ticker）、不对外分发，这是最稳妥的用法。BW 条款明确允许拉 RSS，但禁止“store/aggregate”网站内容，所以别抓 businesswire.com 网页全文入库。
- 想要全文、带 ticker、推送：只能买授权（两家都要询价），或者走第三方（见下面 C）。

## B. 作为发稿方（仅供背景）
| | Business Wire | GlobeNewswire |
|---|---|---|
| 官方公开价 | **$475 起**：400 词，美国本地（local）分发 — https://www.businesswire.com/pricing 。其余（全国、加字、多媒体、翻译）按报价 | **没有公开价目表**，按报价 |
| 第三方估计（非官方） | 美国全国 400 词约 $760–$940，每加 100 词 $195，第一个附件 $425 — reporteroutreach.com / prezly.com / presspilot.io 2026 文章 | 基础约 $350；北美标准 600 词带图约 $900–$1,200 — prezly.com / pressonify.ai / reporteroutreach.com（互相引用，可信度一般） |

## C. 便宜的第三方实时渠道（复述 + 本次补充）
| 渠道 | 覆盖哪些 wire | 实时性 | 价格 | 来源 |
|---|---|---|---|---|
| Benzinga Press Releases API（直连） | ACCESSWIRE、Business Wire、GlobeNewswire、PRNewswire、Newsfile；API 拉取或 TCP 推送；可展示 | 文档称实时（WS / TCP） | **未公开**（官网 “Get Started” / 询价） | https://www.benzinga.com/apis/cloud-product/press-releases/ ；https://benzinga.mintlify.app/api-reference/news-api/press-releases/get-press-releases |
| Massive + Benzinga News | Benzinga 新闻；**是否包含 Press Releases 频道仍未核实** | “Updated in real time” | $99/月/每个数据集 | https://massive.com/docs/rest/partners/benzinga/news.md ；https://massive.com/pricing |
| Alpaca News（免费账号） | Benzinga 提供，约 130+ 条/天，**不是全量通讯稿** | WS 实时 | 免费（Basic） | https://docs.alpaca.markets/us/docs/historical-news-data |
| Finnhub | 官方说 press release 来源包括交易所、BusinessWire、AccessWire、GlobeNewswire、Newsfile、PRNewswire | press-release 实时 WS 和全文**只在 Enterprise** | Fundamental-1 $50/月（按季付 $150）含 Press Releases | https://finnhub.io/docs/api/company-news ；https://finnhub.io/pricing |
| 富途 OpenAPI（已有） | 搜索结果里能看到 GlobeNewswire / PR Newswire / 道琼斯 / MT Newswires / Benzinga | 已收录的条目 3–7 min 内可搜到（本日实测） | 免费（富途账号） | 见 realtime_eval.md |
| Bloomberg / Refinitiv / Dow Jones / Comtex / NewsEdge | GNW 官方列为合作方 | 实时 | 机构级，**未公开**（不便宜） | GNW 合作方页 |
