# Business Wire 行业覆盖 + 新闻稿真实性（Task 5，2026-10-08 BJT）

> 标注：✅=一手页面已核实；🔸=仅搜索摘要/二手，未打开原页；❓=未能核实

## 1. BW 行业分类 & 是否以美股上市公司为主
- feed-options 页只说“250+ subject/industry keyword”，**不列清单** ✅ https://www.businesswire.com/help/feed-options ；媒体工具页：250+ 新闻关键词、数百地理关键词、20 种语言 ✅ https://www.businesswire.com/media-journalist-tools
- 分类细项（distribution-lists 页 WebFetch 500、box curl 不通）🔸：Energy 下有 Alternative Energy / Coal / Nuclear / Oil/Gas / Utilities / Other；Technology 下有 Semiconductor；Artificial Intelligence 为独立 subject。实例：Brookfield–Bloom 稿件标签 Alternative Energy, Energy, AI, Data Management, Technology 🔸 https://www.businesswire.com/news/home/20251013114760/en/
- 完整 250+ 清单 ❓（未拿到）。BW RSS item 本身**不带分类标签**（实测）。
- 美股占比（实测 2026-10-07 快照，导语出现交易所代码判定；描述截断可能低估）：

| BW feed | 条数 | 含美股代码 | 无代码 | 其他 |
|---|---|---|---|---|
| Technology | 220 | 54 (25%) | 163 | Euronext 2, OTC 1 |
| Health | 78 | 20 (26%) | 58 | – |
| M&A | 16 | 7 (44%) | 9 | – |

→ **BW 不是“美股上市公司为主”**：约 3/4 为私营/海外/营销类稿件。需按代码自行过滤（GNW 有 “Public Companies” 专门 feed 带 ticker/ISIN）。

## 2. 核验流程 & 监管地位
| | 发稿人身份核验 | 编辑审核 | 内容准确性 | 额外标记 |
|---|---|---|---|---|
| BW | “Verify Your Organization” 申请，“careful eye on each and every company” ✅ https://www.businesswire.com/en/sign-up/application | 编辑排版/查链接，客户确认后发；自称“journalists 与 fake news 之间的保护层” ✅ https://www.businesswire.com/blog/how-to-create-and-distribute-a-press-release-in-5-steps ；自称编辑“confirm legitimacy and accuracy”+SOC2 Type II（自我宣称）🔸 https://medium.com/@BusinessWire/why-newswires-the-past-present-future-of-trusted-news-f01aee014ff9 | 条款：BW 对站内内容不准确“assumes no responsibility” ✅ https://www.businesswire.com/legal/terms-of-use | **无公开“verified”徽章**（未找到）；2010 Javelin 假稿后停止邮件投稿、只走 BW Connect 🔸(DJN) |
| GNW | 身份核验+编辑审核 ✅ https://www.globenewswire.com/en/about/quality | 同上 | 发稿方负责 | **CLEAR Verified**（可选，自拍+政府ID，逐稿；美国→2025-11-06 加拿大）稿末文字 “published by a CLEAR® Verified individual” ✅ https://www.globenewswire.com/news-release/2025/11/06/3182648/0/en/ ；2021 Walmart 假稿后称加强认证 |
| PRN | 会员表单“so we can verify you as an authorized user” ✅ https://www.newswire.ca/account/online-membership-form/ ；多步认证 🔸 https://cision.atlassian.net/wiki/spaces/CSM/pages/25768198905 | 编辑审核不做事实核查 🔸 https://www.cision.com/legal/prw-terms/ | 发稿方负责 | 无公开徽章 ❓ |

- 监管：Reg FD 认可“通过广泛发行的新闻/通讯社发布新闻稿”为合规公开披露 ✅ https://www.sec.gov/rules-regulations/2000/08/selective-disclosure-insider-trading 。Nasdaq 5250(b)(1)：任何 Reg FD 合规方式即可，盘中 7:00–20:00 ET 重大消息需**提前≥10 分钟通知 Nasdaq MarketWatch** ✅ https://listingcenter.nasdaq.com/rulebook/nasdaq/rules/nasdaq-5200-series ；NYSE 也要求提前通知 ✅ NYSE 2025 Guidance Letter。**交易所不“认证”某家通讯社**，BW/GNW/PRN 只是常用合规渠道。
- 结论：三家都是“验身份（谁在发）”，**不验内容（说的是否真）**；准确性责任在发稿方。

## 3. 假稿/恶作剧事件
| 事件 | 渠道 | 经过 | 发现/撤回速度 | 来源 |
|---|---|---|---|---|
| Emulex 2000-08-25 | Internet Wire（前员工用假名邮件投稿） | 09:30 EDT 发出，10:13 新闻社转载，16 分钟内 $103.94→$43，蒸发 $2.2B | Emulex 否认后 Nasdaq 10:29 停牌，收盘 $105.75；Jakob 判 44 个月，返还 ~$353k + 罚款 $102,642 | ✅ https://www.sec.gov/enforcement-litigation/litigation-releases/lr-16671 ；lr-17079；lr-17094 |
| WCI 2008-07 | 传真给媒体（非通讯社） | 次周一股价上涨 | SEC 起诉 Karp，获利 ~$29k | ✅ https://www.sec.gov/news/press/2009/2009-118.htm |
| General Mills 2010-06 | PR Newswire | 假稿 | 美股正常交易前撤回 | 🔸 DJN https://br.advfn.com/noticias/DJN/2010/artigo/43314158 |
| Javelin 2010-06-18 | **Business Wire**（“stolen identity”） | 假稿；道琼斯/彭博/路透未转发 | 正常交易前撤回，移交 FBI | 🔸 同上 DJN |
| Tower Group 2014-05-13 | 新闻稿（渠道未明） | 假 Euroins 收购要约，股价 +32% | 后并入 Nedev 案 | ✅ https://www.carriermanagement.com/news/2014/05/13/122937.htm |
| Avon 2015-05-14 | **EDGAR**（假 PTG 要约） | 股价 +20% | Avon 当日称未收到要约；SEC 6-04 冻结 >$2M | ✅ https://www.sec.gov/news/press-release/2015-110 |
| Fitbit 2016-11-10 | **EDGAR**（假 ABM Capital SC TO-C） | 10:59 公开，至 11:10 $8.41→$9.28 | Fitbit 当日否认；SEC 2017-05-19 起诉 | ✅ https://www.sec.gov/newsroom/press-releases/2017-107 |
| Walmart/Litecoin 2021-09-13 | **GNW**（新开欺诈账户，域名 walmart-corp.com，账户首稿） | LTC +20–30%；CNBC、路透转载 | 11:18 ET 发 “Notice to Disregard”（约 1.5–2 h） | ✅ https://corporate.walmart.com/news/2021/09/13/walmart-statement-in-response-to-fake-litecoin-press-release ；https://www.coindesk.com/business/2021/09/13/how-media-fell-for-a-lucrative-lie-about-walmart |

- **更正前提**：“Fitbit 2015 GNW 假收购”不实：实为 **2016 年 EDGAR 假 SC TO-C**；Avon 也是 **EDGAR**，非通讯社。“Tesla 通讯社假稿”未找到 ❓；Kroger/比特币现金假稿仅见博客 ❓ https://www.swordandthescript.com/2021/11/fake-press-releases/
- 规律：冒名（邮件/新开账户）；联系人域名非发行人；发行人 IR 站无此稿；账户首稿；收购/要约/加密货币题材；EDGAR 上由**陌生第三方 CIK** 提交的要约文件。⇒ **EDGAR 本身也能被滥用**，确认必须要求“提交人 CIK = 发行人 CIK”（8-K/6-K），而非任意 EDGAR 文件。

## 4. 实用真实性过滤器 + 三事件回测
检查项（实测）：

| 检查 | FRVO（GNW 09-01 20:00 BJT） | CEG（BW 10-06 18:30 BJT） | BKH（GNW 10-07 04:10 BJT） |
|---|---|---|---|
| 交易所代码在电头 | ✅ (Nasdaq: FRVO) | ✗ 仅在 About 段 (Nasdaq: CEG) | ✅ (NYSE: BKH) |
| 发稿账户有历史稿 | ✅ | ❓（BW 页不通） | ✅（7-28、8-05） |
| 联系人域名=发行人 | ✗ 公关公司 v2comms.com | ✅ constellation.com | ✅ blackhillscorp.com（发稿主体为子公司名） |
| GNW CLEAR 徽章 | ✗ | n/a | ✗ |
| 发行人 CIK 8-K | ✅ +5 min (Ex 99.1) | ✗ 无 | ✅ +4h09m (7.01/9.01) |
| 发行人官网同稿 | ✅ fervoenergy.com（今日 curl 200） | ✅ constellationenergy.com/news/2026/10/…（今日核实） | ❓（box 访问不通） |
| 对手方官方页 | ❓ | ✅ googlecloudpresscorner.com + blog.google | ❓ |
| 撤稿/Notice to Disregard | 无 | 无 | 无 |

各规则通过率：
| 规则 | 通过 |
|---|---|
| 必须 CLEAR 徽章 | 0/3 |
| 电头必须有代码 | 2/3（CEG 挂） |
| 联系人域名=发行人 | 2/3（FRVO 挂） |
| 必须 8-K | 2/3（CEG 挂）；≤30 min 内仅 1/3（FRVO） |
| **代码（全文任意位置）映射 NYSE/Nasdaq 发行人 + ≥1 独立一手确认（发行人 CIK 8-K / 发行人官网 / 对手方官网）+ 无撤稿** | **3/3** |
| 同上且确认在 ≤30 min 内 | FRVO ✅；CEG 大概率（官网/Google 页同日，分钟级时间戳 ❓）；BKH ❓ |

CEG：无 8-K，但官网+Google 双方官方页 = 2 个独立一手确认，可信度不低于 8-K。

建议分级：
- **A 已确认**：通讯社稿 + 发行人 CIK 的 8-K/6-K Ex99，或发行人官网/IR 同标题，或对手方官方页（大厂 newsroom/blog）。
- **B 暂定（可推送，标“未独立确认”）**：NYSE/Nasdaq 代码映射正确；账户有历史稿；联系人为发行人域名或已知公关公司；有 CLEAR 更好；无撤稿。之后轮询 A 级确认并升级。
- **C 传闻/拦截**：无代码 / OTC / 小市值；账户首稿；陌生域名；陌生买方的收购/要约；加密货币题材；EDGAR 上第三方 CIK 的 SC TO-C；彭博等“知情人士”泄露（如 10-06 09:12 BJT 的 CEG 泄露），在一手确认前一律视为传闻。
- 持续监控：标题含 “Notice to Disregard / Correction / Retraction” 的稿件；Nasdaq 停牌代码 T1/T2（待发消息/消息发布中）；发行人官网 24h 内是否同步。
