# 新闻源评估 v1（2026-10-07 ~ 10-08）

目的：为 `stock-data-backfill` 的新闻层挑选数据源。要求是**事实类**（公司通稿、监管文件、交易所公告、通讯社事实报道），并且能在**一手发布后 30 分钟内**取到。

验证案例：Google 能源合作
- Fervo（FRVO）：GlobeNewswire，2026-09-01 08:00 ET
- Constellation（CEG）：Business Wire，2026-10-06 06:30 ET；Bloomberg 前一晚约 21:12 ET 先爆料；未交 8-K
- Black Hills（BKH）：GlobeNewswire，2026-10-06 16:10 ET，收盘后发布

## 结论摘要
- 实测满足"事实 + 30 分钟内"的免费源（2026-10-07 美股盘中抓了 35 分钟）：
  - SEC EDGAR 实时 8-K 列表：5 分钟内出现
  - PR Newswire RSS：发布后 2 到 7 分钟出现，有反爬，要加重试
  - Business Wire 行业 RSS：2 到 13 分钟出现
- GlobeNewswire 是一手来源，但 box 和本机网络都连不上，没测成。
- EDGAR 只能做官方确认：CEG 那笔没交 8-K，BKH 的 8-K 晚了 4 小时。
- 富途新闻搜索：已收录的稿子几分钟内就能搜到，但只显示日期，混有观点文章，只适合做提醒。
- 以下来源不达标：
  - Google News RSS：平均晚约 2 小时，链接是跳转地址
  - Massive 自带新闻：每小时更新
  - Finnhub：实时新闻稿只给企业版
  - agy（Gemini）：漏报、时间错误
- 付费备选（均未实测）：Massive + Benzinga（$99/月，需确认含新闻稿频道）、RTPR（$139/月，覆盖 BW/PRN/GNW，自称 500ms）、Benzinga 直连（需询价）。
- 通稿是发稿方付费发布的，内容真实性需要交叉核对（公司官网、8-K、第二个一手来源），按 A/B/C 三级处理，见 `reports/wire_authenticity.md`。

## 目录
- `reports/`：评估报告
  - `realtime_eval.md`：实时性实测
  - `vendor_eval.md`：Finnhub、Massive 等供应商对比
  - `wire_pricing.md`：BW 和 GNW 收费与使用条款
  - `wire_authenticity.md`：通稿真实性、假稿案例和分级过滤规则
- `data/`：汇总数据
  - `deals.json`：三笔交易汇总
  - `source_scorecard.csv`：新闻源评分表
  - `daily_ohlcv.csv`：相关股票日线
- `raw/box/`：在 box 上抓取的原始证据（EDGAR、Nasdaq、Google News、X、agy、各 RSS 快照）
- `raw/futu/`：本机通过 OpenD 做的富途新闻搜索和延迟跟踪输出
- `scripts/`：本机一次性探针脚本，输出写到当前工作目录
  - `nsc_poller.py`：RSS 轮询
  - `nsc_futu_news.py`：富途新闻搜索
  - `nsc_futu_lag.py`：富途收录延迟跟踪

## 注意
- `raw/` 是原始抓取数据，约 18MB，建议不入 Git。
- 结论只是 2026-10-07/08 的一次快照，不代表长期 SLA。
