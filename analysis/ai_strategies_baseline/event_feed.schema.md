# 势能传导策略的历史事件输入

`backtest_ai_strategies.py --event-feed PATH` 接受下列 JSON 结构。`complete_coverage` 只有在数据提供方能说明 2026-05-01 之后相关股票的新闻、订单、财报预期与负面反证被完整采集，且保留每次首次公开的时间和当时版本时才能设为 `true`。今天回填的新闻搜索列表不满足这个条件。

```json
{
  "complete_coverage": true,
  "events": [
    {
      "ticker": "NVDA",
      "published_at_utc": "2026-05-04T14:30:00Z",
      "source_url": "https://example.org/original-source",
      "evidence": 0.8,
      "transmission": 0.7,
      "persistence": 0.6,
      "priced_in": 0.2,
      "contradiction": 0.1
    }
  ]
}
```

五个分数均在 0–1 之间，并且必须只依据 `published_at_utc` 时已公开的信息形成。基础版分数为 `evidence × transmission × persistence × (1 − priced_in) × (1 − contradiction)`，达到 0.25 时入选。美东 16:00 之后发布的事件只会从下一交易日收盘开始形成信号。事件到股票的关联由输入数据显式指定；代码不会从未来价格倒推事件。评估目标与策略 1 相同：30 自然日内先达到 +20%，且此前未先触及 -12%。

富途 [`get_search_news`](https://openapi.futunn.com/futu-api-doc/quote/get-search-news.html) 有 `publish_time`，但公开文档的请求参数只有关键词、返回条数和资讯子类型，返回最多 100 条，不能仅靠它证明整个历史期间的资讯覆盖和预期修订版本。富途 [`get_financials_statements`](https://openapi.futunn.com/futu-api-doc/quote/get-financials-statements.html) 可提供财报数据；要回测预期修订，仍需每次修订的历史快照和首次可见时间。
