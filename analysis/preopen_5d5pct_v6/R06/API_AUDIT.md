# 富途接口可行性结案

所有本轮相关接口已实际调用。四类历史字段已加入固定训练矩阵；其余逐项排除。没有待批准采购、待登录步骤或未实施训练协议。未证明的历史原始版本作为明确限制保留，不能改称真实PIT。

| 数据族 | 返回股票/请求 | 行数 | 实际范围 |
|---|---:|---:|---|
| capital | 124/124 | 30168 | 2025-10-06–2026-09-30 |
| options | 123/124 | 59674 | 2024-10-04–2026-09-30 |
| volatility | 123/124 | 59638 | 2024-10-04–2026-09-30 |
| short_volume | 124/124 | 59957 | 2024-10-04–2026-09-30 |

| 接口 | 实际调用结果 | 结案 |
|---|---|---|
| get_option_underlying_his_statistic | ret=0 | 分两个≤364日窗口并翻页，取得约两年；OI不前移，使用记录日后2/5官方日延迟；未提供历史修订版本，不认证PIT。 |
| get_option_underlying_his_volatility | ret=0 | 历史日级IV/HV已分窗翻页取得；2/5日延迟开发假设；不使用财报后未来IVcrush。 |
| get_daily_short_volume | ret=0 | 每页50行、next_key翻页取得两年；卖空成交量不是净持仓或借券量。 |
| get_short_interest | ret=0 | 结算日不等于可知公布日；没有逐条原始发布时间，不能把结算日当实时特征。 |
| get_financials_earnings_price_history | ret=0 | 包含公布后价格路径、预测波动与未来IVcrush；公布时间存在00:00/None；无历史原始预期版本。普通OHLC也不形成新信息。 |
| get_financials_earnings_price_move | ret=0 | 分析含财报前后day_offset及未来路径与IV，不能作为当时输入。 |
| get_financials_statements | ret=0 | 季度日期与当前结构化数字可得，但原始公布/接收/修订版本不完整，不能冒充当时实际值。保留原SEC accession路线。 |
| get_research_analyst_consensus | ret=0 | 实际为当前目标价与评级汇总，不是历史财报EPS/营收共识。 |
| get_research_rating_summary | ret=0 | 当前机构评级资料，原始接收、历史修订以及两年完整覆盖未由响应证明。 |
| get_search_news | ret=0 | 仅最近条目、公布时间及URL；所调用接口没有两年begin/end检索，不能补成历史新闻面板。 |
| get_daily_short_volume | ret=0 | 每页50行、next_key翻页取得两年；卖空成交量不是净持仓或借券量。 |
| get_option_underlying_his_statistic | ret=0 | 分两个≤364日窗口并翻页，取得约两年；OI不前移，使用记录日后2/5官方日延迟；未提供历史修订版本，不认证PIT。 |
| get_option_underlying_his_volatility | ret=0 | 历史日级IV/HV已分窗翻页取得；2/5日延迟开发假设；不使用财报后未来IVcrush。 |
| get_capital_distribution | ret=0 | 仅当前资金分布快照，无两年当时版本。 |
| get_capital_flow | ret=0 | DAY实取最近一年；INTRADAY传五月日期却返回十月2日，不具有所请求的历史盘中数据。日级用真实同日RTH成交额，缺分母保留空缺。 |
| get_market_snapshot | ret=0 | 实时/最新快照含当前基本面字段，不可回填历史时点。 |
| get_order_book | ret=-1 | 实际订阅取得5档，但服务端时间字段为空；无两年历史深度，不进入训练或成交有效性结论。 |
| get_rt_ticker | ret=-1 | 订阅取得20条，但全为2026-10-02旧逐笔；不当作10月5日实时数据或两年回测。 |
| get_owner_plate | ret=0 | 当前板块归属不是历史成员版本；不改冻结股票池。 |
| get_corporate_actions_dividends | ret=0 | 分红与拆股信息属标签质量审计；本轮保留原公司行动标签，不把重复公司行动当预测信息。 |
| get_corporate_actions_stock_splits | ret=0 | 公司行动仅验证已冻结标签口径，不更改原结果。 |
| get_corporate_actions_buybacks | ret=-1 | 实测美股不支持，不能作为美股历史回购特征。 |
| get_insider_trade_list | ret=0 | 交易日期区间不证明公开披露日；没有可核对原受理时间，不用于事件惊喜。 |
| get_shareholders_institutional | ret=0 | 季度持仓与update_time不证明首次披露及历史版本。 |
| get_macro_indicator_list | ret=0 | 宏观指标目录可得，本身不是可回测信息增量。 |
| get_economic_calendar | ret=0 | 日历有previous/consensus/actual，但未提供原始预期与实际修订版本。 |
| get_earnings_calendar | ret=-1 | 本次历史窗口实测NN_ProtoRet_TimeOut；既有缓存也不证明当时预期可知，未将其作为历史共识。 |
| get_macro_indicator_history | ret=0 | 100行含release_time/value/predict_value/previous_value，也包含未来尚未公布的N/A行；没有原始修订历史，不做as-released惊喜。 |
| get_option_volatility | ret=-1 | 传入AMD股票代码返回只支持期权代码；本轮所需历史标的IV/HV由专用历史接口取得，不留下依赖。 |

盘口单独订阅验证成功，并在满足最低订阅时长后取消本连接订阅。五档Bid/Ask的服务端时间为空，20条逐笔时间全为十月2日，不能认证当下可买；也没有历史盘口。

工程修复：AXTI等响应将数值与N/A混用，首次Parquet序列化失败；保留失败日志，原响应JSON归档后将N/A转为缺失，再有限断点重跑。缺失没有填零或删除股票。资金流归一化只用与冻结OHLCV一致的完整RTH真实turnover，缺失不估算。

方法与权限以实际OpenD返回为准。官方说明：[资金流](https://openapi.futunn.com/futu-api-doc/quote/get-capital-flow.html)、[期权历史统计](https://openapi.futunn.com/futu-api-doc/quote/get-option-underlying-his-statistic.html)、[每日卖空成交](https://openapi.futunn.com/futu-api-doc/quote/get-daily-short-volume.html)。资金流文档有一年/两年表述差异，本轮只报告实取一年；期权统计按≤364日窗口翻页，OI注明T-1仍不回移。

IBKR开户不会自动修复历史共识/修订版本与历史可成交报价。本轮未访问IBKR账户，未下单、未购买权限、未上传R2。
