"""Conclusive dispositions for every relevant provider probe performed here."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import OUT,write,now,sha
P=OUT/'R06'
DISPOSITION={
 'get_option_underlying_his_statistic':('development_input','分两个≤364日窗口并翻页，取得约两年；OI不前移，使用记录日后2/5官方日延迟；未提供历史修订版本，不认证PIT。'),
 'get_option_underlying_his_volatility':('development_input','历史日级IV/HV已分窗翻页取得；2/5日延迟开发假设；不使用财报后未来IVcrush。'),
 'get_daily_short_volume':('development_input','每页50行、next_key翻页取得两年；卖空成交量不是净持仓或借券量。'),
 'get_capital_flow':('development_daily_only','DAY实取最近一年；INTRADAY传五月日期却返回十月2日，不具有所请求的历史盘中数据。日级用真实同日RTH成交额，缺分母保留空缺。'),
 'get_short_interest':('excluded_publication_unknown','结算日不等于可知公布日；没有逐条原始发布时间，不能把结算日当实时特征。'),
 'get_earnings_price_history':('excluded_unsafe_or_duplicate','包含公布后价格路径、预测波动与未来IVcrush；公布时间存在00:00/None；无历史原始预期版本。普通OHLC也不形成新信息。'),
 'get_earnings_price_move':('excluded_future_path','分析含财报前后day_offset及未来路径与IV，不能作为当时输入。'),
 'get_financial_statements':('excluded_as_released_unproven','季度日期与当前结构化数字可得，但原始公布/接收/修订版本不完整，不能冒充当时实际值。保留原SEC accession路线。'),
 'get_stock_analyst_consensus':('excluded_current_summary','实际为当前目标价与评级汇总，不是历史财报EPS/营收共识。'),
 'get_stock_analyst_ratings':('excluded_historical_vintages_unproven','当前机构评级资料，原始接收、历史修订以及两年完整覆盖未由响应证明。'),
 'get_stock_news':('excluded_historical_search_unavailable','仅最近条目、公布时间及URL；所调用接口没有两年begin/end检索，不能补成历史新闻面板。'),
 'get_capital_distribution':('excluded_current_snapshot','仅当前资金分布快照，无两年当时版本。'),
 'get_market_snapshot':('excluded_current_snapshot','实时/最新快照含当前基本面字段，不可回填历史时点。'),
 'get_owner_plate':('excluded_current_membership','当前板块归属不是历史成员版本；不改冻结股票池。'),
 'get_dividends':('excluded_no_increment','分红与拆股信息属标签质量审计；本轮保留原公司行动标签，不把重复公司行动当预测信息。'),
 'get_rehab':('excluded_no_increment','公司行动仅验证已冻结标签口径，不更改原结果。'),
 'get_share_buyback':('unavailable_us','实测美股不支持，不能作为美股历史回购特征。'),
 'get_insider_trades':('excluded_disclosure_unknown','交易日期区间不证明公开披露日；没有可核对原受理时间，不用于事件惊喜。'),
 'get_institutional_holdings':('excluded_original_filing_unknown','季度持仓与update_time不证明首次披露及历史版本。'),
 'get_macro_indicator_list':('excluded_metadata_only','宏观指标目录可得，本身不是可回测信息增量。'),
 'get_economic_calendar':('excluded_vintages_unproven','日历有previous/consensus/actual，但未提供原始预期与实际修订版本。'),
 'get_earnings_calendar':('excluded_vintages_and_timeout','本次历史窗口实测NN_ProtoRet_TimeOut；既有缓存也不证明当时预期可知，未将其作为历史共识。'),
 'get_order_book':('current_only_not_historical','实际订阅取得5档，但服务端时间字段为空；无两年历史深度，不进入训练或成交有效性结论。'),
 'get_rt_ticker':('current_only_stale','订阅取得20条，但全为2026-10-02旧逐笔；不当作10月5日实时数据或两年回测。'),
 'get_macro_indicator_history':('excluded_vintages_unproven','100行含release_time/value/predict_value/previous_value，也包含未来尚未公布的N/A行；没有原始修订历史，不做as-released惊喜。'),
 'get_option_volatility':('unavailable_wrong_instrument','传入AMD股票代码返回只支持期权代码；本轮所需历史标的IV/HV由专用历史接口取得，不留下依赖。')}
for actual,canonical in {
 'get_financials_earnings_price_history':'get_earnings_price_history','get_financials_earnings_price_move':'get_earnings_price_move','get_financials_statements':'get_financial_statements',
 'get_research_analyst_consensus':'get_stock_analyst_consensus','get_research_rating_summary':'get_stock_analyst_ratings','get_search_news':'get_stock_news',
 'get_corporate_actions_dividends':'get_dividends','get_corporate_actions_stock_splits':'get_rehab','get_corporate_actions_buybacks':'get_share_buyback','get_insider_trade_list':'get_insider_trades','get_shareholders_institutional':'get_institutional_holdings'}.items():DISPOSITION[actual]=DISPOSITION[canonical]

def main():
    probes=json.loads((P/'probe_results.json').read_text());rows=[]
    for r in probes['records']:
        method=r['method'];status,reason=DISPOSITION.get(method,('excluded_not_required','不服务于登记的四族历史输入。'))
        parts=[{k:v for k,v in p.items() if k in ['kind','rows','columns','file','sha256','attrs']} for p in r.get('parts',[])]
        rows.append(dict(name=r['name'],method=method,ret=r.get('ret'),requested_at=r.get('requested_at'),received_at=r.get('received_at'),disposition=status,reason=reason,parts=parts,error=r.get('error')))
    for method,r in json.loads((P/'extra_probes.json').read_text()).items():
        status,reason=DISPOSITION[method];rows.append(dict(method=method,disposition=status,reason=reason,**r))
    acquisition=json.loads((P/'acquisition.json').read_text());summary=[]
    for family,method in [('capital','get_capital_flow'),('options','get_option_underlying_his_statistic'),('volatility','get_option_underlying_his_volatility'),('short_volume','get_daily_short_volume')]:
        rs=[v for r in acquisition['records'] for v in r.get('records',[]) if v['family']==family];covered=[r for r in rs if r['rows']]
        summary.append(dict(family=family,method=method,requested=124,returned=len(rs),stocks_with_rows=len(covered),rows=sum(r['rows'] for r in rs),first=min(r['first'] for r in covered) if covered else None,last=max(r['last'] for r in covered) if covered else None,errors=[dict(symbol=r['symbol'],errors=r['errors']) for r in rs if r['errors']]))
    subscription=json.loads((P/'subscription_verification.json').read_text());write(P/'api_audit.json',dict(at=now(),status='completed_all_dispositions_closed',requested=['2024-10-04','2026-09-30'],historical_pit_certified=False,probe_count=len(rows),probes=rows,history=summary,subscription=subscription,orders_sent=False,purchases=0,cloud_upload=False,acquisition_sha256=sha(P/'acquisition.json')))
    lines=['# 富途接口可行性结案','','所有本轮相关接口已实际调用。四类历史字段已加入固定训练矩阵；其余逐项排除。没有待批准采购、待登录步骤或未实施训练协议。未证明的历史原始版本作为明确限制保留，不能改称真实PIT。','','| 数据族 | 返回股票/请求 | 行数 | 实际范围 |','|---|---:|---:|---|']
    for r in summary:lines.append(f"| {r['family']} | {r['stocks_with_rows']}/{r['requested']} | {r['rows']} | {r['first']}–{r['last']} |")
    lines+=['','| 接口 | 实际调用结果 | 结案 |','|---|---|---|']
    for r in rows:lines.append(f"| {r['method']} | ret={r.get('ret')} | {r['reason']} |")
    lines+=['','盘口单独订阅验证成功，并在满足最低订阅时长后取消本连接订阅。五档Bid/Ask的服务端时间为空，20条逐笔时间全为十月2日，不能认证当下可买；也没有历史盘口。','','工程修复：AXTI等响应将数值与N/A混用，首次Parquet序列化失败；保留失败日志，原响应JSON归档后将N/A转为缺失，再有限断点重跑。缺失没有填零或删除股票。资金流归一化只用与冻结OHLCV一致的完整RTH真实turnover，缺失不估算。','','方法与权限以实际OpenD返回为准。官方说明：[资金流](https://openapi.futunn.com/futu-api-doc/quote/get-capital-flow.html)、[期权历史统计](https://openapi.futunn.com/futu-api-doc/quote/get-option-underlying-his-statistic.html)、[每日卖空成交](https://openapi.futunn.com/futu-api-doc/quote/get-daily-short-volume.html)。资金流文档有一年/两年表述差异，本轮只报告实取一年；期权统计按≤364日窗口翻页，OI注明T-1仍不回移。','','IBKR开户不会自动修复历史共识/修订版本与历史可成交报价。本轮未访问IBKR账户，未下单、未购买权限、未上传R2。']
    (P/'API_AUDIT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(status='completed',history=summary)),flush=True)

if __name__=='__main__':main()
