#!/usr/bin/env python3
"""Repeatable coherent horizon report; current company review remains separate from price-control evidence."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,sys,time
from datetime import datetime,timedelta,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import joblib,numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(HERE))
from model import fit_month,nested_probabilities,ALGORITHMS
from financials import extract_facts,snapshot
sys.path.insert(0,str(PARENT))
from data import CACHE,universe,refresh_archive,load_panel,build_dataset,peak_and_stage,sessions
from scripts.model_history_calendar import calendar


def last_completed_session(now=None):
    now=now or datetime.now(timezone.utc)
    day=now.astimezone(ZoneInfo('America/New_York')).date()
    recent=calendar((day-timedelta(days=14)).isoformat(),day.isoformat())['sessions']
    closed=[r['session_date'] for r in recent if datetime.fromisoformat(r['close_at'])<=now]
    return closed[-1]


def latest_quotes():
    p=CACHE/'latest_market_snapshots.json'
    if not p.exists():return {},{'status':'not_acquired','historical_probability_refresh':False}
    j=json.loads(p.read_text())
    return j.get('stocks',{}),{k:v for k,v in j.items() if k not in ('stocks','unavailable')}


def financial_experiment(asof,dataset_key):
    pointer=HERE/'latest_financial_backtest.json'
    if not pointer.exists():return None
    j=json.loads(pointer.read_text());lineage=json.loads((Path(j['path'])/'lineage.json').read_text())
    if lineage['price_dataset_key']!=dataset_key or j['price_data_asof']!=asof:return None
    for name,digest in lineage['source_hashes'].items():
        p=HERE/name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:return None
    models,audit=joblib.load(j['model_path'])
    if not models:return None
    d=pd.read_parquet(j['dataset_path']);rows=d[(d.date==pd.Timestamp(asof))&d.eligible]
    m=models[audit['champion']];p30,p60,_=nested_probabilities(m['calibrated'],rows,m['features'])
    return {'forecasts':{t:(float(a),float(b)) for t,a,b in zip(rows.ticker,p30,p60)},'audit':audit,
            'summary':json.loads((Path(j['path'])/'backtest_summary.json').read_text()),'pointer':j,
            'anchor_signal_eligible':{t:bool(v) for t,v in zip(rows.ticker,rows.joint_signal_eligible)}}


def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x


def issuer_metadata():
    result={}
    for p in (CACHE/'submissions').glob('*.json'):
        try:
            j=json.loads(p.read_text());result[int(j['cik'])]={'sic':j.get('sic'),'sic_description':j.get('sicDescription'),
                'current_tickers':[t.replace('-','.') for t in j.get('tickers',[])], 'entity_type':j.get('entityType'),
                'source_url':f'https://data.sec.gov/submissions/CIK{int(j["cik"]):010d}.json'}
        except (ValueError,KeyError):continue
    return result


def current_financials(u,asof):
    result={};bycik={}
    for t,m in u.items():
        cik=m.get('cik');p=CACHE/'companyfacts'/f'CIK{cik:010d}.json' if cik else None
        if p is None or not p.exists():result[t]={'quality_status':'financial_cache_missing','metrics':{},'sources':{}};continue
        if cik not in bycik:
            payload=json.loads(p.read_text())
            if int(payload.get('cik',0))!=cik or not payload.get('facts'):
                result[t]={'quality_status':'financial_cache_invalid_identity','metrics':{},'sources':{},'source_status':'retained_as_missing'};continue
            bycik[cik]=snapshot(extract_facts(payload),asof)
        result[t]={**bycik[cik],'cik':cik,'source_url':f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json'}
    return result


def percent(x):return '待证' if x is None else f'{x*100:.2f}%'
def dollars(x):return '缺失' if x is None else f'${x:.2f}'
def quality_label(x):
    return {'growth_and_operating_health_checks_pass':'增长且经营健康（必要条件通过）',
            'growing_but_operating_health_unproven':'增长，经营健康待证',
            'financial_cache_missing':'财报尚未缓存',
            'financial_statement_missing_or_unsupported':'财务缺失或不支持',
            'financial_cache_invalid_identity':'财报身份未核验，保留缺失',
            'not_passed_or_incomplete':'未通过或信息不足'}.get(x,x)


def markdown(report):
    m=report['metadata'];lines=['# 首次翻倍时间模型 v2 · 单次运行报告','',
      f"运行 {m['run_at']}；全市场日线截止 **{m['data_asof']}**，最近应有完整交易日 {m['expected_latest_session']}，状态 **{m['freshness']}**。",
      f"完整证券目录 {m['directory_count']}；普通股 / ADR 候选 {m['accepted_count']}；价格可评分 {m['price_scored_count']}。30日≤60日概率矛盾：**0**。",
      f"快照返回 {m['quotes'].get('returned_count',0)} 个候选，接收时间 {m['quotes'].get('received_at','未取得')}；更新日期早于最近完整交易日的 {m['stale_quote_tickers']} 个，保留过期标记。报价与历史概率分别显示，不会补齐缺少的中间日线。",
      f"当前 SEC 财报已覆盖 {m['financial_cache_tickers']} 个候选；其中增长且经营健康必要条件通过 {m['operating_health_tickers']} 个。获取状态：{'已终止' if m['financial_acquisition'].get('terminal') else '仍在运行'}，未覆盖保留待证。",
      '**没有通过完整“60 日翻倍概率 >80%”证据门禁的股票。**','',
      '**日线时段待核验：Massive 批量日线未确认仅含常规盘。已算 High 触及保留作探索标签，不能视为已验证的常规盘翻倍。公司行动与证券历史类型也尚未统一。**','',
      '算法：同一个 multinomial LogisticRegression / multiclass HistGradientBoostingClassifier 预测首次触及时间，联合 temperature scaling 校准，P30=q(前30日)，P60=q(前30日)+q(31–60日)。',
      ('当前概率来自已完成历史开发回测的财务 × 行业量价模型。财务按当时披露日合并，缺失、未通过与未知保留。业务公告热点及新独立验证仍未完成。' if m['joint_effect_backtested'] else '这是价格对照模型加当前财务核验报告。财务尚未作为训练输入，当前经营健康条件没有事后回填到价格对照的历史分母；公司质量 × 热点的联合模型回测仍未完成。'),
      '概率为模型估计，高置信度尾部尚未通过可靠性检验。当前目录、跨批次公司行动和行情过期问题仍限制解释，不能将极高输出当成已证实的80%机会。','',
      '## 增长且经营健康、量价热度候选（探索）','',
      '|股票|快照 last_price / ET更新时间|概率参考价 / 日期|上次确认高峰 / 日期|last_price占旧高峰*|阶段截止 / 阶段|30日模型估计|60日模型估计|最近营收同比|经营证据|量价热度 / 业务催化|',
      '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in report['healthy_and_market_heat_stocks']:
        f=r['financial_metrics'];lines.append(f"|{r['ticker']}|{dollars(r.get('latest_quote_price'))} / {r.get('latest_quote_time_et','缺失')}|{dollars(r['current_close'])} / {r['price_date']}|{dollars(r.get('previous_peak_price'))} / {r.get('previous_peak_date') or '未确认'}|{percent(r.get('latest_quote_to_previous_peak'))}|{r['price_date']} / {r.get('stage','待证')}|{percent(r.get('p30'))}|{percent(r.get('p60'))}|{percent(f.get('revenue_yoy'))}|经营利润率 {percent(f.get('operating_margin'))}；TTM现金流率 {percent(f.get('ocf_margin_ttm'))}|{r['market_heat_status']} / {r['business_catalyst_status']}|")
    if not report['healthy_and_market_heat_stocks']:lines.append('|—|本次已覆盖子集中没有通过者|—|—|—|—|—|—|—|—|—|')
    lines+=['','*最新报价是未复权价，旧高峰为归档拆股调整价；比值为待公司行动核对的参考。阶段和模型均截至概率参考日期。财务核验截至 '+m['financial_asof']+'，不会回填历史模型。']
    lines+=['','## 全池最高模型估计（尚未通过完整公司与概率证据）','',
      '|股票|30日模型估计|60日模型估计|财务质量|量价热度|证券类别核查|','|---|---|---|---|---|---|']
    for r in report['exploratory_price_stocks']:
        lines.append(f"|{r['ticker']}|{percent(r['p30'])}|{percent(r['p60'])}|{quality_label(r['company_quality_status'])}|{r['market_heat_status']}|{r['security_review']}|")
    lines+=['','## 保留的历史前向开发回测','',
      '|目标|算法 / 策略|发出>80%信号|TP / FP|未知|precision|TP/全部信号|门禁|','|---|---|---|---|---|---|---|---|']
    for s in report['backtest']:
        lines.append(f"|{s['horizon']}日|{s['algorithm_name']}|{s['signals']}|{s['tp']} / {s['fp']}|{s['unknown']}|{percent(s['precision'])}|{percent(s['conservative_lower_bound'])}|未通过|")
    if report.get('financial_backtest'):
        lines+=['','## 财务 × 行业量价联合实验（历史开发）','',
          '|目标|算法 / 策略|已发信号|TP / FP|未知|precision|TP/全部信号|数值门禁|','|---|---|---|---|---|---|---|---|']
        for s in report['financial_backtest']:
            lines.append(f"|{s['horizon']}日|{s['algorithm_name']}|{s['signals']}|{s['tp']} / {s['fp']}|{s['unknown']}|{percent(s['precision'])}|{percent(s['conservative_lower_bound'])}|{'数值通过；完整证据仍不足' if s['historical_numeric_gate'] else '未通过'}|")
    lines+=['','## MXL 已暴露案例','']
    for r in report['case_stocks']:
        lines.append(f"{r['ticker']}：最新快照 last_price {dollars(r.get('latest_quote_price'))}，快照更新时间ET {r.get('latest_quote_time_et','缺失')}；盘后 {dollars(r.get('latest_after_price'))}、夜盘 {dollars(r.get('latest_overnight_price'))} 分别保留，夜盘占旧确认高峰 {percent(r.get('latest_overnight_to_previous_peak'))}（复权基准待核对）。不能由快照时间推断各字段的成交时间。模型截至 {r['price_date']}，参考收盘 {dollars(r['current_close'])}，P30={percent(r['p30'])}，P60={percent(r['p60'])}；经营质量层为 {quality_label(r['company_quality_status'])}。这不是独立验证样本。")
    lines+=['','财报按决定日前 filed 的记录读取；季度累计差分、TTM 连续季度与来源 accession 全部保留。业务催化出处与行情、目录、模型切分、财报覆盖见 report.json。全部候选及未覆盖 / 未评分行在 all_stocks.csv；没有事后删除失败。','']
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--refresh',action='store_true');ap.add_argument('--top',type=int,default=30);ap.add_argument('--output-dir',type=Path)
    ap.add_argument('--quality-top',type=int,default=30,help='Compatibility option: current financial cache is reviewed for the entire pool.')
    args=ap.parse_args();cfg=json.loads((PARENT/'config.json').read_text())
    refreshed=refresh_archive() if args.refresh else {'refresh':'skipped'}
    u,um,exclusions=universe(args.refresh);panel,lineage=load_panel(u);d=build_dataset(panel,cfg);asof=lineage['last_session']
    latest=d[d.date==pd.Timestamp(asof)];scored=latest[latest.eligible].copy()
    dataset_key=json.loads((CACHE/'dataset_key.json').read_text())['key'];key=hashlib.sha256((dataset_key+(HERE/'model.py').read_text()).encode()).hexdigest()[:16]
    p=CACHE/'models'/f'joint_v2_{asof[:7]}_{key}.joblib'
    if p.exists():models,audit=joblib.load(p)
    else:
        print('joint current fit',asof[:7],flush=True);models,audit=fit_month(d,asof[:7],cfg);p.parent.mkdir(exist_ok=True);joblib.dump((models,audit),p)
    forecasts={}
    if models:
        model=models[audit['champion']];p30,p60,classes=nested_probabilities(model['calibrated'],scored,model['features'])
        forecasts={t:(float(a),float(b)) for t,a,b in zip(scored.ticker,p30,p60)}
    control_forecasts=forecasts.copy();joint=financial_experiment(asof,dataset_key)
    if joint:forecasts=joint['forecasts'];audit=joint['audit']
    financial_asof=datetime.now(ZoneInfo('America/New_York')).date().isoformat()
    expected_latest=last_completed_session()
    print('current financial review',flush=True);financial=current_financials(u,financial_asof);meta=issuer_metadata();catalysts=json.loads((PARENT/'catalysts.json').read_text())
    quotes,quote_meta=latest_quotes()
    last=panel.groupby('ticker').tail(1).set_index('ticker');feature_lookup=latest.set_index('ticker');rows=[]
    for ticker,item in u.items():
        f=financial[ticker];r={'ticker':ticker,'name':item['name'],'p30':None,'p60':None,'current_close':None,'price_date':None,
               'score_status':'no_archive_history','qualified':False,'company_quality_status':f['quality_status'],'financial_metrics':f['metrics'],
               'market_heat_status':'不足','business_catalyst_status':'待核验','security_review':'Nasdaq名称规则，尚待精确类型核验'}
        if ticker in last.index:
            r.update(current_close=float(last.loc[ticker,'close']),price_date=last.loc[ticker,'date'].date().isoformat(),**peak_and_stage(panel,ticker,asof))
            r['score_status']='insufficient_history_or_liquidity'
            if ticker in forecasts:r.update(p30=forecasts[ticker][0],p60=forecasts[ticker][1],score_status='financial_market_heat_scored' if joint else 'price_control_scored')
            if ticker in control_forecasts:r.update(price_control_p30=control_forecasts[ticker][0],price_control_p60=control_forecasts[ticker][1])
            if joint:r['joint_signal_eligible_at_anchor']=joint['anchor_signal_eligible'].get(ticker,False)
            if ticker in feature_lookup.index:
                z=feature_lookup.loc[ticker];r['price_features']={k:float(z[k]) for k in ['ret20','rs20','volume_ratio','coverage183','dollar_volume_actual']}
                r['market_heat_status']='个股量价热度通过' if z.ret20>.1 and z.rs20>.05 and z.volume_ratio>=1.1 else '量价热度未通过'
            if r['price_date']!=asof:r['score_status']='missing_latest_session'
        cik=item.get('cik');m=meta.get(cik,{})
        if m:
            r.update(sic=m.get('sic'),sic_description=m.get('sic_description'),sic_mode='current_metadata_not_historical_PIT')
            r['security_review']='SEC当前ticker已确认' if ticker in m['current_tickers'] else 'SEC当前ticker未匹配，待核验'
        lower=item['name'].lower()
        if any(x in lower for x in [' fund','trust','income securities','capital acquisition','acquisition corp']):r['security_review']='基金、信托或收购壳可能性，待精确类型核验'
        quote=quotes.get(ticker,{})
        if quote:
            r.update(latest_quote_price=quote.get('last_price'),latest_quote_time_et=quote.get('update_time'),
                     latest_quote_received_at=quote_meta.get('received_at'),latest_quote_source=quote_meta.get('source'),
                     quote_equity_valid=quote.get('equity_valid'),quote_trust_valid=quote.get('trust_valid'),
                     probability_anchor_date=r['price_date'],price_basis_comparison='raw_quote_vs_mixed_vintage_split_adjusted_peak_unreconciled')
            r.update(latest_pre_price=quote.get('pre_price'),latest_after_price=quote.get('after_price'),latest_overnight_price=quote.get('overnight_price'),
                     quote_time_semantics='update_time is snapshot update time, not a separately verified trade timestamp for each session price')
            update_date=str(quote.get('update_time') or '')[:10]
            r['quote_freshness']='stale_quote_update' if update_date<expected_latest else 'quote_update_date_current_or_newer_not_trade_time_proof'
            if quote.get('trust_valid'):r['security_review']='OpenD信托/基金字段有效，普通经营公司资格待核验'
            peak=r.get('previous_peak_price')
            if peak and quote.get('last_price',0)>0:r['latest_quote_to_previous_peak']=quote['last_price']/peak
            if peak and quote.get('overnight_price',0)>0:r['latest_overnight_to_previous_peak']=quote['overnight_price']/peak
        cat=catalysts.get(ticker)
        if cat and cat['published_at']<asof and (pd.Timestamp(asof)-pd.Timestamp(cat['published_at'])).days<=183:
            r['business_catalyst_status']=cat['review_status'];r['business_catalyst_source']=cat['source_url']
        rows.append(clean(r))
    rows.sort(key=lambda r:(-(r['p60'] if r['p60'] is not None else -1),r['ticker']))
    healthy=[r for r in rows if r['company_quality_status']=='growth_and_operating_health_checks_pass' and r['market_heat_status']=='个股量价热度通过' and r['p60'] is not None and r['security_review']=='SEC当前ticker已确认']
    bt=json.loads((HERE/'latest_backtest.json').read_text());summary=json.loads((Path(bt['path'])/'backtest_summary.json').read_text())
    acquisition=json.loads((CACHE/'fundamentals_acquisition.json').read_text()) if (CACHE/'fundamentals_acquisition.json').exists() else {}
    metadata={**um,'version':'joint_v2_financial_market_heat' if joint else 'joint_v2_price_control_plus_current_financial_review','run_at':datetime.now(timezone.utc).isoformat(),
         'data_asof':asof,'expected_latest_session':expected_latest,'freshness':'current_complete_EOD' if asof==expected_latest else 'stale_not_current',
         'financial_asof':financial_asof,'quotes':quote_meta,
         'stale_quote_tickers':sum(r.get('quote_freshness')=='stale_quote_update' for r in rows),
         'label_session_scope':'Massive grouped daily eligible trade OHLC; RTH-only scope unverified',
         'regular_hours_label_verified':False,
         'price_scored_count':len(forecasts),'financial_cache_tickers':sum('cik' in f for f in financial.values()),
         'operating_health_tickers':sum(f['quality_status']=='growth_and_operating_health_checks_pass' for f in financial.values()),
         'healthy_market_heat_count':len(healthy),'qualified_count':0,'order_violations':0,'dataset_key':dataset_key,'model_audit':audit,
         'financial_acquisition':{k:v for k,v in acquisition.items() if k!='results'},'refresh':refreshed,
         'joint_effect_backtested':bool(joint),'financial_experiment':joint['pointer'] if joint else None,
         'pending':['fresh_full_market_daily','RTH_only_label_verification','PIT_membership_and_security_types','corporate_action_reconciliation','business_catalyst_history','new_independent_observation']+([] if joint else ['all_pool_financial_snapshot','historical_financial_market_heat_joint_backtest'])}
    report=clean({'metadata':metadata,'qualified_stocks':[],'healthy_and_market_heat_stocks':healthy[:args.top],
        'exploratory_price_stocks':[r for r in rows if r['p60'] is not None][:args.top],'case_stocks':[r for r in rows if r['ticker']=='MXL'],
        'company_reviews':{r['ticker']:clean(financial[r['ticker']]) for r in healthy[:args.top]+[x for x in rows if x['ticker']=='MXL']},'backtest':summary,
        'financial_backtest':joint['summary'] if joint else []})
    out=args.output_dir or HERE/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out.mkdir(parents=True,exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False));(out/'report.md').write_text(markdown(report))
    pd.DataFrame([{**{k:v for k,v in r.items() if k not in ('financial_metrics','price_features')},**{'fin_'+k:v for k,v in r['financial_metrics'].items()}} for r in rows]).to_csv(out/'all_stocks.csv',index=False)
    pd.DataFrame(exclusions).to_csv(out/'universe_exclusions.csv',index=False)
    (HERE/'latest_run.json').write_text(json.dumps({'path':str(out.resolve()),'data_asof':asof,'run_at':metadata['run_at']}))
    print(json.dumps({'report':str(out),'price_scored':len(forecasts),'healthy_and_heat':len(healthy),'qualified':0,'max_raw_p60':max(x[1] for x in forecasts.values()) if forecasts else None},indent=2))

if __name__=='__main__':main()
