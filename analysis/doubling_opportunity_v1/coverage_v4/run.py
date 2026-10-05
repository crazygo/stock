#!/usr/bin/env python3
"""One registered whole-pool v4 report; live quote and probability reference stay explicit."""
from __future__ import annotations
import argparse,collections,hashlib,importlib.util,json,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import joblib,numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent
sys.path.insert(0,str(HERE));from build import FEATURES,VERSION
sys.path.insert(0,str(PARENT));from data import peak_and_stage
from context import CACHE
_spec=importlib.util.spec_from_file_location('doubling_v3_probability_model',PARENT/'joint_v2/model.py')
_model=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_model);_model.CACHE=CACHE

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    if x is pd.NaT or x is pd.NA:return None
    if isinstance(x,(pd.Timestamp,datetime)):return x.isoformat()
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def pct(x):return '不可评分' if x is None else f'{x:.2%}'
def price(x):return '缺失' if x is None else f'${x:.2f}'
def quote_price(x):
    try:
        value=float(x)
        return value if np.isfinite(value) and value>0 else None
    except (ValueError,TypeError):return None
def quality(x):return {'growth_and_operating_health_checks_pass':'增长且经营健康（必要条件）',
    'growing_but_operating_health_unproven':'增长，经营健康待证','financial_cache_missing':'财务未覆盖',
    'stale_or_missing_financial_statement':'财务缺失或过期','not_passed_or_incomplete':'未通过或资料不足',
    'financial_statement_missing_or_unsupported':'财务缺失或不支持'}.get(x,x)

def markdown(j):
    m=j['metadata'];lines=['# 广池分类与发行人覆盖修复 v4 · 单次报告','',
        '**没有通过完整“当下60日翻倍概率>80%”门禁的股票。**','',
        f"运行 {m['run_at']}；完整日线参考截止 {m['data_asof']}；候选 {m['accepted_count']}，可评分 {m['price_scored_count']}；源不可用 {m['unavailable_equity_count']}。",
        f"其中普通股描述已确认 {m['explicit_equity_count']}，证券类型待核验 {m['security_review_count']}；后者保留概率与来源，但不得发出合格信号。当前目录与CIK映射仍是回溯成员信息。",
        'SEC文件已取得但IFRS或币种不受冻结解析器支持时，财务保持未知，不解释为经营差；all_stocks.csv的financial_source_support与financial_source_amount_units分别显示原因和原文金额单位。当前报价补充默认30秒总预算，超时保留已完整落盘的来源；概率仍属于日线参考价。',
        f"报价缓存接收 {m['quotes'].get('received_at','未取得')}；距本次报告 {round(m['quote_cache_age_seconds']) if m.get('quote_cache_age_seconds') is not None else '未知'} 秒；更新日期早于最近完整交易日的 {m['stale_quote_tickers']} 股。该时间不是各时段价格的成交时间。",
        f"保留报价快照返回 {m['quotes'].get('returned_count',0)}/{m['accepted_count']} 个证券；缺少有效last_price的 {m['missing_valid_quote_price_count']} 个证券保留日线参考。此次补充耗时 {m['quote_refresh_attempt'].get('elapsed_seconds','未执行')} 秒，是否超时：{m['quote_refresh_attempt'].get('timed_out','离线复用')}。完整逐股缺失原因见 all_stocks.csv 的 latest_quote_status / latest_quote_missing_reason。",
        '算法：183日价格/量能＋当时已披露财务＋同业量价 → 首次翻倍时间三分类，multinomial LogisticRegression / HistGradientBoostingClassifier，联合温度校准。',
        '日线来自原生Futu NONE快照，拆股按冻结的原生事件与主源公告核对；未支持动作、代码身份冲突和未核实隔夜跳空保留未知。阶段和概率基于已完成日线，盘前/盘后/夜盘报价独立显示；报价变化不能自动改变旧概率或当前入场目标。',
        '完整公司行动与历史身份、原生日High范围、PIT证券/退市、业务催化历史和新的独立效果观察仍未核验。经营必要条件与工程检查不能证明翻倍概率。','',
        '## 经营必要条件与个股量价热度候选（探索）','',
        '|股票|日线参考价/日期|快照last_price/ET更新时间|夜盘价格|上次确认高峰/日期|日线价占高峰|快照last_price占高峰|日线阶段|P30估计|P60估计|财务期间/营收同比|业务催化|',
        '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in j['healthy_and_market_heat_stocks']:
        f=r['financial_metrics'];lines.append(f"|{r['ticker']}|{price(r['current_close'])}/{r['price_date']}|{price(r.get('latest_quote_price'))}/{r.get('latest_quote_time_et','缺失')}|{price(r.get('latest_overnight_price'))}|{price(r.get('previous_peak_price'))}/{r.get('previous_peak_date','未确认')}|{pct(r.get('current_to_previous_peak'))}|{pct(r.get('latest_quote_to_previous_peak'))}|{r.get('stage','待证')}|{pct(r['p30'])}|{pct(r['p60'])}|{r.get('financial_period_end','未覆盖')}/{pct(f.get('revenue_yoy'))}|{r['business_catalyst_status']}|")
    if not j['healthy_and_market_heat_stocks']:lines.append('|—|没有同时通过已登记条件的探索候选|—|—|—|—|—|—|—|—|—|—|')
    lines+=['', '以上是经营必要条件与个股热度候选；同业条件和完整概率证据仍须通过，不能称合格买入。',
        '上次确认高峰：183日历天内，前后各5根已观测日线的局部High最大值，且后5根至少回落5%；最近5根尚不能确认高峰。阶段是该日线截止下的量价描述。','',
        '## 全池概率排序（探索，保留质量未通过者）','',
        '|股票|日线价/日期|上次高峰/日期|占高峰|阶段|P30估计|P60估计|经营质量|同业热度|',
        '|---|---|---|---|---|---|---|---|---|']
    for r in j['exploratory_price_stocks']:
        lines.append(f"|{r['ticker']}|{price(r.get('current_close'))}/{r.get('price_date')}|{price(r.get('previous_peak_price'))}/{r.get('previous_peak_date') or '未确认'}|{pct(r.get('current_to_previous_peak'))}|{r.get('stage')}|{pct(r['p30'])}|{pct(r['p60'])}|{quality(r['company_quality_status'])}|{r.get('industry_heat_status','缺失')}|")
    diagnostic=j.get('frozen_probability_diagnostic')
    if diagnostic:
        s=next(x for x in diagnostic['summary'] if x['algorithm']=='monthly_selected' and x['horizon']==60)
        lines+=['',f"已冻结全池高估计诊断：不施加公司质量/热点门禁，仍沿用>80%及同股60日冷却。月前选择策略60日有 {s['signals']} 次诊断机会，{s['tp']} TP / {s['fp']} FP / {s['unknown']} 未知，成熟precision={pct(s['precision'])}。这些不是原主策略有效信号；小样本和未知不能证明当前高估计可靠。诊断不改模型、标签或资格，独立产物见 {diagnostic['path']}。"]
    lines+=['','## 固定4—9月历史前向开发回测','',
        '|目标|算法/策略|发出>80%信号|股票/日期|TP/FP|未知|precision|Wilson95%|TP/全部信号|同股基准|增量|数值门禁|',
        '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for s in j['backtest']:
        lines.append(f"|{s['horizon']}日|{s['algorithm_name']}|{s['signals']}|{s['stock_count']}/{s['decision_dates']}|{s['tp']}/{s['fp']}|{s['unknown']}|{pct(s['precision'])}|{'–'.join(pct(v) for v in s['wilson95'])}|{pct(s['conservative_lower_bound'])}|{pct(s['same_stock_base_rate'])}|{pct(s['increment'])}|{'数值通过，完整证据仍不足' if s['historical_numeric_gate'] else '未通过'}|")
    lines+=['','未知与失败全部保留；多算法重复、每日预测行和相邻窗口不当作独立机会。成熟标签需完整未来前缀，合股自身不算翻倍；没有信号时准确率不可评分。','',
        '## MXL已暴露案例','']
    for r in j['case_stocks']:
        lines.append(f"MXL：日线参考 {r.get('price_date')} {price(r.get('current_close'))}，上次确认高峰 {r.get('previous_peak_date')} {price(r.get('previous_peak_price'))}，占比 {pct(r.get('current_to_previous_peak'))}，阶段 {r.get('stage')}；P30={pct(r.get('p30'))}，P60={pct(r.get('p60'))}。这是已暴露案例，不能当独立验证。")
    if m.get('native_minute_source_comparison'):
        s=m['native_minute_source_comparison']
        lines+=['',f"已有本地NONE分钟源核对：登记 {s['registered_existing_minute_stocks']} 股，审计 {s['audited_stocks']} 股；完整常规盘5分钟与日线同时覆盖 {s['complete_rth_5m_and_native_daily_days']} 个股票日，其中日线High高出常规盘至少0.5%的 {s['daily_high_above_complete_rth_at_least_0_5_percent']} 日。仅诊断已有缓存，不改变标签，不代表全市场RTH覆盖。"]
    lines+=['','全池、未评分与未覆盖在 all_stocks.csv；来源SHA、拆股数学、财报披露切分、日历、未知、股票簇/周簇区间与概率参考见 report.json 和 backtests 的 lineage.json。','']
    return '\n'.join(lines)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--top',type=int,default=30)
    ap.add_argument('--reuse-quotes',action='store_true',help='Explicitly reuse the existing quote snapshot for an offline report; its age stays visible.')
    args=ap.parse_args()
    if args.top<1:ap.error('--top must be positive')
    pointer=json.loads((HERE/'latest_backtest.json').read_text());back=Path(pointer['path']);lineage=json.loads((back/'lineage.json').read_text())
    for path,digest in {**lineage['source_hashes'],**lineage['training_source_hashes'],**lineage['financial_input_hashes'],**lineage['derived_hashes'],**lineage['artifact_hashes']}.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen input/code changed; no stale model report: '+path)
    asof=pointer['asof'];run_at=datetime.now(timezone.utc)
    expected=max(x['session_date'] for x in lineage['calendar']['sessions'] if datetime.fromisoformat(x['close_at'])<=run_at)
    models,audit=joblib.load(pointer['model_path']);d=pd.read_parquet(pointer['dataset_path'])
    latest=d[d.date==pd.Timestamp(asof)].set_index('ticker');scored=latest[latest.eligible];forecasts={}
    if models and len(scored):
        model=models[audit['champion']];a,b,_=_model.nested_probabilities(model['calibrated'],scored,FEATURES)
        forecasts={t:(float(x),float(y)) for t,x,y in zip(scored.index,a,b)}
    if not args.reuse_quotes:
        ranked=sorted(forecasts,key=lambda t:forecasts[t][1],reverse=True)[:args.top]
        healthy=[t for t,z in scored.iterrows() if z.quality_status_at_event=='growth_and_operating_health_checks_pass' and z.ret20>.1 and z.rs20>.05 and z.volume_ratio>=1.1][:args.top]
        priority=list(dict.fromkeys(ranked+healthy+['MXL']))
        print('report stage: bounded whole-pool quote supplementation (30 seconds), report candidates first',flush=True)
        subprocess.run([sys.executable,str(HERE/'acquire_quotes.py'),'--refresh','--budget-seconds','30','--priority-symbols',*priority],check=True)
    run_at=datetime.now(timezone.utc)
    expected=max(x['session_date'] for x in lineage['calendar']['sessions'] if datetime.fromisoformat(x['close_at'])<=run_at)
    panel=pd.read_parquet(pointer['panel_path'],filters=[('date','>',pd.Timestamp(asof)-pd.Timedelta(days=183))]);groups={t:g for t,g in panel.groupby('ticker')}
    cover=pd.read_csv(Path(pointer['source_path'])/'coverage.csv',keep_default_na=False).set_index('ticker').to_dict('index')
    u=lineage['registration']['universe']['stocks'];qs=CACHE/'latest_market_snapshots.json'
    quote_bytes=qs.read_bytes() if qs.exists() else b'{"stocks":{}}';quotes=json.loads(quote_bytes)
    attempt_file=CACHE/'latest_quote_refresh_attempt.json'
    quote_attempt=json.loads(attempt_file.read_text()) if attempt_file.exists() and not args.reuse_quotes else {}
    received=quotes.get('received_at');quote_cache_age_seconds=None
    if received:
        try:
            received_time=pd.Timestamp(received)
            if received_time.tzinfo is not None:quote_cache_age_seconds=(run_at-received_time.to_pydatetime()).total_seconds()
        except (ValueError,TypeError):pass
    catalysts=json.loads((PARENT/'catalysts.json').read_text());unconfirmed={r['ticker'] for r in lineage['unconfirmed_current_ticker_mappings']};rows=[]
    for t,item in u.items():
        row={'ticker':t,'name':item['name'],'catalog_category':item['category'],'security_description_confirmed':item['security_description_confirmed'],'qualified':False,'p30':None,'p60':None,'current_close':None,'price_date':None,
            'latest_quote_price':None,'latest_quote_time_et':None,
            'score_status':'source_unavailable','source_coverage':cover.get(t,{}),'company_quality_status':'financial_cache_missing',
            'financial_metrics':{},'market_heat_status':'不足','business_catalyst_status':'待核验','probability_anchor_date':asof,
            'security_review':'证券类型待核验' if not item['security_description_confirmed'] else
                'SEC当前ticker未匹配，财务保留缺失' if t in unconfirmed or not item.get('cik') else 'SEC当前ticker已确认'}
        source_profile=lineage['financial_source_profiles'].get(str(item.get('cik')),{})
        row['financial_source_support']=source_profile.get('financial_source_support','SEC_companyfacts_missing')
        row['financial_reporting_units']=';'.join(source_profile.get('financial_reporting_units',[]))
        row['financial_source_amount_units']=';'.join(source_profile.get('financial_amount_unit_candidates',[]))
        row['financial_source_taxonomies']=';'.join(source_profile.get('financial_source_taxonomies',[]))
        row['financial_source_support_note']='来源支持状态不证明公司健康；IFRS或币种未支持保持未知，不解释为经营差'
        if t in groups:
            p=groups[t];valid=p.dropna(subset=['open','high','low','close'])
            row.update(**peak_and_stage(valid,t,asof));row['price_date']=row.get('actual_price_date')
            row['current_close']=float(valid.iloc[-1].native_close) if len(valid) else None
            row['score_status']='missing_latest_official_session'
        if t in latest.index:
            z=latest.loc[t];row.update(current_close=float(z.close) if pd.notna(z.close) else None,price_date=asof,
                score_status='insufficient_or_invalid_price_history',company_quality_status=z.quality_status_at_event,
                financial_period_end=z.report_period_end,financial_available_at=z.available_at,
                joint_signal_eligible_at_anchor=bool(z.joint_signal_eligible),industry_code=z.industry_code,
                industry_ret20=z.industry_ret20,industry_rs20=z.industry_rs20,industry_volume_ratio=z.industry_volume_ratio,
                industry_members=int(z.industry_members),unsupported_action_feature_window=bool(z.unsupported_action_feature_window))
            row['price_metrics']={k:z[k] for k in ['ret5','ret20','ret60','volume_ratio','coverage183','dollar_volume_actual','rs20','rs60']}
            row['industry_heat_status']='同业量价热度通过' if z.industry_missing==0 and z.industry_rs20>.03 and z.industry_volume_ratio>=1.05 else '同业热度未通过或缺失'
            row['financial_metrics']={k:z['fin_'+k] for k in ['revenue_yoy','revenue_qoq','gross_margin','operating_margin','net_margin','ocf_margin_ttm','current_ratio','report_age_days']}
            row['financial_features']={k:z[k] for k in FEATURES if k.startswith('fin_')}
            if z.unsupported_action_feature_window:
                row['unverified_peak_computation']={k:row.get(k) for k in ['previous_peak_price','previous_peak_date','current_to_previous_peak','stage']}
                row.update(previous_peak_price=None,previous_peak_date=None,current_to_previous_peak=None,stage='来源疑点，峰值与阶段未评分')
            row['market_heat_status']='个股量价热度通过' if z.ret20>.1 and z.rs20>.05 and z.volume_ratio>=1.1 else '量价热度未通过'
            if t in forecasts:row.update(p30=forecasts[t][0],p60=forecasts[t][1],score_status='split_financial_heat_model_scored')
        q=quotes.get('stocks',{}).get(t,{})
        row['latest_quote_status']='not_in_retained_snapshot'
        row['latest_quote_missing_reason']=quotes.get('unavailable',{}).get(t,{}).get('reason','no_retained_snapshot_for_security')
        if q:
            row.update(latest_quote_price=quote_price(q.get('last_price')),latest_quote_time_et=q.get('update_time'),latest_pre_price=quote_price(q.get('pre_price')),
                latest_after_price=quote_price(q.get('after_price')),latest_overnight_price=quote_price(q.get('overnight_price')),latest_quote_received_at=quotes.get('received_at'),
                quote_time_semantics='snapshot update_time is not verified per-session trade time')
            row['latest_quote_status']='price_available_in_snapshot' if row['latest_quote_price'] is not None else 'snapshot_price_missing_or_invalid'
            row['latest_quote_missing_reason']='' if row['latest_quote_price'] is not None else 'last_price_missing_nonpositive_or_nonfinite'
            update_date=str(q.get('update_time') or '')[:10]
            row['quote_freshness']=('missing_update_time_not_verified' if not update_date else
                'stale_quote_update' if update_date<expected else 'quote_update_date_current_or_newer_not_trade_time_proof')
            row['quote_cache_age_seconds']=quote_cache_age_seconds
            if q.get('trust_valid'):row['security_review']='OpenD信托/基金字段有效，普通经营公司资格待核验'
            if row.get('previous_peak_price') and row['latest_quote_price'] is not None:row['latest_quote_to_previous_peak']=row['latest_quote_price']/row['previous_peak_price']
            if row.get('current_close') and row['latest_quote_price'] is not None:row['quote_vs_probability_reference_ratio']=row['latest_quote_price']/row['current_close']
        cat=catalysts.get(t)
        if cat and cat['published_at']<asof and (pd.Timestamp(asof)-pd.Timestamp(cat['published_at'])).days<=183:
            row.update(business_catalyst_status=cat['review_status'],business_catalyst_source=cat['source_url'])
        rows.append(clean(row))
    rows.sort(key=lambda r:(-(r['p60'] if r['p60'] is not None else -1),r['ticker']))
    healthy=[r for r in rows if r['company_quality_status']=='growth_and_operating_health_checks_pass' and r['market_heat_status']=='个股量价热度通过' and r['p60'] is not None and r['security_review']=='SEC当前ticker已确认' and r['security_description_confirmed']]
    metadata={'version':VERSION,'run_at':run_at.isoformat(),'data_asof':asof,'expected_latest_session':expected,
        'freshness':'latest_completed_session' if asof==expected else 'stale_not_current','accepted_count':len(u),'explicit_equity_count':sum(x['security_description_confirmed'] for x in u.values()),'security_review_count':sum(not x['security_description_confirmed'] for x in u.values()),'price_scored_count':len(forecasts),
        'unavailable_equity_count':sum(r['source_coverage'].get('source_status')!='available' for r in rows),
        'healthy_market_heat_count':len(healthy),'joint_policy_eligible_count':sum(bool(r.get('joint_signal_eligible_at_anchor')) for r in rows),
        'financial_source_support_counts':dict(collections.Counter(r['financial_source_support'] for r in rows)),
        'source_missing_latest_equity_count':sum(r['source_coverage'].get('current_session_covered') not in (True,'True',1) for r in rows),
        'qualified_count':0,'order_violations':sum(a>b for a,b in forecasts.values()),'dataset_key':pointer['dataset_key'],
        'model_audit':audit,'quotes':{k:v for k,v in quotes.items() if k not in ['stocks','unavailable']},'financial_asof':asof,
        'quote_cache_age_seconds':quote_cache_age_seconds,'stale_quote_tickers':sum(r.get('quote_freshness')=='stale_quote_update' for r in rows),
        'missing_valid_quote_price_count':sum(r.get('latest_quote_price') is None for r in rows),
        'quote_snapshot_sha256':hashlib.sha256(quote_bytes).hexdigest(),'quote_refresh_mode':'explicit_offline_cache_reuse' if args.reuse_quotes else quote_attempt.get('mode','refresh_attempt_unverified'),
        'quote_refresh_attempt':quote_attempt,
        'quote_acquirer_code_sha256':hashlib.sha256((HERE/'acquire_quotes.py').read_bytes()).hexdigest(),
        'previous_peak_definition':'last 183-calendar-day High maximum over 5 preceding and 5 following observed bars, with at least 5 percent pullback in the following 5 bars; latest 5 bars cannot confirm a peak',
        'reference_cost':lineage['config']['reference_cost'],
        'probability_reference':'latest completed native daily Close with frozen reference cost; differing live session quote is a different entry reference',
        'regular_hours_label_verified':False,'new_independent_effect_verification':False,'source':'Futu NONE + frozen native/primary split reconciliation; unsupported actions and unresolved overnight basis/news retained unknown',
        'action_source_audit':lineage['action_source_summary'],
        'pending':['complete_corporate_action_and_historical_identity_coverage','RTH_only_label_verification','PIT_and_delisted_securities','business_catalyst_history','new_independent_effect_observation','live_entry_reference_probability_validation']}
    report=clean({'metadata':metadata,'qualified_stocks':[],'healthy_and_market_heat_stocks':healthy[:args.top],
        'exploratory_price_stocks':[r for r in rows if r['p60'] is not None][:args.top],'case_stocks':[r for r in rows if r['ticker']=='MXL'],
        'backtest':json.loads((back/'backtest_summary.json').read_text()),'source_lineage_path':str(back/'lineage.json')})
    diagnostic_pointer=HERE/'latest_probability_diagnostic.json'
    if diagnostic_pointer.exists():
        dp=json.loads(diagnostic_pointer.read_text())
        if dp['training_key']==pointer['training_key']:
            root=Path(dp['path']);dl=json.loads((root/'lineage.json').read_text())
            if dl['registration']['prediction_sha256']!=lineage['artifact_hashes'][pointer['prediction_path']]:raise ValueError('Probability diagnostic uses another frozen prediction source')
            for script,key in [('diagnose_probabilities.py','diagnostic_code_sha256'),('PROBABILITY_DIAGNOSTIC_PROTOCOL.md','protocol_sha256')]:
                if hashlib.sha256((HERE/script).read_bytes()).hexdigest()!=dl['registration'][key]:raise ValueError('Diagnostic recipe changed; run the complete scan before publishing it')
            for file,digest in dl['artifacts'].items():
                if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest:raise ValueError('Probability diagnostic artifact changed: '+file)
            verified=json.loads((root/'VERIFICATION.json').read_text())
            if verified['key']!=dp['key'] or verified['training_key']!=pointer['training_key']:raise ValueError('Diagnostic native labels were verified for another source')
            dr=json.loads((root/'report.json').read_text())
            report['frozen_probability_diagnostic']={'path':str(root),'key':dp['key'],'training_key':dp['training_key'],'summary':dr['summary'],'native_label_verification':verified,'diagnostic_only':True,'main_signal_policy_changed':False}
    minute_pointer=CACHE/'native_minute_audit_latest.json'
    if minute_pointer.exists():
        minute=json.loads(minute_pointer.read_text());mp=Path(minute['path'])/'manifest.json';mm=json.loads(mp.read_text())
        if mm['registration']['source_audit']==lineage['action_source_audit']:
            for file,digest in mm['files'].items():
                if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest:raise ValueError('Minute audit artifact changed: '+file)
            report['metadata']['native_minute_source_comparison']=minute['summary']
            report['metadata']['native_minute_source_comparison_path']=minute['path']
    report['metadata']['report_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out=HERE/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out.mkdir(parents=True,exist_ok=True)
    (out/'quote_snapshot.json').write_bytes(quote_bytes)
    (out/'quote_refresh_attempt.json').write_text(json.dumps(quote_attempt,indent=2))
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False));(out/'report.md').write_text(markdown(report))
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['source_coverage','financial_metrics','financial_features','price_metrics']}|
        {(k if k.startswith('source_') else 'source_'+k):v for k,v in r['source_coverage'].items()}|
        r['financial_metrics']|r.get('price_metrics',{}) for r in rows]).to_csv(out/'all_stocks.csv',index=False)
    pd.DataFrame(lineage['registration']['universe']['exclusions']).to_csv(out/'universe_exclusions.csv',index=False)
    (HERE/'latest_run.json').write_text(json.dumps({'path':str(out.resolve()),'data_asof':asof,'run_at':metadata['run_at']},indent=2))
    print(json.dumps({'report':str(out.resolve()),'candidate_count':len(rows),'scored_count':len(forecasts),'healthy_and_market_heat':len(healthy),'qualified':0},indent=2))

if __name__=='__main__':main()
