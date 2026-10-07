#!/usr/bin/env python3
"""Reproducible, descriptive multi-stage quality/recovery exploration. No predictions."""
from __future__ import annotations
import csv, hashlib, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from context import HERE, ROOT, CACHE
OLD = ROOT / 'analysis/doubling_opportunity_v1/coverage_v4'
sys.path.insert(0, str(OLD.parent/'joint_v2'))
from financials import extract_facts, snapshot, asof_records
sys.path.insert(0, str(ROOT))
from scripts.model_history_calendar import calendar

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp,datetime)): return value.isoformat()[:10]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value,float) and not math.isfinite(value): return None
    return value

def number(value):
    try:
        n=float(value);return n if math.isfinite(n) else None
    except (ValueError,TypeError):return None

def facts(payload, tags, unit, cutoff, flow=False):
    rows=[]
    for priority,tag in enumerate(tags):
        for namespace in ('us-gaap','dei'):
            for r in payload.get('facts',{}).get(namespace,{}).get(tag,{}).get('units',{}).get(unit,[]):
                if r.get('form') not in ('10-K','10-Q','10-K/A','10-Q/A','20-F','20-F/A','40-F','40-F/A'):continue
                if not all(r.get(x) for x in ('filed','end','accn')) or r['filed']>=cutoff or r['end']>=cutoff:continue
                if number(r.get('val')) is None:continue
                dur=(pd.Timestamp(r['end'])-pd.Timestamp(r['start'])).days+1 if r.get('start') else 0
                if flow and not 300<=dur<=400:continue
                if not flow and r.get('start'):continue
                rows.append({**r,'tag':tag,'priority':priority,'duration_days':dur})
    selected={}
    for r in rows:
        key=(r.get('start'),r['end']);rank=(r['filed'],-r['priority'])
        if key not in selected or rank>(selected[key]['filed'],-selected[key]['priority']):selected[key]=r
    return list(selected.values())

def matched(rows,end,start=None):
    same=[r for r in rows if r['end']==end and (start is None or r.get('start')==start)]
    return max(same,key=lambda r:(r['filed'],-r['priority'])) if same else None

def quality(payload, cutoff):
    parsed=extract_facts(payload);snap=snapshot(parsed,cutoff);metrics=snap['metrics']
    balance_end=metrics.get('report_period_end');balance_supplements={}
    if metrics.get('equity') is None:
        equity=matched(facts(payload,['StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],'USD',cutoff),balance_end)
        if equity:
            metrics['equity']=equity['val'];balance_supplements['equity']=equity
    annual={k:[r for r in asof_records(parsed[k],cutoff) if 300<=r.get('duration_days',0)<=400] for k in ['revenue','net_income','operating_cashflow']}
    extra={
      'capex':facts(payload,['PaymentsToAcquirePropertyPlantAndEquipment','PaymentsToAcquireProductiveAssets'], 'USD',cutoff,True),
      'sbc':facts(payload,['ShareBasedCompensation','StockBasedCompensation'], 'USD',cutoff,True),
      'diluted_shares':facts(payload,['WeightedAverageNumberOfDilutedSharesOutstanding'],'shares',cutoff,True)}
    revenue=sorted(annual['revenue'],key=lambda r:r['end']);periods=[]
    # One fiscal period per year; exact start/end pairing prevents cumulative mixing.
    for r in reversed(revenue):
        if periods and (pd.Timestamp(periods[-1]['end'])-pd.Timestamp(r['end'])).days<300:continue
        periods.append(r)
        if len(periods)==3:break
    periods.reverse();history=[]
    for r in periods:
        rec={'start':r['start'],'end':r['end'],'revenue':r['val'],'source_filed':r['filed'],'source_accn':r['accn'],'fact_sources':{'revenue':r}}
        for k in ['net_income','operating_cashflow','capex','sbc','diluted_shares']:
            fact=matched(annual.get(k,extra.get(k,[])),r['end'],r['start'])
            rec[k]=fact['val'] if fact else None
            rec['fact_sources'][k]=fact
        if rec['operating_cashflow'] is not None and rec['capex'] is not None:rec['fcf']=rec['operating_cashflow']-rec['capex']
        else:rec['fcf']=None
        rec['owner_cash_after_sbc']=rec['fcf']-rec['sbc'] if rec['fcf'] is not None and rec['sbc'] is not None else None
        history.append(rec)
    latest=history[-1] if history else {};ocf=latest.get('operating_cashflow');cash=metrics.get('cash');liabilities=metrics.get('liabilities')
    # With missing cash, all liabilities / OCF is an even more conservative bound.
    upper=(liabilities-(cash if cash is not None else 0))/ocf if None not in (liabilities,ocf) and ocf>0 else None
    shares=[r.get('diluted_shares') for r in history[-2:]]
    dilution=shares[-1]/shares[-2]-1 if len(shares)==2 and None not in shares and shares[-2]>0 else None
    good=lambda k:len(history)==3 and all(r.get(k) is not None and r[k]>0 for r in history)
    checks={
      'three_years_GAAP_profit':good('net_income'),
      'three_years_positive_OCF':good('operating_cashflow'),
      'FCF_latest_and_two_of_three':len(history)==3 and latest.get('fcf') is not None and latest['fcf']>0 and sum(r.get('fcf') is not None and r['fcf']>0 for r in history)>=2,
      'revenue_not_contracting':metrics.get('revenue_yoy',-1)>=0,
      'liquidity':metrics.get('current_ratio',0)>=1 or (cash is not None and metrics.get('current_liabilities') is not None and cash>=metrics['current_liabilities']),
      'positive_equity':metrics.get('equity',-1)>0,
      'liability_pressure_upper_bound_le3':upper is not None and upper<=3,
      'diluted_share_growth_le5pct':dilution is not None and dilution<=.05,
      'owner_cash_after_SBC_positive':latest.get('owner_cash_after_sbc') is not None and latest['owner_cash_after_sbc']>0}
    weights=[15,15,15,10,10,10,10,5,10]
    score=sum(w for k,w in zip(checks,weights) if checks[k])
    return {'metrics':metrics,'annual':history,'checks':checks,'score':score,'liability_pressure_upper_bound':upper,'liability_bound_cash_missing':cash is None,'balance_supplements':balance_supplements,'diluted_share_growth':dilution,'quarter_source':snap['sources'].get('revenue')}

SECTORS={'73':'软件与商业服务','38':'医疗/精密设备','36':'电子与半导体','35':'工业设备','28':'医药与化工','58':'餐饮','56':'服装零售','59':'零售','80':'医疗服务','48':'通信','67':'金融/地产','49':'能源公用事业','50':'分销','20':'食品饮料','10':'矿业','33':'金属'}
def main():
    registration=json.loads((HERE/'source_registration.json').read_text());latest=registration['backtest_registration'];lineage=json.loads((Path(latest['path'])/'lineage.json').read_text());asof=registration['daily_asof'];cutoff=registration['financial_cutoff_exclusive']
    catalog=lineage['registration']['universe'];universe=catalog['stocks'];reportdir=Path(registration['source_report_path'])
    pool=pd.read_csv(reportdir/'all_stocks.csv',keep_default_na=False).set_index('ticker')
    panel=pd.read_parquet(latest['panel_path'],columns=['ticker','date','open','high','low','close','volume']).sort_values('date')
    pricecols=['open','high','low','close','volume']
    valid=np.isfinite(panel[pricecols]).all(axis=1)&panel[['open','high','low','close']].gt(0).all(axis=1)&panel.volume.ge(0)&panel.high.ge(panel[['open','close','low']].max(axis=1))&panel.low.le(panel[['open','close','high']].min(axis=1))
    invalid_counts=panel.loc[~valid].groupby('ticker').size().to_dict()
    duplicates=panel.loc[panel.duplicated(['ticker','date'],keep=False)].groupby('ticker').size().to_dict()
    all_sessions=calendar('2023-01-01',asof);valid_session_dates={s['session_date'] for s in all_sessions['sessions']}
    panel=panel.loc[valid&panel.date.le(asof)&panel.date.dt.strftime('%Y-%m-%d').isin(valid_session_dates)]
    grouped={t:g for t,g in panel.groupby('ticker',sort=False)}
    qpath=CACHE/'latest_market_snapshots.json';quotes=json.loads(qpath.read_text()) if qpath.exists() else {'stocks':{},'received_at':None}
    notes=json.loads((HERE/'business_reviews.json').read_text()) if (HERE/'business_reviews.json').exists() else {}
    ai_snapshot=json.loads((HERE/'ai_relevance.json').read_text());ai_labels=ai_snapshot['labels']
    expected=calendar('2024-10-02',asof);sessions=[r['session_date'] for r in expected['sessions']]
    year_expected={d for d in valid_session_dates if d>='2026-01-01'}
    q2_expected={d for d in year_expected if '2026-04-01'<=d<='2026-06-30'}
    windows={'YTD':sorted(year_expected),'AprJun':sorted(q2_expected),**{f'M{m:02d}':sorted(d for d in year_expected if d[5:7]==f'{m:02d}') for m in [4,5,6]}}
    def audit(g):
        days=set(g.date.dt.strftime('%Y-%m-%d')) if g is not None else set();first=min(days) if days else None;out={}
        for name,ds in windows.items():
            missing=sorted(set(ds)-days);leading=[d for d in missing if first and d<first];internal=[d for d in missing if first and d>=first]
            status='完整' if not missing else '完全缺失' if len(missing)==len(ds) else '起点不足且区间缺日' if leading and internal else '起点不足' if leading else '区间缺日'
            out.update({f'{name}_expected':len(ds),f'{name}_observed':len(ds)-len(missing),f'{name}_missing':len(missing),f'{name}_status':status,f'{name}_missing_dates':missing,f'{name}_before_first':len(leading),f'{name}_after_first':len(internal)})
        return out
    result=[];outcomes=[];rawhashes={};matched_core=0;notquote=0;fresh=0
    for t,meta in universe.items():
        row=pool.loc[t].to_dict() if t in pool.index else {};g=grouped.get(t);coverage=audit(g)
        base={'ticker':t,'name':meta.get('name'),'cik':meta.get('cik'),'price_date':row.get('price_date'),'ratio_source':'completed_daily_reference','reason':[],**coverage,'first_daily_date':g.date.min().date().isoformat() if g is not None else None,'latest_daily_date':g.date.max().date().isoformat() if g is not None else None,'invalid_panel_rows':int(invalid_counts.get(t,0)),'duplicate_panel_rows':int(duplicates.get(t,0)),'source_unknown_events':int(number(row.get('source_unknown_events')) or 0),'source_whole_history_unknown':str(row.get('source_whole_history_unknown')).lower()=='true','security_description_confirmed':bool(meta.get('security_description_confirmed',False))}
        base.update(ai_grade=ai_labels.get(t,{}).get('grade'),ai_status=ai_labels.get(t,{}).get('status','outside_AI_review_scope'))
        if g is None or g.empty:base['reason'].append('missing_valid_daily');outcomes.append(base);continue
        year=g.loc[g.date.ge('2026-01-01')];last=g.iloc[-1]
        if year.empty:base['reason'].append('missing_2026_daily');outcomes.append(base);continue
        peak=year.loc[year.high.idxmax()];after=year.loc[year.date.ge(peak.date)];trough=after.loc[after.low.idxmin()]
        ref=float(last.close);quote=quotes['stocks'].get(t,{});qp=number(quote.get('last_price'))
        current_quote=qp is not None and qp>0 and str(quote.get('update_time','')).startswith(registration['quote_session_date'])
        current=qp if current_quote else ref;fresh+=int(current_quote)
        base.update(reference_price=current,daily_close=ref,year_peak=float(peak.high),year_peak_date=peak.date.date().isoformat(),year_ratio=float(peak.high)/current,daily_year_ratio=float(peak.high)/ref,ratio_source='current_snapshot' if current_quote else 'completed_daily_reference',quote_time_et=quote.get('update_time'),quote_received_at=quote.get('snapshot_received_at',quotes.get('received_at')),year_peak_is_complete=coverage['YTD_missing']==0)
        numeric={k:number(row.get(k)) for k in ['operating_margin','net_margin','ocf_margin_ttm','revenue_yoy','report_age_days','dollar_volume_actual']}
        gates={'operating_margin_ge8':numeric['operating_margin'] is not None and numeric['operating_margin']>=.08,'net_margin_ge4':numeric['net_margin'] is not None and numeric['net_margin']>=.04,'OCF_margin_ge6':numeric['ocf_margin_ttm'] is not None and numeric['ocf_margin_ttm']>=.06,'revenue_yoy_ge_minus10':numeric['revenue_yoy'] is not None and numeric['revenue_yoy']>=-.10,'recent_period':numeric['report_age_days'] is not None and numeric['report_age_days']<=150,'liquid_10m':numeric['dollar_volume_actual'] is not None and numeric['dollar_volume_actual']>=1e7,'price_ge5':ref>=5,'confirmed_security':meta.get('security_description_confirmed',False)}
        base.update(numeric);base.update({k:row.get(k) for k in ['financial_source_support','financial_source_support_note','financial_reporting_units','financial_source_taxonomies']});base['unparsed_financial_fields']=[k for k,v in numeric.items() if v is None and k!='dollar_volume_actual'];base['core_pass']=all(gates.values());base['core_checks']=gates;base['reason']=[k for k,v in gates.items() if not v]
        # No ratio pre-screen. Every basic-operating candidate is re-evaluated, including R<1.2.
        if not all(gates.values()) and t!='MXL':outcomes.append(base);continue
        matched_core+=int(all(gates.values()));cik=meta.get('cik');fpath=ROOT/f'.cache/doubling_opportunity_v1/coverage_v4/companyfacts/CIK{cik:010d}.json' if cik else None
        if not fpath or not fpath.exists():base['reason'].append('missing_companyfacts');outcomes.append(base);continue
        financial=quality(json.loads(fpath.read_text()),cutoff);rawhashes[str(fpath)]=sha(fpath)
        support=int((year.close>=.88*peak.high).sum());year_complete=coverage['YTD_missing']==0
        price_clean=base['source_unknown_events']==0 and not base['source_whole_history_unknown'] and str(row.get('source_latest_official_session_covered')).lower()=='true' and support>=3 and year_complete and not base['invalid_panel_rows'] and not base['duplicate_panel_rows']
        cap=number(quote.get('total_market_val')) if current_quote else None;large=cap is not None and cap>=2e9
        recovery=(current-trough.low)/(peak.high-trough.low) if peak.high>trough.low else None
        ret20=float(last.close/g.iloc[-21].close-1) if len(g)>=21 else None
        if current<trough.low or (recovery is not None and recovery<.10 and ret20 is not None and ret20<-.03):stage='仍处低点 / 下行'
        elif recovery is not None and recovery<.10:stage='低位磨底'
        elif recovery is not None and recovery<.35:stage='初步回升'
        elif recovery is not None and recovery<.65:stage='回升中'
        else:stage='明显恢复'
        tier='A 财务证据较扎实' if all(financial['checks'].values()) and large and price_clean else 'B 仍有质量或口径疑点'
        risks=[k for k,v in financial['checks'].items() if not v]
        if not large:risks.append('市值不足20亿美元或未取得同日市值')
        if not price_clean:risks.append('价格事件、今年覆盖或高点多日支持待核验')
        if not current_quote:notquote+=1
        annual=financial['annual'];annual_last=annual[-1] if annual else {}
        earning_yield=annual_last.get('net_income')/cap if cap and annual_last.get('net_income') is not None else None
        fcf_yield=annual_last.get('fcf')/cap if cap and annual_last.get('fcf') is not None else None
        sources={'companyfacts_sha256':rawhashes[str(fpath)],'companyfacts_url':f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json','SEC_filings_url':f'https://www.sec.gov/edgar/browse/?CIK={cik}&owner=exclude','financial_period_end':financial['metrics'].get('report_period_end'),'financial_snapshot_cutoff':cutoff,'financial_source_received_no_later_than':registration['financial_source_received_no_later_than']}
        bars=g.loc[g.date.ge('2024-10-02'),['date',*pricecols]].copy();bars['date']=bars.date.dt.strftime('%Y-%m-%d')
        rec={**base,'comparison_only':t=='MXL','core_checks':gates,'quality_tier':tier,'quality_score':financial['score'],'quality_risks':risks,'financial':financial,'sector':SECTORS.get(str(row.get('industry_code'))[:2],'其他/待核验'),'market_cap':cap,'snapshot_pe_ttm':number(quote.get('pe_ttm_ratio')) if current_quote else None,'annual_earning_yield':earning_yield,'annual_FCF_yield':fcf_yield,'liquidity_dollars20':numeric['dollar_volume_actual'],'year_peak_close_support_days':support,'price_basis_clean':price_clean,'peak_after_low':float(trough.low),'peak_after_low_date':trough.date.date().isoformat(),'recovery_fraction':recovery,'rebound_from_peak_after_low':float(current/trough.low-1),'ret20':ret20,'stage':stage,'sessions_since_peak':len(after),'observed_2026_sessions':len(year),'financial_sources':sources,'review':notes.get(t),'bars':bars.values.tolist(),'outside_prior_1_45_prescreen':peak.high/ref<1.45,'price_year_complete':year_complete,'expected_2026_sessions':len(year_expected)}
        rec['ai']=ai_labels[t]
        rec['reason']=[];result.append(rec);outcomes.append({k:v for k,v in rec.items() if k not in ['bars','financial','financial_sources','review','ai']})
    result.sort(key=lambda r:(r['comparison_only'],not r['quality_tier'].startswith('A'),-r['quality_score'],-r['year_ratio']))
    candidates=[r for r in result if not r['comparison_only']];ratios=[1.2,1.5,1.8,1.9,2.,2.5]
    coverage_summary={key:{'expected_sessions':len(ds),'complete':sum(r[f'{key}_missing']==0 for r in outcomes),'partial':sum(0<r[f'{key}_observed']<len(ds) for r in outcomes),'zero':sum(r[f'{key}_observed']==0 for r in outcomes),'before_first_total_missing_days':sum(r[f'{key}_before_first'] for r in outcomes),'after_first_total_missing_days':sum(r[f'{key}_after_first'] for r in outcomes)} for key,ds in windows.items()}
    summary={'revision':'whole_pool_1_2_v2','generated_at':datetime.now(timezone.utc).isoformat(),'client_date':'2026-10-06','date_timezone':'America/New_York','daily_asof':asof,'price_reference_note':'Daily chart uses completed Oct-02. Same-session timestamped Oct-05 snapshots are descriptive references; missing quotes explicitly retain completed daily Close.','registered_pool':len(universe),'preliminary_core_candidates':matched_core,'display_stocks_including_comparison':len(result),'quotes_requested':quotes.get('requested_count'),'quotes_returned':quotes.get('returned_count'),'same_session_quotes':fresh,'quotes_received_at':quotes.get('received_at'),'quote_request_interval_start':quotes.get('request_interval_start',quotes.get('requested_at')),'unquoted_display_stocks':notquote,'tiers':{tier:sum(r['quality_tier'].startswith(tier) for r in candidates) for tier in ['A','B']},'ratio_counts':{str(v):sum(r['year_ratio']>=v for r in candidates) for v in ratios},'navigation_start':'2024-10-02','navigation_end':asof,'source_dataset_key':latest['dataset_key'],'source_report_path':str(reportdir),'current_members_not_PIT':True,'probabilities_or_signals_created':False,'independent_effect_verified':False,'ratio_prescreen_removed':True,'coverage':coverage_summary,'registered_tickers_sha256':hashlib.sha256('\n'.join(sorted(universe)).encode()).hexdigest(),'all_pool_daily_ratio_counts':{str(v):sum(r.get('daily_year_ratio',0)>=v and r['YTD_missing']==0 for r in outcomes) for v in ratios},'all_pool_snapshot_ratio_counts':{str(v):sum(r.get('year_ratio',0)>=v and r['YTD_missing']==0 and r.get('ratio_source')=='current_snapshot' for r in outcomes) for v in ratios}}
    summary['A_ratio_counts']={str(v):sum(r['year_ratio']>=v and r['quality_tier'].startswith('A') for r in candidates) for v in ratios}
    summary['price_comparable_ratio_counts']={str(v):sum(r['year_ratio']>=v and r['price_basis_clean'] for r in candidates) for v in ratios}
    summary['A_early_ratio_counts']={str(v):sum(r['year_ratio']>=v and r['quality_tier'].startswith('A') and r['recovery_fraction']<.35 for r in candidates) for v in ratios}
    summary['new_1_2_to_1_5']={'base':sum(1.2<=r['year_ratio']<1.5 for r in candidates),'A':sum(1.2<=r['year_ratio']<1.5 and r['quality_tier'].startswith('A') for r in candidates),'outside_prior_prescreen':sum(r['year_ratio']>=1.2 and r['outside_prior_1_45_prescreen'] for r in candidates)}
    summary['A_candidates_with_peak_AprJun']=[r['ticker'] for r in candidates if r['quality_tier'].startswith('A') and r['year_ratio']>=1.2 and '2026-04-01'<=r['year_peak_date']<='2026-06-30']
    summary['complete_year_peak_month_counts']={f'{m:02d}':sum(r['YTD_missing']==0 and str(r.get('year_peak_date',''))[5:7]==f'{m:02d}' for r in outcomes) for m in range(1,11)}
    summary['exclusion_reason_counts']={k:sum(k in r['reason'] for r in outcomes) for k in sorted({k for r in outcomes for k in r['reason']})}
    summary['official_business_reviews']=len(notes);summary['calendar_sources']=expected['sources'];summary['calendar_sessions_sha256']=expected['sessions_sha256']
    summary['catalog']={'directory_count':catalog['metadata']['directory_count'],'registered':len(universe),'outside_registered':len(catalog['exclusions']),'outside_reason_counts':{k:sum(r.get('classification_reason')==k for r in catalog['exclusions']) for k in sorted({r.get('classification_reason') for r in catalog['exclusions']})},'security_review_pending':len(catalog['review']),'catalog_received_footer':catalog['metadata']['nasdaq_footer']}
    summary['daily_core_ratio_counts']={str(v):sum(r['daily_year_ratio']>=v for r in candidates) for v in ratios}
    summary['daily_A_ratio_counts']={str(v):sum(r['daily_year_ratio']>=v and r['quality_tier'].startswith('A') for r in candidates) for v in ratios}
    summary['basic_candidates_with_same_session_quotes']=sum(r['ratio_source']=='current_snapshot' for r in candidates)
    summary['basic_candidates_with_complete_AprJun']=sum(r['AprJun_missing']==0 for r in candidates)
    summary['basic_candidates_with_complete_YTD']=sum(r['YTD_missing']==0 for r in candidates)
    summary['registered_with_unparsed_basic_financial_fields']=sum(bool(r.get('unparsed_financial_fields')) for r in outcomes)
    summary['revision']='whole_pool_1_2_v2_ai_v1'
    summary['ai_classified_at']=ai_snapshot['metadata']['classified_at']
    summary['ai_scope_candidates']=len(candidates)
    summary['ai_counts']={k:sum((r['ai_grade'] or 'pending')==k for r in candidates) for k in ['++','+','0','pending']}
    related=[r for r in candidates if r['ai_grade'] in ['++','+']]
    summary['ai_ratio_counts']={str(v):sum(r['year_ratio']>=v and r['price_basis_clean'] for r in related) for v in ratios}
    summary['ai_A_ratio_counts']={str(v):sum(r['year_ratio']>=v and r['quality_tier'].startswith('A') for r in related) for v in ratios}
    summary['ai_default_counts']={k:sum(r['ai_grade']==k and r['year_ratio']>=1.2 and r['quality_tier'].startswith('A') for r in candidates) for k in ['++','+']}
    summary['ai_early_A_1_2']=sum(r['year_ratio']>=1.2 and r['quality_tier'].startswith('A') and r['recovery_fraction']<.35 for r in related)
    summary['ai_1_2_to_1_5_A']=sum(1.2<=r['year_ratio']<1.5 and r['quality_tier'].startswith('A') for r in related)
    summary['ai_A_1_2_hidden']={k:sum((r['ai_grade'] or 'pending')==k and r['year_ratio']>=1.2 and r['quality_tier'].startswith('A') for r in candidates) for k in ['0','pending']}
    catalog_rows=[{'ticker':t,'name':m.get('name'),'catalog_status':'registered','reason':m.get('classification_reason'),'cik':m.get('cik')} for t,m in universe.items()]+[{'ticker':r['ticker'],'name':r.get('name'),'catalog_status':'outside_registered','reason':r.get('classification_reason'),'cik':r.get('cik')} for r in catalog['exclusions']]
    pd.DataFrame(catalog_rows).to_csv(HERE/'catalog_outcomes.csv',index=False)
    summaries=[{k:v for k,v in r.items() if k!='bars'} for r in result]
    (HERE/'screen_results.json').write_text(json.dumps(clean({'metadata':summary,'stocks':summaries}),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    csv_out=[{k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in clean(outcomes)]
    pd.DataFrame(csv_out).to_csv(HERE/'all_pool_outcomes.csv',index=False)
    coverage_keys=['ticker','name','ai_grade','ai_status','first_daily_date','latest_daily_date','year_peak_date','year_peak_is_complete','daily_year_ratio','year_ratio','ratio_source','quote_time_et','quote_received_at','core_pass','quality_tier','price_basis_clean','reason','unparsed_financial_fields','financial_source_support','financial_source_support_note','financial_reporting_units','source_unknown_events','source_whole_history_unknown','invalid_panel_rows','duplicate_panel_rows']+[k for k in outcomes[0] if any(k.startswith(w+'_') for w in windows)]
    audit_rows=[{k:r.get(k) for k in coverage_keys} for r in outcomes]
    (HERE/'coverage_audit.json').write_text(json.dumps(clean({'metadata':summary,'stocks':audit_rows}),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    local_sources=[HERE/'PROTOCOL.md',HERE/'AI_PROTOCOL.md',HERE/'ai_relevance.json',HERE/'source_registration.json',HERE/'business_reviews.json',Path(__file__).resolve(),HERE/'acquire_quotes.py']
    receipt={'generated_at':summary['generated_at'],'source_paths_and_sha256':{latest['panel_path']:sha(latest['panel_path']),str(Path(latest['path'])/'lineage.json'):sha(Path(latest['path'])/'lineage.json'),str(reportdir/'all_stocks.csv'):sha(reportdir/'all_stocks.csv'),str(qpath):sha(qpath),**{str(p):sha(p) for p in local_sources},**rawhashes},'no_legacy_source_mutation':True,'registered_source_cache_reused':True,'raw_OHLC_and_quotes_not_in_git':True}
    (HERE/'SOURCE_EVIDENCE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    data=clean({'metadata':summary,'stocks':result,'sessions':sessions,'audit':audit_rows})
    template=(HERE/'template.html').read_text()
    html=template.replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/'))
    (HERE/'index.html').write_text(html)
    (HERE/'HTML_MANIFEST.json').write_text(json.dumps({'generated_at':summary['generated_at'],'html_sha256':sha(HERE/'index.html'),'template_sha256':sha(HERE/'template.html'),'data_summary_sha256':sha(HERE/'screen_results.json'),'stocks':len(result),'bytes':len(html.encode()),'source_dataset_key':latest['dataset_key']},indent=2)+'\n')
    print(json.dumps(clean(summary),ensure_ascii=False,indent=2))
    print('A >=1.2',[(r['ticker'],round(r['year_ratio'],3),r['year_peak_date'],r['stage']) for r in candidates if r['quality_tier'].startswith('A') and r['year_ratio']>=1.2])
if __name__=='__main__':main()
