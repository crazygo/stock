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
    universe=lineage['registration']['universe']['stocks']
    reportdir=Path(registration['source_report_path'])
    pool=pd.read_csv(reportdir/'all_stocks.csv',keep_default_na=False).set_index('ticker')
    panel=pd.read_parquet(latest['panel_path'],columns=['ticker','date','open','high','low','close','volume']).sort_values('date')
    pricecols=['open','high','low','close','volume']
    valid=np.isfinite(panel[pricecols]).all(axis=1)&panel[['open','high','low','close']].gt(0).all(axis=1)&panel.volume.ge(0)&panel.high.ge(panel[['open','close','low']].max(axis=1))&panel.low.le(panel[['open','close','high']].min(axis=1))
    all_sessions=calendar('2023-01-01',asof);valid_session_dates={s['session_date'] for s in all_sessions['sessions']}
    panel=panel.loc[valid&panel.date.le(asof)&panel.date.dt.strftime('%Y-%m-%d').isin(valid_session_dates)]
    grouped={t:g for t,g in panel.groupby('ticker',sort=False)}
    qpath=CACHE/'latest_market_snapshots.json';quotes=json.loads(qpath.read_text()) if qpath.exists() else {'stocks':{},'received_at':None}
    notes=json.loads((HERE/'business_reviews.json').read_text()) if (HERE/'business_reviews.json').exists() else {}
    cache=ROOT/'.cache/potential_recovery_v1';cache.mkdir(exist_ok=True)
    expected=calendar('2024-10-02',asof);sessions=[r['session_date'] for r in expected['sessions']]
    result=[];outcomes=[];rawhashes={};matched_core=0;notquote=0
    for t,meta in universe.items():
        row=pool.loc[t].to_dict() if t in pool.index else {};g=grouped.get(t)
        base={'ticker':t,'name':meta.get('name'),'cik':meta.get('cik'),'price_date':row.get('price_date'),'ratio_source':'completed_daily_reference','reason':[]}
        if g is None or g.empty:base['reason'].append('missing_valid_daily');outcomes.append(base);continue
        year=g.loc[g.date.ge('2026-01-01')];last=g.iloc[-1]
        if year.empty:base['reason'].append('missing_2026_daily');outcomes.append(base);continue
        peak=year.loc[year.high.idxmax()];after=year.loc[year.date.ge(peak.date)];trough=after.loc[after.low.idxmin()]
        ref=number(row.get('current_close'));quote=quotes['stocks'].get(t,{});qp=number(quote.get('last_price'))
        current=qp if qp and str(quote.get('update_time','')).startswith(registration['quote_session_date']) else ref
        if current is None or current<=0:base['reason'].append('missing_reference_price');outcomes.append(base);continue
        base.update(reference_price=current,daily_close=ref,year_peak=float(peak.high),year_peak_date=peak.date.date().isoformat(),year_ratio=float(peak.high)/current,ratio_source='current_snapshot' if current==qp and qp is not None else 'completed_daily_reference')
        # Pre-screen is explicit and fixed to the completed daily reference, not future quote winners.
        numeric={k:number(row.get(k)) for k in ['operating_margin','net_margin','ocf_margin_ttm','revenue_yoy','report_age_days','dollar_volume_actual']}
        gates={'day_ratio_ge1_45':ref is not None and peak.high/ref>=1.45,'operating_margin_ge8':numeric['operating_margin'] is not None and numeric['operating_margin']>=.08,'net_margin_ge4':numeric['net_margin'] is not None and numeric['net_margin']>=.04,'OCF_margin_ge6':numeric['ocf_margin_ttm'] is not None and numeric['ocf_margin_ttm']>=.06,'revenue_yoy_ge_minus10':numeric['revenue_yoy'] is not None and numeric['revenue_yoy']>=-.10,'recent_period':numeric['report_age_days'] is not None and numeric['report_age_days']<=150,'liquid_10m':numeric['dollar_volume_actual'] is not None and numeric['dollar_volume_actual']>=1e7,'price_ge5':ref is not None and ref>=5,'confirmed_security':meta.get('security_description_confirmed',False)}
        base['reason']=[k for k,v in gates.items() if not v]
        if not all(gates.values()) and t!='MXL':outcomes.append(base);continue
        matched_core+=int(all(gates.values()));cik=meta.get('cik');fpath=ROOT/f'.cache/doubling_opportunity_v1/coverage_v4/companyfacts/CIK{cik:010d}.json'
        financial=quality(json.loads(fpath.read_text()),cutoff);rawhashes[str(fpath)]=sha(fpath)
        support=int((year.close>=.88*peak.high).sum())
        year_expected={d for d in valid_session_dates if d>='2026-01-01'};year_complete={d.date().isoformat() for d in year.date}==year_expected
        price_clean=int(number(row.get('source_unknown_events')) or 0)==0 and str(row.get('source_whole_history_unknown')).lower()!='true' and str(row.get('source_latest_official_session_covered')).lower()=='true' and support>=3 and year_complete
        cap=number(quote.get('total_market_val'));large=cap is not None and cap>=2e9;ratio=base['year_ratio']
        recovery=(current-trough.low)/(peak.high-trough.low) if peak.high>trough.low else None
        ret20=float(last.close/g.iloc[-21].close-1) if len(g)>=21 else None
        if current<trough.low or (recovery is not None and recovery<.10 and ret20 is not None and ret20<-.03):stage='仍处低点 / 下行'
        elif recovery is not None and recovery<.10:stage='低位磨底'
        elif recovery is not None and recovery<.35:stage='初步回升'
        elif recovery is not None and recovery<.65:stage='回升中'
        else:stage='明显恢复'
        tier='A 财务证据较扎实' if all(financial['checks'].values()) and large and price_clean else 'B 仍有质量或口径疑点'
        risks=[k for k,v in financial['checks'].items() if not v]
        if not large:risks.append('市值不足20亿美元或未取得本次市值')
        if not price_clean:risks.append('价格事件、最新覆盖或高点多日支持待核验')
        if not quote:notquote+=1
        annual=financial['annual'];annual_last=annual[-1] if annual else {}
        shares=annual_last.get('diluted_shares');earning_yield=annual_last.get('net_income')/cap if cap and annual_last.get('net_income') is not None else None
        fcf_yield=annual_last.get('fcf')/cap if cap and annual_last.get('fcf') is not None else None
        sources={'companyfacts_sha256':rawhashes[str(fpath)],'companyfacts_url':f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json','SEC_filings_url':f'https://www.sec.gov/edgar/browse/?CIK={cik}&owner=exclude','financial_period_end':financial['metrics'].get('report_period_end'),'financial_snapshot_cutoff':cutoff,'financial_source_received_no_later_than':'2026-10-05T14:55:42Z'}
        bars=g.loc[g.date.ge('2024-10-02'),['date',*pricecols]].copy();bars['date']=bars.date.dt.strftime('%Y-%m-%d')
        rec={**base,'comparison_only':t=='MXL','core_checks':gates,'quality_tier':tier,'quality_score':financial['score'],'quality_risks':risks,'financial':financial,'sector':SECTORS.get(str(row.get('industry_code'))[:2],'其他/待核验'),'quote_time_et':quote.get('update_time'),'quote_received_at':quotes.get('received_at'),'market_cap':cap,'snapshot_pe_ttm':number(quote.get('pe_ttm_ratio')),'annual_earning_yield':earning_yield,'annual_FCF_yield':fcf_yield,'liquidity_dollars20':numeric['dollar_volume_actual'],'year_peak_close_support_days':support,'price_basis_clean':price_clean,'peak_after_low':float(trough.low),'peak_after_low_date':trough.date.date().isoformat(),'recovery_fraction':recovery,'rebound_from_peak_after_low':float(current/trough.low-1),'ret20':ret20,'stage':stage,'sessions_since_peak':len(after),'observed_2026_sessions':len(year),'first_daily_date':g.date.min().date().isoformat(),'latest_daily_date':last.date.date().isoformat(),'financial_sources':sources,'review':notes.get(t),'bars':bars.values.tolist()}
        rec['price_year_complete']=year_complete;rec['expected_2026_sessions']=len(year_expected)
        rec['reason']=[];result.append(rec);outcomes.append({k:v for k,v in rec.items() if k not in ['bars','financial','financial_sources','review']})
    result.sort(key=lambda r:(r['comparison_only'],not r['quality_tier'].startswith('A'),-r['quality_score'],-r['year_ratio']))
    summary={'generated_at':datetime.now(timezone.utc).isoformat(),'client_date':'2026-10-06','date_timezone':'America/New_York','daily_asof':asof,'price_reference_note':'Daily chart is completed 2026-10-02; current reference uses explicitly timestamped Oct-05 snapshot when available. No partial Oct-05 candle.','registered_pool':len(universe),'preliminary_core_candidates':matched_core,'display_stocks_including_comparison':len(result),'quotes_requested':quotes.get('requested_count'),'quotes_returned':quotes.get('returned_count'),'quotes_received_at':quotes.get('received_at'),'unquoted_display_stocks':notquote,'tiers':{tier:sum(r['quality_tier'].startswith(tier) and not r['comparison_only'] for r in result) for tier in ['A','B']},'ratio_counts':{str(v):sum(r['year_ratio']>=v and not r['comparison_only'] for r in result) for v in [1.5,1.8,1.9,2.,2.5]},'navigation_start':'2024-10-02','navigation_end':asof,'source_dataset_key':latest['dataset_key'],'source_report_path':str(reportdir),'current_members_not_PIT':True,'probabilities_or_signals_created':False,'independent_effect_verified':False}
    summary['A_ratio_counts']={str(v):sum(r['year_ratio']>=v and r['quality_tier'].startswith('A') and not r['comparison_only'] for r in result) for v in [1.5,1.8,1.9,2.,2.5]}
    summary['price_comparable_ratio_counts']={str(v):sum(r['year_ratio']>=v and r['price_basis_clean'] and not r['comparison_only'] for r in result) for v in [1.5,1.8,1.9,2.,2.5]}
    summary['official_business_reviews']=len(notes);summary['calendar_sources']=expected['sources'];summary['calendar_sessions_sha256']=expected['sessions_sha256']
    summaries=[{k:v for k,v in r.items() if k!='bars'} for r in result]
    (HERE/'screen_results.json').write_text(json.dumps(clean({'metadata':summary,'stocks':summaries}),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    pd.DataFrame(clean(outcomes)).to_csv(HERE/'all_pool_outcomes.csv',index=False)
    local_sources=[HERE/'PROTOCOL.md',HERE/'source_registration.json',HERE/'business_reviews.json',Path(__file__).resolve()]
    receipt={'generated_at':summary['generated_at'],'source_paths_and_sha256':{latest['panel_path']:sha(latest['panel_path']),str(reportdir/'all_stocks.csv'):sha(reportdir/'all_stocks.csv'),str(qpath):sha(qpath),**{str(p):sha(p) for p in local_sources},**rawhashes},'no_legacy_source_mutation':True,'registered_source_cache_reused':True,'raw_OHLC_and_quotes_not_in_git':True}
    (HERE/'SOURCE_EVIDENCE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    data=clean({'metadata':summary,'stocks':result,'sessions':sessions})
    template=(HERE/'template.html').read_text() if (HERE/'template.html').exists() else None
    if template:
        html=template.replace('__DATA__',json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/'))
        (HERE/'index.html').write_text(html)
        (HERE/'HTML_MANIFEST.json').write_text(json.dumps({'generated_at':summary['generated_at'],'html_sha256':sha(HERE/'index.html'),'template_sha256':sha(HERE/'template.html'),'data_summary_sha256':sha(HERE/'screen_results.json'),'stocks':len(result),'bytes':len(html.encode()),'source_dataset_key':latest['dataset_key']},indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('A candidates',[(r['ticker'],round(r['year_ratio'],2),r['stage']) for r in result if r['quality_tier'].startswith('A')])
if __name__=='__main__':main()
