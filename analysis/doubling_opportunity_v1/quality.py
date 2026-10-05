"""Point-in-time SEC facts for current candidate review; not a historical quality backtest."""
from __future__ import annotations
from datetime import date
from pathlib import Path
import json
from data import get_json,CACHE

TAGS={
 'revenue':['RevenueFromContractWithCustomerExcludingAssessedTax','Revenues','SalesRevenueNet','RevenueFromContractWithCustomerIncludingAssessedTax'],
 'gross_profit':['GrossProfit'], 'net_income':['NetIncomeLoss','ProfitLoss'],
 'cash':['CashAndCashEquivalentsAtCarryingValue'], 'current_assets':['AssetsCurrent'],
 'current_liabilities':['LiabilitiesCurrent'], 'equity':['StockholdersEquity'],
 'operating_cashflow':['NetCashProvidedByUsedInOperatingActivities']}


def quality_from_facts(facts,asof):
    us=facts.get('facts',{}).get('us-gaap',{});latest={};quarters={}
    cutoff=date.fromisoformat(asof)
    for field,tags in TAGS.items():
        candidates=[]
        for tag in tags:
            for r in us.get(tag,{}).get('units',{}).get('USD',[]):
                if not r.get('filed') or date.fromisoformat(r['filed'])>=cutoff:continue
                if r.get('form') not in ('10-K','10-Q','20-F','40-F'):continue
                if date.fromisoformat(r['end'])>cutoff:continue
                x={**r,'tag':tag}
                if field in ('revenue','gross_profit','net_income','operating_cashflow'):
                    if not r.get('start'):continue
                    duration=(date.fromisoformat(r['end'])-date.fromisoformat(r['start'])).days
                    if not 60<=duration<=120:continue
                candidates.append(x)
        # Latest publicly filed revision known at cutoff. Never use future restatements.
        if candidates:
            dedup={}
            for r in sorted(candidates,key=lambda r:r['filed']):dedup[(r.get('start'),r['end'])]=r
            candidates=list(dedup.values());latest[field]=max(candidates,key=lambda r:(r['end'],r['filed']))
            quarters[field]=candidates
    vals={k:r['val'] for k,r in latest.items()};metrics={}
    if 'revenue' in latest:
        curr=latest['revenue'];end=date.fromisoformat(curr['end'])
        prior=[r for r in quarters['revenue'] if 350<=(end-date.fromisoformat(r['end'])).days<=380]
        if prior:
            prev=max(prior,key=lambda r:(r['end'],r['filed']))
            if prev['val']>0:metrics['revenue_yoy']=curr['val']/prev['val']-1
    if vals.get('revenue',0)>0 and 'gross_profit' in vals and latest['gross_profit']['end']==latest['revenue']['end']:
        metrics['gross_margin']=vals['gross_profit']/vals['revenue']
    if vals.get('current_liabilities',0)>0 and 'current_assets' in vals and latest['current_assets']['end']==latest['current_liabilities']['end']:
        metrics['current_ratio']=vals['current_assets']/vals['current_liabilities']
    metrics.update({k:vals.get(k) for k in ['cash','net_income','operating_cashflow','equity']})
    same_period=latest.get('revenue',{}).get('end')
    fresh=bool(same_period and (cutoff-date.fromisoformat(same_period)).days<=183)
    # A transparent necessary-condition checklist, not an intrinsic valuation or a guarantee of company quality.
    checks={'recent_filing_period':fresh,'growing_revenue':metrics.get('revenue_yoy',-1)>.1,
            'positive_gross_margin':metrics.get('gross_margin',-1)>.2,
            'current_ratio_at_least_one':metrics.get('current_ratio',-1)>=1,
            'positive_equity':metrics.get('equity') is not None and metrics['equity']>0,
            'positive_profit_or_operating_cashflow':any(metrics.get(k) is not None and metrics[k]>0 for k in ['net_income','operating_cashflow'])}
    return {'entity_name':facts.get('entityName'),'asof':asof,'status':'quantitative_check_pass' if all(checks.values()) else 'not_passed_or_incomplete',
            'metrics':metrics,'checks':checks,'source_facts':{k:{x:r.get(x) for x in ['val','start','end','filed','accn','form','tag']} for k,r in latest.items()},
            'scope':'current review; business catalyst and historical joint-filter backtest remain separate'}


def inspect_company(ticker,meta,asof,refresh=False):
    cik=meta.get('cik')
    if ticker=='MXL' and not cik:cik=1288469  # MaxLinear official filing header; source in catalysts.json.
    if not cik:return {'status':'cik_mapping_missing','ticker':ticker}
    p=CACHE/'companyfacts'/f'CIK{cik:010d}.json'
    mxl=CACHE/'mxl_companyfacts.json'
    if ticker=='MXL' and mxl.exists() and not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(mxl.read_bytes())
    url=f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json'
    try:
        j=get_json(url,p,refresh)
        return {**quality_from_facts(j,asof),'ticker':ticker,'cik':cik,'source_url':url}
    except Exception as exc:
        return {'ticker':ticker,'cik':cik,'source_url':url,'status':'download_unavailable','error_type':type(exc).__name__}
