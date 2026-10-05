"""Conservative SEC as-of quarterly and TTM normalization with source lineage."""
from __future__ import annotations
import hashlib,json,math,os,sys
from datetime import date,timedelta
from pathlib import Path
import numpy as np,pandas as pd

PARENT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PARENT))
from data import CACHE

TAGS={
 'revenue':['RevenueFromContractWithCustomerExcludingAssessedTax','Revenues','SalesRevenueNet','RevenueFromContractWithCustomerIncludingAssessedTax'],
 'gross_profit':['GrossProfit'], 'operating_income':['OperatingIncomeLoss'],
 'net_income':['NetIncomeLoss','ProfitLoss'], 'operating_cashflow':['NetCashProvidedByUsedInOperatingActivities'],
 'cash':['CashAndCashEquivalentsAtCarryingValue'], 'current_assets':['AssetsCurrent'],
 'current_liabilities':['LiabilitiesCurrent'], 'equity':['StockholdersEquity'],
 'assets':['Assets'], 'liabilities':['Liabilities']}
FLOW=set(['revenue','gross_profit','operating_income','net_income','operating_cashflow'])
FEATURES=['fin_revenue_yoy','fin_revenue_qoq','fin_gross_margin','fin_operating_margin','fin_net_margin',
          'fin_ocf_margin_ttm','fin_revenue_ttm_log','fin_current_ratio','fin_equity_assets_ratio',
          'fin_cash_assets_ratio','fin_report_age_days','fin_statement_missing','fin_growth_missing',
          'fin_ttm_missing','fin_liquidity_missing','fin_fresh']


def extract_facts(payload):
    """Keep raw standard USD facts once; no future information is selected here."""
    us=payload.get('facts',{}).get('us-gaap',{});result={}
    for field,tags in TAGS.items():
        rows=[]
        for priority,tag in enumerate(tags):
            for r in us.get(tag,{}).get('units',{}).get('USD',[]):
                if r.get('form') not in ('10-K','10-Q','10-K/A','10-Q/A','20-F','20-F/A','40-F','40-F/A'):continue
                if not all(r.get(x) for x in ('filed','end','accn')):continue
                if not isinstance(r.get('val'),(int,float)) or not math.isfinite(r['val']):continue
                if field in FLOW and not r.get('start'):continue
                x={k:r.get(k) for k in ['start','end','filed','val','accn','form']}
                x.update(tag=tag,priority=priority,source_kind='direct',source_records=[{k:r.get(k) for k in ['start','end','filed','val','accn','form']}])
                if x['start']:
                    dur=(date.fromisoformat(x['end'])-date.fromisoformat(x['start'])).days+1
                    if not 60<=dur<=400:continue
                    x['duration_days']=dur
                rows.append(x)
        result[field]=rows
    return result


def asof_records(rows,cutoff):
    """Latest known revision of an exact period; tag priority breaks same-filing ties."""
    selected={}
    for r in rows:
        if r['filed']>=cutoff or r['end']>=cutoff:continue
        key=(r['start'],r['end'])
        rank=(r['filed'],-r['priority'])
        if key not in selected or rank>(selected[key]['filed'],-selected[key]['priority']):selected[key]=r
    return list(selected.values())


def quarters(rows,cutoff):
    visible=asof_records(rows,cutoff);out={}
    # Directly reported standalone quarter always outranks an inferred cumulative difference.
    for r in visible:
        if 60<=r.get('duration_days',0)<=120:
            if r['end'] not in out or r['filed']>out[r['end']]['filed']:out[r['end']]=r
    for cumulative in visible:
        if not 121<=cumulative.get('duration_days',0)<=400:continue
        prev=[r for r in visible if r['start']==cumulative['start'] and r['tag']==cumulative['tag'] and r['end']<cumulative['end']
              and 60<=(date.fromisoformat(cumulative['end'])-date.fromisoformat(r['end'])).days<=120]
        if not prev:continue
        prior=max(prev,key=lambda r:(r['end'],r['filed']))
        start=(date.fromisoformat(prior['end'])+timedelta(days=1)).isoformat()
        q={**cumulative,'start':start,'val':cumulative['val']-prior['val'],
           'filed':max(cumulative['filed'],prior['filed']),'duration_days':(date.fromisoformat(cumulative['end'])-date.fromisoformat(start)).days+1,
           'source_kind':'cumulative_difference','source_records':cumulative['source_records']+prior['source_records']}
        if q['end'] not in out:out[q['end']]=q
    return sorted(out.values(),key=lambda r:r['end'])


def previous_quarter(rows,current):
    prev=[r for r in rows if r['end']<current['start'] and 1<=(date.fromisoformat(current['start'])-date.fromisoformat(r['end'])).days<=7]
    return max(prev,key=lambda r:r['end']) if prev else None


def trailing_four(rows,end):
    current=next((r for r in reversed(rows) if r['end']==end),None)
    if current is None:return None
    selected=[current]
    for _ in range(3):
        p=previous_quarter(rows,selected[-1])
        if p is None:return None
        selected.append(p)
    return {'val':sum(r['val'] for r in selected),'start':selected[-1]['start'],'end':end,
            'filed':max(r['filed'] for r in selected),'source_records':[s for r in selected for s in r['source_records']]}


def snapshot(parsed,asof):
    flow={f:quarters(parsed[f],asof) for f in FLOW};inst={f:asof_records(parsed[f],asof) for f in TAGS if f not in FLOW}
    revenue=flow['revenue'][-1] if flow['revenue'] else None
    cutoff=date.fromisoformat(asof);latest={}
    for field,rows in inst.items():
        if rows:latest[field]=max(rows,key=lambda r:(r['end'],r['filed']))
    metrics={};sources={}
    if revenue:
        end=revenue['end'];start=revenue['start'];metrics['report_period_end']=end
        metrics['report_age_days']=(cutoff-date.fromisoformat(end)).days;metrics['revenue_quarter']=revenue['val'];sources['revenue_quarter']=revenue
        for field in FLOW-{'revenue'}:
            matched=[r for r in flow[field] if r['start']==start and r['end']==end]
            if matched:
                r=matched[-1];metrics[field+'_quarter']=r['val'];sources[field+'_quarter']=r
                if revenue['val']>0:
                    names={'gross_profit':'gross_margin','operating_income':'operating_margin','net_income':'net_margin','operating_cashflow':'ocf_margin_quarter'}
                    metrics[names[field]]=r['val']/revenue['val']
        prev=previous_quarter(flow['revenue'],revenue)
        if prev and prev['val']>0:metrics['revenue_qoq']=revenue['val']/prev['val']-1;sources['revenue_qoq_prior']=prev
        prior=[r for r in flow['revenue'] if 350<=(date.fromisoformat(end)-date.fromisoformat(r['end'])).days<=380 and abs(r['duration_days']-revenue['duration_days'])<=10]
        if prior:
            prev=max(prior,key=lambda r:r['end'])
            if prev['val']>0:metrics['revenue_yoy']=revenue['val']/prev['val']-1;sources['revenue_yoy_prior']=prev
        for field in FLOW:
            ttm=trailing_four(flow[field],end)
            if ttm:metrics[field+'_ttm']=ttm['val'];sources[field+'_ttm']=ttm
        if metrics.get('revenue_ttm',0)>0 and 'operating_cashflow_ttm' in metrics:
            metrics['ocf_margin_ttm']=metrics['operating_cashflow_ttm']/metrics['revenue_ttm']
    for field,r in latest.items():
        # Stale or mismatched instant data are retained in sources, not mixed into fresh period ratios.
        sources[field]=r
        if revenue and r['end']==revenue['end']:metrics[field]=r['val']
    if metrics.get('current_liabilities',0)>0 and 'current_assets' in metrics:metrics['current_ratio']=metrics['current_assets']/metrics['current_liabilities']
    if metrics.get('assets',0)>0:
        for numerator,name in [('equity','equity_assets_ratio'),('cash','cash_assets_ratio')]:
            if numerator in metrics:metrics[name]=metrics[numerator]/metrics['assets']
    missing_growth='revenue_yoy' not in metrics;missing_ttm='revenue_ttm' not in metrics
    age=metrics.get('report_age_days');fresh=age is not None and age<=183
    checks={'recent_financial_period':fresh,'commercial_revenue':metrics.get('revenue_ttm',-1)>=10_000_000,
            'revenue_yoy_above_10pct':metrics.get('revenue_yoy',-1)>.1,'gross_margin_at_least_20pct':metrics.get('gross_margin',-1)>=.2,
            'current_ratio_at_least_one':metrics.get('current_ratio',-1)>=1,'positive_equity':metrics.get('equity',-1)>0,
            'positive_operating_margin_or_ttm_cashflow':metrics.get('operating_margin',-1)>0 or metrics.get('ocf_margin_ttm',-1)>0}
    if all(checks.values()):quality='growth_and_operating_health_checks_pass'
    elif fresh and metrics.get('revenue_yoy',-1)>.1 and metrics.get('gross_margin',-1)>=.2:quality='growing_but_operating_health_unproven'
    elif not revenue:quality='financial_statement_missing_or_unsupported'
    else:quality='not_passed_or_incomplete'
    vector={f'fin_{k}':metrics.get(k,np.nan) for k in ['revenue_yoy','revenue_qoq','gross_margin','operating_margin','net_margin','ocf_margin_ttm','current_ratio','equity_assets_ratio','cash_assets_ratio','report_age_days']}
    vector['fin_revenue_ttm_log']=math.log1p(max(0,metrics['revenue_ttm'])) if 'revenue_ttm' in metrics else np.nan
    vector.update(fin_statement_missing=float(not revenue),fin_growth_missing=float(missing_growth),fin_ttm_missing=float(missing_ttm),
                  fin_liquidity_missing=float('current_ratio' not in metrics),fin_fresh=float(fresh))
    return {'asof':asof,'quality_status':quality,'quality_checks':checks,'metrics':metrics,'features':vector,'sources':sources}


def issuer_events(path,start,end):
    payload=json.loads(path.read_text());parsed=extract_facts(payload)
    dates=sorted({r['filed'] for rows in parsed.values() for r in rows if r['filed']<=end})
    # Include the latest pre-start disclosure state and every subsequent actual change.
    available=[(date.fromisoformat(s)+timedelta(days=1)).isoformat() for s in dates]
    before=[s for s in available if s<start];keep=([before[-1]] if before else [])+[s for s in available if start<=s<=end]
    rows=[]
    for d in keep:
        snap=snapshot(parsed,d)
        rows.append({'cik':int(payload['cik']),'available_at':pd.Timestamp(d),'quality_status_at_event':snap['quality_status'],
                     'report_period_end':snap['metrics'].get('report_period_end'),**snap['features']})
    return pd.DataFrame(rows)


def prepare_events(start='2025-08-01',end='2026-09-29'):
    sourcehash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();out=CACHE/'financial_events_v2';out.mkdir(exist_ok=True)
    files=sorted((CACHE/'companyfacts').glob('*.json'));manifest=[]
    for i,p in enumerate(files):
        raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest();expected_cik=int(p.stem[3:])
        try:
            payload=json.loads(raw)
            if int(payload.get('cik',0))!=expected_cik or not payload.get('facts'):raise ValueError('Issuer identity or facts missing')
        except (ValueError,TypeError):
            manifest.append({'cik':expected_cik,'raw_sha256':digest,'path':None,'rows':0,'source_valid':False,'status':'invalid_cached_identity_or_facts'})
            print('financial cache invalid',expected_cik,'retained as missing',flush=True);continue
        key=hashlib.sha256(f'{digest}{sourcehash}{start}{end}'.encode()).hexdigest()
        target=out/(p.stem+'_'+key[:16]+'.parquet')
        if not target.exists():
            e=issuer_events(p,start,end)
            if not e.empty:
                temp=target.with_suffix(f'.{os.getpid()}.tmp')
                e.to_parquet(temp,compression='zstd',compression_level=7,index=False);temp.replace(target)
        manifest.append({'cik':expected_cik,'raw_sha256':digest,'path':str(target) if target.exists() else None,'rows':len(pd.read_parquet(target)) if target.exists() else 0,'source_valid':True,'status':'valid' if target.exists() else 'valid_source_no_supported_financial_events'})
        if (i+1)%100==0:print('financial events',i+1,'/',len(files),flush=True)
    m={'start':start,'end':end,'parser_sha256':sourcehash,'available_issuer_count':sum(r['source_valid'] for r in manifest),'invalid_cached_issuer_count':sum(not r['source_valid'] for r in manifest),'issuers':manifest,
       'scope':'coverage changes while raw acquisition runs; not a complete joint-model evaluation snapshot'}
    temp=out/f'manifest.{os.getpid()}.tmp';temp.write_text(json.dumps(m,indent=2));temp.replace(out/'manifest.json');return m


def join_price_features(price,universe,manifest):
    """Forward-fill released financial states only; recompute period age every decision day."""
    events={r['cik']:pd.read_parquet(r['path']).sort_values('available_at') for r in manifest['issuers'] if r['path']}
    out=[]
    for ticker,rows in price.groupby('ticker',sort=False):
        q=rows.copy().sort_values('date');cik=universe.get(ticker,{}).get('cik')
        if cik in events:
            q=pd.merge_asof(q,events[cik].drop(columns=['cik']),left_on='date',right_on='available_at',direction='backward')
            valid=q.report_period_end.notna();q.loc[valid,'fin_report_age_days']=(q.loc[valid,'date']-pd.to_datetime(q.loc[valid,'report_period_end'])).dt.days.astype(float)
            q['fin_fresh']=(q.fin_report_age_days<=183).astype(float)
            for flag in ['fin_statement_missing','fin_growth_missing','fin_ttm_missing','fin_liquidity_missing']:q[flag]=q[flag].fillna(1.)
            q.loc[q.fin_fresh==0,'quality_status_at_event']='stale_or_missing_financial_statement'
        else:
            for f in FEATURES:q[f]=np.nan
            q['fin_statement_missing']=1.;q['fin_growth_missing']=1.;q['fin_ttm_missing']=1.;q['fin_liquidity_missing']=1.;q['fin_fresh']=0.
            q['available_at']=pd.NaT;q['report_period_end']=None;q['quality_status_at_event']='financial_cache_missing'
        out.append(q)
    return pd.concat(out,ignore_index=True)

if __name__=='__main__':
    m=prepare_events();print(json.dumps({k:v for k,v in m.items() if k!='issuers'},indent=2))
