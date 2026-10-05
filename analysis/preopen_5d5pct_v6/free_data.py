"""Free-source audit and resumable read-only collection. Never sends trade requests."""
from __future__ import annotations
import argparse, json, time, urllib.request, urllib.error
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from common import OUT, ROOT, OLD, now, write, save, sha, journal

def local_inventory():
    rows = []
    for name in ['us_5m', 'us_60m', 'us_60m_raw', 'corporate_actions']:
        paths = sorted((ROOT/'market_data'/name).glob('**/*.parquet'))
        rows.append(dict(source='local:'+name, files=len(paths), symbols=len({p.parent.name for p in paths}),
                         bytes=sum(p.stat().st_size for p in paths),
                         schema=pq.read_schema(paths[0]).names if paths else [],
                         price_basis='QFQ' if name=='us_60m' else 'NONE' if 'us_' in name else 'events'))
    coverage = json.loads((OLD/'raw_coverage_audit.json').read_text())
    write(OUT/'local_inventory.json', dict(observed_at=now(), requested=['2024-10-04','2026-09-30'],
         sources=rows, existing_minute_registry=coverage,
         feature_panel=dict(rows=pq.ParquetFile(OLD/'weekly_v1/panel.parquet').metadata.num_rows,
                            sha256=sha(OLD/'weekly_v1/panel.parquet')),
         membership='current_snapshot_retrospective_not_PIT'))

def r2_inventory():
    from scripts.r2_client import R2Client
    client = R2Client(); report=[]
    for prefix in ['us_5m/', 'us_60m_raw/', 'corporate_actions/', 'model_training_history_v1/parts/']:
        try:
            items=client.list_objects(prefix)
            report.append(dict(prefix=prefix, files=len(items), bytes=sum(x['size'] for x in items),
                               status='read_only_inventory', objects=items))
        except Exception as e: report.append(dict(prefix=prefix, status='unavailable', error=type(e).__name__))
    write(OUT/'cache/r2_inventory.json', dict(at=now(), sources=report))
    write(OUT/'r2_summary.json',dict(at=now(),sources=[{k:v for k,v in x.items() if k!='objects'} for x in report]))

def records(data):
    return data.to_dict('records') if isinstance(data,pd.DataFrame) else {'error':str(data)[:240]}

def futu_collect(start, end, actions=True):
    import futu as ft
    ft.SysConfig.enable_proto_encrypt(False)
    q=ft.OpenQuoteContext(host='127.0.0.1',port=11111)
    try:
        ret, groups=q.get_user_security_group(); group_rows=records(groups); time.sleep(3.2)
        snapshots=[]
        if ret==ft.RET_OK:
            for name in groups.group_name:
                key=__import__('hashlib').sha256(str(name).encode()).hexdigest()[:16]
                path=OUT/'cache/watchlists'/f'{key}.json'
                if path.exists():snapshots.append(json.loads(path.read_text()));continue
                rr, stocks=q.get_user_security(group_name=str(name))
                item=dict(group=str(name),received_at=now(),status='ok' if rr==ft.RET_OK else 'unavailable',members=records(stocks))
                write(path,item);snapshots.append(item);journal('watchlist',dict(group=str(name),status=item['status']));time.sleep(3.2)
        write(OUT/'watchlist_snapshot.json',dict(observed_at=now(),groups=group_rows,snapshots=snapshots,membership='current_not_PIT'))
        ret,quota=q.get_history_kl_quota(get_detail=False)
        write(OUT/'quota_audit.json',dict(at=now(),status='ok' if ret==ft.RET_OK else 'unavailable',value=records(quota) if isinstance(quota,pd.DataFrame) else quota));time.sleep(3.2)
        errors=0; d=date.fromisoformat(start); finish=date.fromisoformat(end); summaries=[]
        while d<=finish:
            last=min(d+timedelta(days=6),finish);path=OUT/'cache/earnings'/f'{d}_{last}.json'
            if path.exists():item=json.loads(path.read_text())
            else:
                ret,data=q.get_earnings_calendar(market=ft.Market.US,begin_date=str(d),end_date=str(last))
                item=dict(start=str(d),end=str(last),received_at=now(),status='ok' if ret==ft.RET_OK else 'unavailable',records=records(data),
                    historical_consensus_PIT=False, revision_history='not_provided', historical_available_at=None)
                write(path,item);journal('earnings',dict(start=str(d),status=item['status'],rows=len(data) if isinstance(data,pd.DataFrame) else 0));time.sleep(3.2)
            errors=errors+1 if item['status']!='ok' else 0
            summaries.append(dict(start=str(d),end=str(last),status=item['status'],rows=len(item['records']) if isinstance(item['records'],list) else 0))
            if errors>=3:break
            d=last+timedelta(days=1)
        write(OUT/'earnings_audit.json',dict(at=now(),requested=[start,end],weeks=summaries,
            historical_PIT=False,training_eligibility='no_consensus_surprise_without_original_versions',
            reason='release timestamp is not evidence of the originally available consensus or later revisions'))
        if actions:
            symbols=[r['symbol'] for r in json.loads((OLD/'horizon_v1/coverage.json').read_text())['symbols']]
            summary=[]
            for s in symbols:
                path=OUT/'cache/corporate_actions'/f'{s}.parquet';meta=path.with_suffix('.json')
                if meta.exists():summary.append(json.loads(meta.read_text()));continue
                ret,data=q.get_rehab('US.'+s)
                item=dict(symbol=s,received_at=now(),status='ok' if ret==ft.RET_OK else 'unavailable',rows=len(data) if isinstance(data,pd.DataFrame) else 0,
                          announcement_versions=False)
                if ret==ft.RET_OK:save(data,path);item.update(sha256=sha(path),fields=list(data))
                else:item['error']=str(data)[:160]
                write(meta,item);summary.append(item);journal('corporate_actions',item);time.sleep(3.2)
            write(OUT/'corporate_actions_audit.json',dict(at=now(),symbols=summary,
                usage='quality audit; R00 legacy labels immutable, changed labels require new registered version'))
        sample=['US.'+s for s in ['ALAB','AMD','MRVL','TER','TXG']]
        ret,data=q.get_market_snapshot(sample)
        write(OUT/'cache/quote_snapshot.json',dict(received_at=now(),status='ok' if ret==ft.RET_OK else 'unavailable',records=records(data),historical_PIT=False))
    finally:q.close()

def sec_collect():
    def get(url):
        req=urllib.request.Request(url,headers={'User-Agent':'StockResearch local public-data audit','Accept-Encoding':'identity'})
        with urllib.request.urlopen(req,timeout=30) as r:return json.load(r)
    source='https://www.sec.gov/files/company_tickers.json';summary=[]
    try:
        path=OUT/'cache/sec/company_tickers.json'
        if path.exists():tickers=json.loads(path.read_text())
        else:tickers=get(source);write(path,tickers)
    except Exception as e:
        # The repository retains official EDGAR CIKs from an earlier retrieval.
        # data.sec.gov is a distinct documented public API, not a proxy for the
        # blocked index. Verify each returned company's ticker before using it.
        archived=ROOT/'data/ai_sec_annual_facts.json'
        if not archived.exists():
            write(OUT/'sec_audit.json',dict(at=now(),status='unavailable',source=source,error=type(e).__name__,http_status=getattr(e,'code',None)))
            return
        prior=json.loads(archived.read_text())
        tickers={s:dict(ticker=s,cik_str=r['cik']) for s,r in prior['companies'].items()}
        source='local archived official CIKs: data/ai_sec_annual_facts.json'
    lookup={r['ticker']:r['cik_str'] for r in tickers.values()}
    overrides=OUT/'sec_cik_overrides.json'
    if overrides.exists():lookup.update({s:r['cik'] for s,r in json.loads(overrides.read_text()).items()})
    symbols=[r['symbol'] for r in json.loads((OLD/'horizon_v1/coverage.json').read_text())['symbols']]
    rows=[];errors=0
    for symbol in symbols:
        cik=lookup.get(symbol)
        if cik is None:summary.append(dict(symbol=symbol,status='no_cik'));continue
        url=f'https://data.sec.gov/submissions/CIK{int(cik):010d}.json';p=OUT/'cache/sec'/f'{symbol}.json'
        try:
            receipt=p.with_suffix('.receipt.json')
            if p.exists():data=json.loads(p.read_text())
            else:
                data=get(url);write(p,data);write(receipt,dict(received_at=now(),source=url,sha256=sha(p)));time.sleep(.25)
            received_at=json.loads(receipt.read_text())['received_at'] if receipt.exists() else None
            if symbol not in data.get('tickers',[]):
                summary.append(dict(symbol=symbol,status='ticker_CIK_mismatch',returned_tickers=data.get('tickers',[])));continue
            filings=[data['filings']['recent']]
            for older in data['filings'].get('files',[]):
                if older['filingTo']<'2024-10-04' or older['filingFrom']>'2026-09-30':continue
                archive=OUT/'cache/sec'/older['name']
                if archive.exists():filings.append(json.loads(archive.read_text()))
                else:
                    payload=get('https://data.sec.gov/submissions/'+older['name']);write(archive,payload);filings.append(payload);time.sleep(.25)
            count=0
            for filing in filings:
              for i, form in enumerate(filing['form']):
                accepted=filing.get('acceptanceDateTime',['']*len(filing['form']))[i]
                day=filing['filingDate'][i]
                if form not in ['8-K','10-Q','10-K','6-K','20-F'] or day<'2024-10-04' or day>'2026-09-30':continue
                rows.append(dict(symbol=symbol,cik=int(cik),form=form,items=filing.get('items',['']*len(filing['form']))[i],
                    accepted_at=accepted,filing_date=day,accession=filing['accessionNumber'][i],primary_document=filing['primaryDocument'][i],
                    available_at_assumption='acceptance+60s',received_at=received_at,cache_observed_at=now(),revision_policy='original accession; amendment distinct'))
                count+=1
            summary.append(dict(symbol=symbol,status='ok',recent_events=count,has_older_files=bool(data['filings'].get('files'))));errors=0
        except Exception as e:
            summary.append(dict(symbol=symbol,status='unavailable',error=type(e).__name__));errors+=1
            if errors>=3:break
    write(OUT/'cache/sec/events.json',dict(events=rows,received_at=now()))
    write(OUT/'sec_audit.json',dict(at=now(),status='partial',source=source,symbols=summary,events=len(rows),
        limitation='filing acceptance is conservative event time, not earnings publication time; no analyst consensus'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--futu',action='store_true');ap.add_argument('--sec',action='store_true');ap.add_argument('--r2',action='store_true')
    ap.add_argument('--start',default='2024-10-04');ap.add_argument('--end',default='2026-10-04');a=ap.parse_args()
    local_inventory()
    if a.r2:r2_inventory()
    if a.sec:sec_collect()
    if a.futu:futu_collect(a.start,a.end)
if __name__=='__main__':main()
