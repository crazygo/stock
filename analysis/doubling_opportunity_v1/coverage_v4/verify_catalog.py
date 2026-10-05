#!/usr/bin/env python3
"""Recompute full real directory conservation and exact official CIK mapping."""
import csv,io,json
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from catalog import sha,classify
from context import CACHE,HERE

def require(ok,message):
    if not ok:raise ValueError(message)

def main():
    u=json.loads((CACHE/'universe.json').read_text());m=u['metadata'];raw=(CACHE/'nasdaqtraded.txt').read_text()
    require(sha(CACHE/'nasdaqtraded.txt')==m['raw_sha256'],'Directory source changed')
    require(sha(m['sec_map_path'])==m['sec_map_sha256'],'Official derived ticker source changed')
    require(sha(m['source_extraction_evidence_path'])==m['source_extraction_evidence_sha256'],'Source extraction windows changed')
    require(sha(HERE/'catalog.py')==m['catalog_code_sha256'] and sha(HERE/'PROTOCOL.md')==m['protocol_sha256'],'Catalog code/protocol changed')
    mapping=defaultdict(set)
    for cik,name,ticker,exchange in json.loads(Path(m['sec_map_path']).read_text())['data']:mapping[ticker.replace('-','.')].add(cik)
    rows={x['Symbol']:x for x in csv.DictReader(io.StringIO(raw),delimiter='|') if x.get('Symbol') and not x['Symbol'].startswith('File Creation')}
    excluded={x['ticker']:x for x in u['exclusions']};review={x['ticker']:x for x in u['review']};stocks=u['stocks']
    require(set(rows)==set(stocks)|set(excluded),'A directory row disappeared')
    require(not set(stocks)&set(excluded) and set(review)<=set(stocks),'Category sets overlap incorrectly')
    confirmed=0;mapped=0
    for t,row in rows.items():
        ciks=mapping[t];category,reason=classify(row,len(ciks)==1);item=stocks.get(t) or excluded[t]
        require(item['category']==category and item['classification_reason']==reason,'Classification changed: '+t)
        require(item['security_description_confirmed']==(category=='candidate'),'Uncertain security promoted: '+t)
        require((t in review)==(category=='review'),'Review row omitted: '+t)
        require(('cik' in item)==(len(ciks)==1),'CIK guessed or lost: '+t)
        if len(ciks)==1:require(item['cik']==next(iter(ciks)),'CIK source mismatch: '+t)
        if t in stocks:confirmed+=category=='candidate';mapped+='cik' in item
    require(confirmed+len(review)+len(excluded)==len(rows)==m['directory_count'],'Full source conservation failed')
    require((confirmed,len(review),len(excluded),len(stocks))==(m['explicit_equity_count'],m['review_count'],m['excluded_count'],m['accepted_count']),'Metadata counts differ')
    report={'verified_at':datetime.now(timezone.utc).isoformat(),'catalog_key':json.loads((CACHE/'catalog_latest.json').read_text())['key'],
        'directory_rows_recomputed':len(rows),'explicit_equities':confirmed,'review_price_watch_securities':len(review),
        'excluded_rows':len(excluded),'watch_pool':len(stocks),'watch_tickers_with_unique_exact_CIK':mapped,
        'conservation_passed':True,'current_not_PIT':True,'independent_effect_verified':False,
        'source_sha256':m['raw_sha256'],'SEC_mapping_sha256':m['sec_map_sha256'],'verifier_sha256':sha(Path(__file__))}
    (HERE/'CATALOG_VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
